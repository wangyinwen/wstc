# VLM + 深度辅助监督的 GradNorm 权重平衡方案
## ms-swift + Qwen3-VL 实现版（完整教学）

## 0. 阅读者须知

这份材料给"另一个 LLM"看，目的是让它在已有的 **ms-swift + Qwen3-VL** 训练框架上**正确实现 GradNorm 简化版**，自动平衡 VLM 主任务和深度辅助任务的 loss 权重，并完成 TensorBoard 可视化。

材料中所有架构假设、超参起手值、坑点提示都基于以下场景：

- **训练框架**：[ms-swift](https://github.com/modelscope/ms-swift)（基于 HF Trainer 的 `Seq2SeqTrainer`）
- **模型**：Qwen3-VL（HuggingFace 实现，`Qwen3VLForConditionalGeneration`）
- **辅助任务接入点**：深度 head 接在 LLM 输出端的 **visual tokens 隐层** 之后

如果训练框架、模型或接入点不一样，文末"架构差异处理"一节给出了调整方向。

---

## 1. 问题设定

### 1.1 模型结构（Qwen3-VL）

```
Qwen3VLForConditionalGeneration
├── visual (Qwen3VisionTransformer)
│   ├── blocks: ModuleList[Qwen3VLVisionBlock]    ← ViT blocks
│   └── merger (PatchMerger)                       ← 2×2 spatial merge,出 visual tokens
└── model (Qwen3Model)                             ← LLM 主体
    └── layers: ModuleList[Qwen3DecoderLayer]     ← LLM blocks
```

数据流：

```
Image → visual.blocks → visual.merger → visual tokens ─┐
                                                        ├──> 与 text tokens 拼接,过 model.layers
text  ───────────────────────────────────────────────────┘
                                                                          │
                                                                          ▼
                                  model.layers 最后层 hidden states
                                          │
                                          ├──> 全部 tokens → LM head → next-token logits  (L_vlm)
                                          │
                                          └──> 仅 visual tokens 部分 → Depth Head → depth map (L_depth)
```

### 1.2 损失定义

总 loss：

```
L_total = λ_vlm * L_vlm + λ_depth * L_depth
```

- `L_vlm`：标准语言建模 cross-entropy
- `L_depth`：深度估计 loss（推荐 SiLog，或 depth 归一化到 [0,1] 后的 L1）
- `λ_vlm`, `λ_depth`：**动态调整**的权重（GradNorm 调控）

### 1.3 目标

让两个任务在**共享参数（LLM 最后 2 层）**上的梯度范数趋于平衡，避免 depth 辅助任务压制主任务或反过来。

---

## 2. 架构上的关键风险（必须知道）

由于 depth head 接在 LLM 输出端，存在 4 个特殊风险，**实现前必须处理**：

### 2.1 文本 prompt 污染监督信号
- LLM 输出的 visual token 隐层依赖**输入 prompt**
- 但 depth ground truth 只和图像有关，与 prompt 无关
- **处理**：训练 depth loss 时使用**固定 prompt**（例如 `"<image>"` 或一个统一的占位语句）。每个 depth supervision step 都用同一个 prompt，保证监督信号一致。

### 2.2 空间结构在 LLM 内部丢失
- LLM 是 1D 因果序列模型，visual tokens 失去 2D 网格结构
- Qwen3-VL 的 merger 已经做了 2×2 spatial 合并，LLM 看到的视觉网格是 `(H//2, W//2)`
- **处理**：Depth head 内部必须根据 `image_grid_thw` 把 visual tokens reshape 回 2D，再用卷积上采样 decoder。**不能用简单 MLP**。

### 2.3 反传链路过长，LM 能力易漂移
- depth 梯度反传穿过整个 LLM + merger + ViT
- **处理**：初始 `λ_depth` 取**比常规更小**的起点（全量 0.1 / LoRA 0.05），让 LLM 先稳定再让 depth 介入。

### 2.4 ViT / merger 可能被间接污染
- 即便 GradNorm 在 LLM 层调控，梯度仍会传到下游
- **处理**：TensorBoard 上**额外监控 ViT 最后层（或 merger）的梯度范数**（只看不调），用于诊断。

---

## 3. GradNorm 简化版原理

### 3.1 核心思想

> **测每个任务对共享参数施加的"梯度力度"，谁太猛就压一压，谁太弱就抬一抬，让两个任务用力相当。**

### 3.2 三步循环（每 N 个 optimizer step 执行一次）

**Step 1：测力度**

对每个任务，单独算它的**加权 loss**在共享参数上的 L2 梯度范数：

```
g_i = || ∇_shared (λ_i * L_i) ||_2
```

**Step 2：定目标**

```
g_avg = mean(g_vlm, g_depth)
```

**Step 3：调权重**

```
λ_i ← λ_i * (g_avg / g_i) ** α
```

`α` 控制调整激进度，**推荐 α = 0.5**（温和、抗噪）。

### 3.3 防漂移：归一化

更新后做一次归一化，保持总权重恒定：

```
λ_vlm, λ_depth ← 归一化使得 λ_vlm + λ_depth = 2
```

否则两个 λ 可能一起放大或缩小，破坏学习率假设。

---

## 4. 共享参数的选择（关键决策）

### 4.1 选 LLM 最后 2 层，不是 ViT

**理由**：
- depth head 和 LM head 都从 **LLM 最后层 hidden states** 读特征 → 这是两个 loss 第一次"碰头"的地方
- ViT / merger 的梯度信号已经被 LLM 衰减/混合，失真严重
- 监控 LLM 最后层 = 监控真实冲突点

### 4.2 选 2 层不是 1 层

- 最后 1 层有"特殊性"（接近 LM head，分布偏），单点采样噪声大
- 最后 2 层降噪、稳定、计算成本可控

### 4.3 具体取哪些参数

**全量微调**：最后 2 个 transformer block 内的
- Attention Q/K/V/O 投影矩阵
- MLP 的 up/down/gate 投影矩阵
- **排除** LayerNorm、bias、embedding

**LoRA 微调**（ms-swift 常用）：最后 2 个 block 内的
- **只取 LoRA A/B 矩阵**（参数名含 `lora_`）
- base 权重已冻结，没梯度，取了也没意义

### 4.4 同时监控 ViT 或 merger（诊断用）

把诊断梯度范数也 log 出来，**只看不调**。具体取哪个：

| ViT 配置 | 取哪里做诊断 |
|---|---|
| ViT 可训练 | `visual.blocks` 的最后 1 层 |
| ViT 冻结（`freeze_vit=true`） | `visual.merger` 的所有可训练参数 |

---

## 5. ms-swift 框架集成要点

### 5.1 不要写独立训练循环
ms-swift 用 `Seq2SeqTrainer`（继承自 HF Trainer），应该通过以下方式介入：

| 介入点 | 做什么 |
|---|---|
| 自定义 Trainer 子类，重写 `compute_loss` | 算两个 loss 并加权 |
| 自定义 `TrainerCallback` | 周期性更新 λ + log |
| 用 `self.log({...})` | 内置同步到 TensorBoard/WandB/SwanLab |

### 5.2 关键启动参数与 GradNorm 的相互影响

| ms-swift 参数 | 对 GradNorm 的影响 | 推荐处理 |
|---|---|---|
| `--gradient_accumulation_steps N` | 必须用 **optimizer step** 计数 | 使用 `state.global_step`（HF Trainer 自动按 optimizer step 计） |
| `--torch_dtype bfloat16` | bf16 数值范围窄，范数会失真 | 范数计算强制 `.float()` 转 fp32 |
| `--gradient_checkpointing true` | **和 `retain_graph=True` 直接冲突** | 二选一：关掉 checkpointing，或者用"独立 probe batch 两次 backward"方案（见 6.3） |
| `--deepspeed zero3.json` / FSDP | 参数被分片，单卡梯度不完整 | 范数计算后 `dist.all_reduce` 聚合 |
| `--train_type lora` | shared params 变成 LoRA adapter | 重新校准 `λ_depth` 初值（用 0.05） |
| `--freeze_vit true` | ViT 端无梯度，无法诊断 | 诊断切换到 `visual.merger` |

### 5.3 数据格式分流
ms-swift 用 messages-list 格式。需要在 dataset / collator 里加 `task` 字段区分：

```python
# VLM 样本(自由 prompt)
{
    "messages": [
        {"role": "user", "content": "<image>What's in the image?"},
        {"role": "assistant", "content": "..."}
    ],
    "images": [...],
    "task": "vlm",
}

# Depth 样本(固定 prompt!)
{
    "messages": [
        {"role": "user", "content": "<image>"}   # 固定占位
    ],
    "images": [...],
    "depth_gt": ...,
    "task": "depth",
}
```

`compute_loss` 根据 `task` 选走哪条路径。

---

## 6. 完整实现伪代码

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from swift.trainers import Seq2SeqTrainer
from transformers import TrainerCallback


# ============================================================
# 6.1 工具函数:剥包装、找共享参数、找诊断参数
# ============================================================
def unwrap_model(model):
    """剥掉 DDP / FSDP / PeftModel 包装,拿到真正的 Qwen3VLForConditionalGeneration"""
    while True:
        if hasattr(model, "module"):  # DDP / FSDP
            model = model.module
        elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            model = model.base_model.model  # PeftModel
        else:
            break
    return model


def get_shared_params_llm(model, last_n=2, use_lora=False):
    """
    GradNorm 调控用的共享参数:Qwen3 LLM 最后 last_n 个 block 的
    attention 和 MLP 主干权重(全量)或 LoRA adapter(LoRA)。
    """
    base = unwrap_model(model)
    layers = base.model.layers   # Qwen3-VL 的 LLM 路径
    total = len(layers)
    shared = []
    for idx in range(total - last_n, total):
        for name, p in layers[idx].named_parameters():
            if not p.requires_grad:
                continue
            if use_lora:
                # LoRA 训练时只取 adapter 权重
                if "lora_" in name:
                    shared.append(p)
            else:
                # 全量微调时取 attention / MLP 主权重
                if "weight" not in name:
                    continue
                if "norm" in name.lower():
                    continue
                shared.append(p)
    return shared


def get_diag_params(model, target="vit", last_n=1):
    """
    诊断参数:只 log 不调。
    target='vit': 取 visual.blocks 最后 last_n 层(ViT 可训练时用)
    target='merger': 取 visual.merger 全部可训练参数(ViT 冻结时用)
    """
    base = unwrap_model(model)
    params = []
    if target == "vit":
        layers = base.visual.blocks
        total = len(layers)
        for idx in range(total - last_n, total):
            for name, p in layers[idx].named_parameters():
                if p.requires_grad and "weight" in name and "norm" not in name.lower():
                    params.append(p)
    elif target == "merger":
        for p in base.visual.merger.parameters():
            if p.requires_grad:
                params.append(p)
    return params


# ============================================================
# 6.2 梯度范数计算(fp32 + 多卡聚合 + 不污染 .grad)
# ============================================================
def compute_grad_norm(loss, params, retain_graph=False):
    """
    用 torch.autograd.grad 单独算 loss 在 params 上的 L2 范数。
    - 不写入 .grad,避免污染 optimizer
    - fp32 累加,避免 bf16 失真
    - 分布式聚合,适配 DeepSpeed Zero3 / FSDP
    """
    if not params:
        return torch.zeros(1, device=loss.device)

    grads = torch.autograd.grad(
        loss.float(), params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    total = torch.zeros(1, dtype=torch.float32, device=loss.device)
    for g in grads:
        if g is not None:
            total = total + g.detach().float().pow(2).sum()

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total.sqrt()


# ============================================================
# 6.3 Depth Head(处理 Qwen3-VL 动态分辨率)
# ============================================================
class Qwen3VLDepthHead(nn.Module):
    """
    输入:LLM 最后层的 visual tokens hidden states
    输出:每张图独立的 depth map
    关键:用 image_grid_thw 还原 2D 网格,处理 merger 的 2x2 下采样
    """
    def __init__(self, hidden_dim, out_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 8, out_channels, 1),
        )

    def forward(self, visual_hidden, image_grid_thw):
        """
        visual_hidden: (sum_of_visual_tokens, D)
        image_grid_thw: (N_images, 3) -> (T, H, W) per image (ViT patch grid)
        返回 list of depth maps,每张图一个
        """
        outputs = []
        offset = 0
        for (t, h, w) in image_grid_thw.tolist():
            # merger 做了 2x2 spatial 合并
            h_m, w_m = h // 2, w // 2
            num_tok = t * h_m * w_m
            tokens = visual_hidden[offset:offset + num_tok]
            tokens_2d = tokens.reshape(t, h_m, w_m, -1).permute(0, 3, 1, 2)  # (T, D, H', W')
            depth = self.decoder(tokens_2d)  # (T, 1, H'*4, W'*4)
            outputs.append(depth)
            offset += num_tok
        return outputs


# ============================================================
# 6.4 自定义 Trainer
# ============================================================
class GradNormVLMTrainer(Seq2SeqTrainer):
    def __init__(self, *args, gradnorm_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = gradnorm_cfg or {}
        device = self.args.device

        # 关键:depth 起手 lambda 比 vlm 小
        # 全量微调 0.1,LoRA 微调 0.05
        init_depth = 0.05 if cfg.get("use_lora", False) else 0.1
        self.lambda_vlm = torch.tensor(1.0, device=device)
        self.lambda_depth = torch.tensor(init_depth, device=device)

        self.shared_params = get_shared_params_llm(
            self.model, last_n=2,
            use_lora=cfg.get("use_lora", False)
        )
        # ViT 冻结时切换到 merger 做诊断
        diag_target = "merger" if cfg.get("freeze_vit", False) else "vit"
        self.diag_params = get_diag_params(self.model, target=diag_target)

        # 深度 head 单独保留以便 Callback 调用
        self.depth_head = cfg["depth_head"]
        self.image_token_id = cfg["image_token_id"]
        self.fixed_prompt = cfg.get("fixed_prompt_for_depth", "<image>")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        task = inputs.pop("task", "vlm")
        # Qwen3-VL forward 必须 output_hidden_states=True 才能取到隐层
        if task == "vlm":
            outputs = model(**inputs)
            L = outputs.loss
            loss = self.lambda_vlm * L
            self._last_L_vlm = L.detach()
            self._last_inputs_vlm = inputs
        elif task == "depth":
            inputs["output_hidden_states"] = True
            outputs = model(**inputs)
            L = self._compute_depth_loss(outputs, inputs)
            loss = self.lambda_depth * L
            self._last_L_depth = L.detach()
            self._last_inputs_depth = inputs
        return (loss, outputs) if return_outputs else loss

    def _compute_depth_loss(self, outputs, inputs):
        """
        outputs.hidden_states[-1]: (B, L, D)
        从 input_ids == image_token_id 的位置取 visual tokens
        """
        hidden = outputs.hidden_states[-1]
        input_ids = inputs["input_ids"]
        mask = (input_ids == self.image_token_id)
        visual_hidden = hidden[mask]                          # (sum_visual_tokens, D)
        image_grid_thw = inputs["image_grid_thw"]             # Qwen3-VL 标配字段

        pred_depths = self.depth_head(visual_hidden, image_grid_thw)
        gt_depths = inputs["depth_gt"]                        # list of GT
        # SiLog 或归一化 L1
        return self._silog_loss(pred_depths, gt_depths)


# ============================================================
# 6.5 GradNorm Callback(周期更新 lambda + log)
# ============================================================
class GradNormCallback(TrainerCallback):
    """
    每 update_every 个 optimizer step 触发一次:
    用独立 probe batch 做两次小 backward,避开 gradient_checkpointing 冲突
    """
    def __init__(self, trainer, probe_loader_vlm, probe_loader_depth,
                 update_every=100, alpha=0.5):
        self.trainer = trainer
        self.probe_vlm = iter(probe_loader_vlm)
        self.probe_depth = iter(probe_loader_depth)
        self.update_every = update_every
        self.alpha = alpha
        self.eps = 1e-8

    def _next(self, it):
        try:
            return next(it)
        except StopIteration:
            return None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.update_every != 0:
            return

        t = self.trainer
        model = t.model
        model.eval()  # 探针阶段关 dropout 等

        # --- 1. probe L_vlm 的梯度范数 ---
        batch_v = self._next(self.probe_vlm)
        batch_v = {k: v.to(args.device) for k, v in batch_v.items()}
        batch_v.pop("task", None)
        out_v = model(**batch_v)
        g_vlm = compute_grad_norm(t.lambda_vlm * out_v.loss, t.shared_params)
        g_vlm_diag = compute_grad_norm(t.lambda_vlm * out_v.loss, t.diag_params)

        # --- 2. probe L_depth 的梯度范数 ---
        batch_d = self._next(self.probe_depth)
        batch_d = {k: v.to(args.device) for k, v in batch_d.items()}
        batch_d["output_hidden_states"] = True
        batch_d.pop("task", None)
        out_d = model(**batch_d)
        L_depth = t._compute_depth_loss(out_d, batch_d)
        g_depth = compute_grad_norm(t.lambda_depth * L_depth, t.shared_params)
        g_depth_diag = compute_grad_norm(t.lambda_depth * L_depth, t.diag_params)

        model.train()

        # --- 3. 更新 lambda ---
        g_avg = (g_vlm + g_depth) / 2.0
        new_v = t.lambda_vlm * (g_avg / (g_vlm + self.eps)) ** self.alpha
        new_d = t.lambda_depth * (g_avg / (g_depth + self.eps)) ** self.alpha
        total = new_v + new_d
        t.lambda_vlm = (new_v * 2.0 / total).detach()
        t.lambda_depth = (new_d * 2.0 / total).detach()

        # --- 4. log(HF Trainer 的 log 自动路由到 TB/WandB/SwanLab) ---
        t.log({
            "gradnorm_main/g_vlm": g_vlm.item(),
            "gradnorm_main/g_depth": g_depth.item(),
            "gradnorm_main/ratio_d_over_v": (g_depth / (g_vlm + self.eps)).item(),
            "gradnorm_main/g_avg": g_avg.item(),

            "diag/g_vlm": g_vlm_diag.item(),
            "diag/g_depth": g_depth_diag.item(),
            "diag/ratio_d_over_v": (g_depth_diag / (g_vlm_diag + self.eps)).item(),

            "lambda/vlm": t.lambda_vlm.item(),
            "lambda/depth": t.lambda_depth.item(),
        })

    # 每个常规 step 也 log raw loss
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        t = self.trainer
        extra = {}
        if hasattr(t, "_last_L_vlm"):
            extra["loss/vlm_raw"] = t._last_L_vlm.item()
            extra["loss/vlm_weighted"] = (t.lambda_vlm * t._last_L_vlm).item()
        if hasattr(t, "_last_L_depth"):
            extra["loss/depth_raw"] = t._last_L_depth.item()
            extra["loss/depth_weighted"] = (t.lambda_depth * t._last_L_depth).item()
        logs.update(extra)
```

---

## 7. TensorBoard 指标含义与诊断价值

### 7.1 `gradnorm_main/*`（主调控指标，最重要）
| 标量 | 看什么 |
|---|---|
| `g_vlm`, `g_depth` | 在 **LLM 最后 2 层**上各自的梯度力度 |
| **`ratio_d_over_v`** | **最关键单指标**。健康区间 `[0.3, 1.5]`；> 2 报警；< 0.1 说明 depth 没起作用 |
| `g_avg` | 调整目标参考线 |

### 7.2 `diag/*`（ViT 或 merger 污染诊断，只看不调）
| 标量 | 看什么 |
|---|---|
| `g_vlm`, `g_depth` | 诊断层（ViT 或 merger）的梯度力度 |
| `ratio_d_over_v` | 如果**这里**远大于主调控端的比 → depth 梯度在下游有放大效应，**有污染风险** |

### 7.3 `lambda/*`（权重轨迹）
| 标量 | 看什么 |
|---|---|
| `lambda/vlm`, `lambda/depth` | 是否平稳收敛到某个比例。`λ_depth` 应该比 `λ_vlm` 小 |

### 7.4 `loss/*`（原始与加权 loss）
| 标量 | 看什么 |
|---|---|
| `vlm_raw`, `depth_raw` | 主任务和辅助任务原始下降曲线 |
| `vlm_weighted`, `depth_weighted` | 实际进入 backward 的贡献；两者量级应接近 |

---

## 8. 健康度判读规则

按从最重要到次要的顺序：

### 8.1 第一优先：主任务下游 benchmark
**唯一最终判据**。哪怕 GradNorm 一切看着完美，只要 VLM benchmark（MMBench/MMVet/SEED-Bench 等）掉点，方案就失败。在 ms-swift 里通过 `eval_dataset` + `evaluation_strategy` 配置自动跑。

### 8.2 第二优先：`gradnorm_main/ratio_d_over_v`
- 区间 `[0.3, 1.5]` → 健康
- `> 2` 持续 → depth 在压制 VLM，降低 `λ_depth` 起手值或调大 `α`
- `< 0.1` 持续 → depth 没起作用，要么 head 设计有问题，要么任务本身无信号

### 8.3 第三优先：`diag/ratio_d_over_v` vs `gradnorm_main/ratio_d_over_v`
- 两者比例接近 → 下游没被特别污染
- diag 端比例**显著大于** main 端 → depth 梯度在下游有放大效应
- 这种情况即使 main 端看着平衡，VLM benchmark 也可能掉点

### 8.4 第四优先：`loss/vlm_raw` 单调下降趋势
- 平稳下降 → 健康
- 平 / 上升 → 主任务被搞坏（即使其他指标好看）

### 8.5 第五优先：`lambda` 轨迹
- 先剧烈调整（前 1k 步）再稳定 → 健康
- 单调走高 / 走低不收敛 → `α` 太大或 loss 数值差距悬殊

---

## 9. 推荐起手超参（ms-swift + Qwen3-VL）

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `lambda_vlm` 初值 | `1.0` | 标准 |
| `lambda_depth` 初值（全量微调） | **`0.1`** | 反传链路长，起手要小 |
| `lambda_depth` 初值（LoRA） | **`0.05`** | LoRA 梯度量级更小，要更保守 |
| `alpha` | **`0.5`** | 温和、抗噪 |
| `update_every` | `100`（optimizer step） | 不要被 grad_accum 干扰 |
| `last_n`（LLM 共享层） | **`2`** | 最后 2 个 Qwen3 block |
| `last_n`（ViT 诊断层） | `1` | 最后 1 个 ViT block |
| diag 目标（ViT 冻结） | **`merger`** | 改诊断 `visual.merger` |
| depth loss 类型 | SiLog 或归一化 L1 | 避免极端值带飞 GradNorm |
| depth supervision prompt | **固定 `"<image>"`** | 防止 prompt 污染 |
| depth head 结构 | reshape 2D + 卷积上采样 | **不能用 MLP** |
| probe batch size | `2~4` | Callback 探针用，小一点 |

---

## 10. 启动命令模板（ms-swift CLI）

```bash
swift sft \
    --model Qwen/Qwen3-VL-xxx \
    --train_type lora \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing false \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 500 \
    --report_to tensorboard \
    --custom_trainer your_pkg.GradNormVLMTrainer \
    --custom_callbacks your_pkg.GradNormCallback \
    --custom_dataset your_pkg.MixedVLMDepthDataset \
    --output_dir ./output/qwen3vl-gradnorm-depth
```

**关键开关说明**：
- `--gradient_checkpointing false`：用了 probe 方案可以打开，但建议先关掉验证
- `--freeze_vit true`：ViT 冻结是典型起步配置，对应诊断切到 merger
- `--train_type lora`：搭配 `λ_depth=0.05` 和 `use_lora=True`
- `--report_to tensorboard`：ms-swift 自动接好，`self.log({...})` 会同步过去
- `--custom_trainer / --custom_callbacks`：把自定义类注册进 ms-swift 的执行流

---

## 11. 常见坑（必须告诉另一个 LLM）

1. **`gradient_checkpointing` 和 `retain_graph` 不能共存**。要么关掉 checkpointing，要么用独立 probe batch 做两次 backward（推荐后者，见 6.5）。
2. **`update_every` 用 `state.global_step`**，这是 HF Trainer 已经按 optimizer step 计的值，不会被 `gradient_accumulation_steps` 干扰。
3. **`lambda_*` 用 `.detach()`**，不要让 optimizer 把它当参数更新。
4. **梯度范数用 `torch.autograd.grad`**，不要用 `loss.backward()` 后读 `.grad`，避免污染 optimizer 状态。
5. **bf16 下范数必须 `.float()`** 转 fp32 累加，否则数值范围窄会失真。
6. **Zero3 / FSDP 下范数要 `dist.all_reduce` 聚合**，否则只反映单卡分片。
7. **`unwrap_model` 必须正确处理 DDP + PeftModel 双重包装**（LoRA + 多卡时）。
8. **`get_shared_params_llm` 要根据 `use_lora` 切换路径**，LoRA 时只取 `lora_*`。
9. **ViT 冻结时诊断目标必须换成 merger**，否则 `diag_params` 为空，范数计算返回 0。
10. **Qwen3-VL forward 必须 `output_hidden_states=True`** 才能拿到 LLM 隐层。
11. **从 hidden states 取 visual tokens 用 `input_ids == image_token_id` mask**，不要靠位置硬编码（Qwen3-VL 是动态分辨率，每个样本视觉 token 数不同）。
12. **depth head 必须用 `image_grid_thw` 重建 2D 网格**，并记得 merger 已经 2×2 下采样过了。
13. **depth supervision 必须用固定 prompt**，否则信号在 batch 间不一致。
14. **depth loss 数量级**第一次跑要 log 出来看；如果原始值是 100+ 这种，先归一化到 1~10 量级再开 GradNorm，否则起手 `λ_depth = 0.1` 也会出问题。
15. **首次跑训练**先把 `α = 0` 关掉 GradNorm，确认整套 pipeline 跑通、loss 都正常下降，再开启。
16. **`compute_loss` 内 `inputs.pop("task")`** 后必须确保剩下的字段是 model forward 真的接受的，否则 HF Trainer 会报错。
17. **多个数据源混训**：可以用 ms-swift 的 `interleave_datasets` 或自定义 sampler，按一定比例混 VLM 和 depth 样本。

---

## 12. 架构差异处理（如果接入点不一样）

如果 depth head 实际接在别的位置，按下表调整：

| Depth head 接在哪 | shared_params 取哪里 | 诊断 params 取哪里 | λ_depth 初值（全量 / LoRA） |
|---|---|---|---|
| **LLM 最后层 visual tokens**（本文场景） | `model.layers` 最后 2 层 | `visual.blocks` 最后 1 层（或 merger） | 0.1 / 0.05 |
| LLM 中间层 visual tokens | `model.layers` 中那一层附近的 ±1 层 | 同上 | 0.2 / 0.1 |
| Merger 之后、LLM 之前 | `visual.merger` + `visual.blocks` 最后 1 层 | `model.layers` 最后 1 层（次要） | 0.3 / 0.15 |
| ViT 最后层 | `visual.blocks` 最后 2 层 | `model.layers` 最后 1 层（次要） | 0.5 / 0.25 |

其他逻辑（GradNorm 公式、TensorBoard 项、坑点）完全一致。

---

## 13. 给另一个 LLM 的最短 prompt 模板

如果不想贴整篇，最少需要告诉它这些：

> 我用 **ms-swift + Qwen3-VL** 训练，需要加 GradNorm 简化版来自动平衡主任务 loss `L_vlm` 和深度辅助任务 loss `L_depth`。
>
> **架构特点**：depth head 接在 LLM 输出端的 visual tokens 隐层之后，通过 `input_ids == image_token_id` mask 提取，配合 `image_grid_thw` reshape 回 2D 后用卷积上采样得到 depth map。
>
> **请按以下方案改造**：
>
> 1. **集成方式**：继承 `swift.trainers.Seq2SeqTrainer` 写自定义 Trainer，重写 `compute_loss`，按 `inputs['task']` 字段分流 vlm/depth；GradNorm 更新通过自定义 `TrainerCallback.on_step_end` 触发。
> 2. **共享参数**：`unwrap_model(model).model.layers` 的最后 2 个 block 的权重；LoRA 时只取 `lora_*` 参数。
> 3. **诊断参数**：ViT 可训练时取 `visual.blocks[-1:]`；ViT 冻结时取 `visual.merger` 全部可训练参数。
> 4. **更新频率**：每 100 个 `state.global_step`（optimizer step，不是 micro-batch）触发一次。
> 5. **GradNorm 公式**：`λ_i ← λ_i * (g_avg / g_i) ** α`，`α = 0.5`。
> 6. **归一化**：保持 `λ_vlm + λ_depth = 2`。
> 7. **初始权重**：`λ_vlm = 1.0`，`λ_depth = 0.1`（全量）或 `0.05`（LoRA）。
> 8. **固定 prompt**：depth supervision 用 `"<image>"`。
> 9. **梯度计算**：用 `torch.autograd.grad`，`.float()` 转 fp32，多卡 `dist.all_reduce` 聚合；**不要**用 `retain_graph=True`，改用独立 probe batch 做两次小 backward 避开和 `gradient_checkpointing` 的冲突。
> 10. **logging**：用 `Trainer.log({...})`（自动同步到 TB/WandB/SwanLab）。指标：
>     - `gradnorm_main/{g_vlm, g_depth, ratio_d_over_v, g_avg}`（LLM 端，用于调控）
>     - `diag/{g_vlm, g_depth, ratio_d_over_v}`（ViT/merger，只诊断）
>     - `lambda/{vlm, depth}`
>     - `loss/{vlm_raw, depth_raw, vlm_weighted, depth_weighted}`
>
> **实现注意**：
> - `unwrap_model` 要处理 DDP + PeftModel 双层包装
> - Qwen3-VL forward 必须 `output_hidden_states=True`
> - depth head 内要用 `image_grid_thw` 处理动态分辨率，merger 已做 2×2 spatial 下采样
> - depth loss 必须归一化到 1~10 量级
>
> **健康判据**：
> - 主判据：VLM 下游 benchmark 不掉
> - 辅判据：`gradnorm_main/ratio_d_over_v ∈ [0.3, 1.5]`
> - 二次诊断：`diag/ratio_d_over_v` 不应明显大于 main 端

---

## 14. TL;DR（一句话版）

**继承 `Seq2SeqTrainer` 重写 `compute_loss` 按 task 分流 → 在 Qwen3 LLM 最后 2 层（LoRA 取 adapter）上测两个任务的梯度范数 → 按比例调权重 → 归一化保持总和恒定 → 用独立 probe batch 避开 gradient_checkpointing 冲突 → 同时监控 visual.blocks 或 visual.merger 防下游污染 → 固定 prompt + 用 image_grid_thw reshape 的 depth head → 通过 `self.log({...})` 同步到 TensorBoard → 最终用 VLM benchmark 验证。**
