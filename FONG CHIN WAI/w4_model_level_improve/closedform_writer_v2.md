# W4 模型级改写率改善：MEMIT 闭式解写入器（closedform writer v2）

> 目标：把 CALM-Editor 的 `model-level rewrite rate`（真正写进权重、能被模型自己答出来的比例）从超网络路线的 1.6%–3.9% 拉到可用水平。
> 结论：**500 条顺序编辑下真实命中 90.8%（454/500）**，相对超网络基线提升约 **23–57 倍**；代价是流畅度下降与长程保留归零，见「已知限制」。

---

## 1. 指标定义（与旧审计口径一致）

| 指标 | 定义 |
| --- | --- |
| `hit`（rewrite_ok_rate） | 每次编辑**写入后立即**用 5-token greedy 解码该 prompt，答案包含 `target_new` 即算命中。与 W4 前序 128-write 审计的口径完全一致，可直接对比 |
| `degen`（degenerate_rate） | 答案为空 / 含零宽字符 / ≥4 个连续重复 token，即退化输出 |
| `probe NLL` | 12 条固定中性英文散文的平均 token NLL，编辑前 3.5687；衡量顺序写入对语言能力的破坏（越低越好） |
| `recheck` | 流跑完后回查**最早 50 条**编辑是否仍保持，衡量长程保留 |
| `heldout` | 500 条之外未编辑事实的原始正确率，衡量局部性（见 §5 说明，本数据集基线为 0，指标失效） |

> 旧超网络路线：WRITE 真实命中 1.6%–4.0%，另有 22.2% 是 REJECT 假命中（路由到 side memory 被算成写成功）。闭式解写入器全部走参数写入，不存在这类虚高。

---

## 2. 最终配置（推荐）

```
writer            = closedform (MEMIT/AlphaEdit 闭式解)
layers            = 6,7,8          # c_proj 层
l2 / l2_mode      = 1.0 / rel      # lambda = 1.0 * trace(KK^T)/n_ctx
z_clamp           = 0.6            # |z| <= 0.6 * mean|h_z|
resid_ref         = z_layer        # MEMIT 约定：残差在 z_layer 一次测量，按 1/(N-i) 分配
null_space        = none           # AlphaEdit-lite 投影默认关闭（见 §4）
probe_every       = 25
```

复现命令：

```powershell
cd EasyEdit
..\.venv\Scripts\python.exe -u examples\w4_closedform_smoke.py `
  --edits 500 --layers 6,7,8 --l2 1 --l2-mode rel --z-clamp 0.6 `
  --probe-every 25 --locality-n 50 --recheck-n 50
```

---

## 3. 主结果（500 条顺序编辑，同一关系 `mother tongue / religion / capital` 混合流）

| 配置 | hit | degen | probe NLL Δ | recheck(50) |
| --- | --- | --- | --- | --- |
| 超网络基线（W4 前序审计） | 0.016–0.039 | — | — | — |
| **闭式解 L6-8, z=0.6（推荐）** | **0.908** | 0.228 | +3.97（3.57→7.54） | 0/50 |
| 闭式解 L4-8, z=0.6 | 0.900 | 0.144 | +4.83（→8.40） | — |
| 闭式解 + AlphaEdit-lite 投影 (mix 0.5) | 0.778 | 0.082 | +5.13（→8.70） | 0/50 |

分块命中（每 100 条）L6-8：`1.00 / 0.92 / 0.88 / 0.84 / 0.90` —— 命中不随写入次数崩塌，退化率从 0.09 爬到 0.31。

---

## 4. 消融与调参轨迹（全部为真实运行结果）

### 4.1 lambda 的尺度（决定性坑）

早期把 `l2` 当绝对值（MEMIT 字面 `mom2_update_weight`）用：`l2=15000` → hit 0.125、NLL 几乎不动（等于没写）；`l2=10` → hit 0.44 但 degen 0.59、NLL +6.86（写崩）。根因是 in-context key 的 Gram 特征值量级在数百，而 `mean(diag)` 只有 0.2。改成 `lambda = coef * trace(KK^T)/n_ctx`（`--l2-mode rel`）后才有可用区间。

### 4.2 层选择

| layers | n | hit | degen | NLL Δ |
| --- | --- | --- | --- | --- |
| 4-8 | 128 | 0.977 | 0.023 | +4.68 |
| 6-8 | 128 | 1.000 | 0.023 | +2.50 |
| 7-8 | 16 | 1.000 | 0.250 | +5.51 |
| 9,10,11 | — | 0（delta 恒为 0） | — | — |

`9,10,11` 是此前的 bug 源：`compute_z` 的 `v_loss_layer` 必须严格大于 rewrite 层，否则同层 TraceDict 重复注册、delta 无梯度。代码里已加硬校验。

### 4.3 z 范数钳制（第二个决定性参数）

`compute_z` 是无约束 Adam 上升，gpt2-small 上常给出 |z| 300–1400（普通 prompt 的 |h| 约 100），即 off-manifold 目标。按 `z_clamp * mean|h_z|` 缩放后：

| z_clamp | n | hit | degen | NLL Δ |
| --- | --- | --- | --- | --- |
| 不钳制 | 16 | 1.000 | 0.188 | +5.45 |
| 1.0 | 16 | 1.000 | 0.063 | +0.78 |
| 0.6 | 64 | 1.000 | 0.016 | +1.98 |
| 0.6 | 128 | 0.977 | 0.023 | +4.68 |
| 0.5 | 16 | 1.000 | 0.000 | +0.62 |

### 4.4 残差参考层

`resid_ref=cur_layer`（每层各自测当前残差）会让各层都去补"全部剩余误差"，delta 总和约为所需 2–3 倍；改为 MEMIT 官方的 `z_layer` 单次测量 + `1/(N-i)` 分配后才可控。同时修复了 `compute_z` 需要 flat contexts 的问题。

### 4.5 AlphaEdit-lite 零空间投影（自实现，不依赖缺失的 Wikipedia 统计）

官方 AlphaEdit 的 `P` 需要离线 mom2 协方差（仓库 `data/` 下没有，直接调会失败）。这里用已写入过的 key 的累计 Gram 构造等价投影：`P = λ(G+λI)^{-1}`，`G = Σ K_j K_j^T`（即 `kv_cache` 累积量），对 `[d_in, d_out]` 的更新左乘。

| 配置（n=128） | hit | degen | NLL Δ | recheck(32) |
| --- | --- | --- | --- | --- |
| 不投影 | 1.000 | 0.023 | +2.50 | 0 |
| 全投影（mix 1.0, proj λ×4） | 0.820 | 0.000 | +1.27 | 0.156 |
| 半投影（mix 0.5, λ×1） | 0.945 | 0.031 | +1.43 | 0 |
| 半投影（mix 0.5, λ×10） | 0.953 | 0.062 | +1.53 | 0 |

短程（≤128）投影确实把退化压到 0、NLL 减半且拿到部分保留；但拉到 500 条时半投影反而更差（hit 0.778、NLL +5.13），因此**默认仍关闭投影**，只作为可选参数保留（`--null-space keys --proj-mix 0.5`）。

---

## 5. 已知限制（必须写进报告的诚实部分）

1. **长程保留为 0**：500 条同关系顺序编辑跑完，最早 50 条全部失效（`recheck 0/50`）。这是顺序单键闭式解的固有覆盖问题，不是实现 bug——`hit` 是"写入后立即"指标。缓解方向：批量联合求解（一次解一个 batch 的键矩阵）、或把参数写入与 side memory 双写。
2. **流畅度代价**：500 条后中性散文 NLL 3.57→7.54，退化输出 22.8%。128 条以内可接受（+2.5、2.3%），超出后明显恶化。
3. **局部性指标在本数据集失效**：`heldout` 基线就是 0/50（5-token greedy 下模型对这批 prompt 本就不答 `target_true`），所以"保留率"无法计算。要么换长答案解码 + 首 token 匹配，要么换 locality 数据集，属未完成项。
4. **CALM 模式**：门控侧正常（64 条中 WRITE 63 / ROUTE 1 / REJECT 0），写入器接入无异常；但跑完整流后统一口径重测命中为 0，与限制 1 同源（同为覆盖失效）。CALM 内部评估器用的是长答案生成 + 自己的匹配规则，与本文 5-token 口径不可直接比较。

---

## 6. 交付物位置

```
FONG CHIN WAI/w4_model_level_improve/
├── closedform_writer_v2.md                      # 本文件
└── results/
    ├── closedform_smoke_direct_n500_L6-7-8_l21/      # 推荐配置 500 条：summary.json + per_edit.jsonl
    ├── closedform_smoke_direct_n500_L4-5-6-7-8_l21/  # L4-8 对照
    ├── closedform_smoke_direct_n500_L6-7-8_l21(proj) # 投影版 500 条
    ├── closedform_smoke_direct_n128_L4-5-6-7-8_l21/  # 128 条
    └── closedform_smoke_calm_n64_L6-7-8_l21/         # CALM 集成冒烟
```

代码：
- `EasyEdit/easyeditor/models/calm/closed_form_writer.py`（闭式解 + 投影）
- `EasyEdit/easyeditor/editors/calm_editor.py`（`hypernet_alg=closedform` 分支）
- `EasyEdit/examples/w4_closedform_smoke.py`（direct / calm 双模式流式评测）

---

## 7. 下一步（若要继续提分）

1. **批量联合求解**：MEMIT 原意是一次解一个 batch 的所有键（`K` 为 `[d_in, n_batch*ctx]`），把覆盖冲突交给一次最小二乘处理，可直接改善长程保留。
2. **写入门控与预算**：同一 subject 只写一次、冲突 subject 改走 side memory（CALM 已有 `_first_write_guard`，需在模型级指标里体现）。
3. **locality 重做**：换成首 token 匹配 + 长答案解码，或换 zsRE / CounterFact 的标准 locality 子集。
