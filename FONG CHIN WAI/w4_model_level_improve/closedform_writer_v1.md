# W4 写入器替换 · MEMIT/AlphaEdit 闭式解 v1 骨架

日期：2026-11-06 ｜ 负责人：Fong Chin Wai（Vivi）｜ 模型：gpt2-small（12 层，8GB 显存）

## 1. 为什么换写入器

W4 已测的全部路线的**真实**模型级改写率都落在 1.6%–3.9%（22.2% 是 REJECT 假命中）。瓶颈被定位为超网络学到的 `delta = scale * U @ V^T` 方向本身不对，而不是路由或预算。MEMIT/AlphaEdit 不训练生成器，直接在模型上解闭式方程：

```
min_delta ||(W + delta) K - (W K + R)||^2 + lambda ||delta||^2
=>  delta = R K^T (K K^T + lambda I)^-1
```

K = 改写模块（MLP `c_proj`）在主语末 token 的输入激活，R = 指向优化出的价值向量 z 的残差误差。方向由模型自身导出，不依赖在别的目标上训过的生成器。

## 2. 落地方式（已写入仓库）

| 文件 | 变更 |
| --- | --- |
| `EasyEdit/easyeditor/models/calm/closed_form_writer.py` | 新增 `ClosedFormEditGenerator`，与 `HypernetworkEditGenerator` 同接口：`generate_update` / `apply_update` / `generate_and_apply` / `summary` |
| `EasyEdit/easyeditor/models/calm/calm_hparams.py` | 新增 `cf_*` 超参块（layers / l2 / v_lr / v_steps / kv_cache / null_space / module 模板等），`hypernet_alg` 支持 `"closedform"` |
| `EasyEdit/easyeditor/editors/calm_editor.py` | `hypernet_alg="closedform"` 分支，路由/预算/verify-commit 逻辑完全复用，零改动 |
| `EasyEdit/examples/w4_closedform_smoke.py` | W4 骨架运行器：`--mode direct` 隔离写入器、`--mode calm` 走完整门控链路；输出 `per_edit.jsonl` + `summary.json` |

设计要点：
- z 值向量直接复用 `easyeditor/models/alphaedit/compute_z.py`（Adam 优化单个 hidden 向量，目标 tied 到最后一层）。
- 逐层求解时**按 MEMIT 顺序**：第 i 层先落地再算第 i+1 层的 keys；由于每个权重只被加一次增量，计算完恢复原权重后再统一 `apply_update`，与顺序循环终态严格等价。
- 每次写入记录 `delta_norm / weight_norm_before / weight_norm_after / rel_delta` 与 `z_norm`、`solve_seconds`，供 `audit_per_edit.py` 复用。
- AlphaEdit 的 `cache_c`（历史 keys 统计）作为 `cf_kv_cache` 保留；**null-space 投影默认关闭**，`cf_null_space="stats"` 会显式报错而不是静默退化成 MEMIT（本地没有 mom2 统计与 `P_loc`）。

## 3. 关键踩坑：改写层必须早于最后一层

W3 的 `inner_params` 是 `transformer.h.{9,10,11}.mlp`，直接拿来当闭式解层会得到 **delta 范数恒为 0、loss 恒定不变**：

- `compute_z` 在 `transformer.h.{layer}` 输出（残差流）上注入 delta，loss 读的是**最后一个 token** 位置的输出；
- 若 `layer = 11`（最后一层），注入发生在主语末 token，而预测位 token 的残差在该层已经算完，二者之间没有任何下游层做因果传播 → 对 delta 的梯度严格为 0（已实测：`Delta norm 0.0`）；
- 因此闭式解层必须**严格小于 `n_layer - 1`**。gpt2-small 取 `[4,5,6,7,8]`（对应 gpt2-xl 48 层用 `[13..17]` 的相对位置），并在构造时加硬校验，越界直接抛错。

## 4. 首测结果（direct 模式，无门控，严格顺序写入同一模型）

32 条 counterfact 顺序写入，每次写后立即用 greedy 5-token 复测（`results/closedform_smoke_direct_n32_L4-5-6-7-8_l210/`）：

| 指标 | 数值 |
| --- | --- |
| 写入后命中 target_new | **30/32 = 93.75%** |
| 写入前已命中（防虚高对照） | 0/32 = 0% |
| 单条耗时（含 z 优化 20 步 + 求解 + 复测） | 均值 2.46 s |
| 32 条总耗时 | 78.7 s |
| delta 相对权重范数比例 | 见 `per_edit.jsonl` 的 `apply_stats` |

对照：W4 已测所有超网络路线真实 WRITE 命中 1.6%–3.9%。即写入器替换把「命中」这一环从个位数百分比拉到 93.75%，证明瓶颈判断成立。

## 5. 本结果**不能**直接报的边界（诚实口径）

1. 只测了 32 条、单一 l2=10、单 seed 顺序写入，**尚未跑 500 流**，也未接 `assert_contract`。
2. 90%+ 的命中伴随明显退化输出（连续重复 token，如 `Lisbon Lisbon Lisbon`、`pope pope pope`）——这正是 AlphaEdit null-space 要解决的问题；fluency/locality 必须在下一轮量化。
3. `rewrite_ok` 用的是本项目既有的包含式判据（5 token greedy 内出现 target 子串），口径与 eval_contract 的 L2 `contain_1st_sent` 一致，但**尚未按 conflict_tag 分层**，也没有 pre-training-conflict 硬 0 声明。
4. direct 模式**绕过了 CALM 门控**，因此这不是模型级改写率，而是写入器命中率；模型级指标仍需 `--mode calm` + 128-write 审计（分离 WRITE 真命中与 REJECT 假命中）。

## 6. 下一步（按顺序）

1. `--mode calm` 跑同一批编辑，确认门控在闭式解下不会把命中拦成 REJECT（gate 阈值需重标）。
2. 500 流全量 + `audit_per_edit.py`，报 L2 + margin、按 conflict_tag 分层、pre-training-conflict 硬 0。
3. fluency/locality 量化（continuation NLL / 邻域 perplexity），据此决定是否上 AlphaEdit null-space（需要 mom2 统计 + `P_loc`）。
4. l2 / kv_cache / layers 三档小网格（l2 ∈ {5, 10, 20}），确认收益不是单点运气。

## 7. 复现命令

```powershell
cd "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$env:PYTHONPATH="EasyEdit"
.\.venv\Scripts\python.exe EasyEdit\examples\w4_closedform_smoke.py --edits 32            # 写入器隔离
.\.venv\Scripts\python.exe EasyEdit\examples\w4_closedform_smoke.py --mode calm --edits 32  # 端到端（含门控）
```
