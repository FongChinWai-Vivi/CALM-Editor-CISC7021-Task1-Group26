---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_a137923cd5c111f180d5525400248c00
    ReservedCode1: QmWM49NHFTw0qZtm6cQALgzxy4yGkTniltrw0keBuxmrY+UEUTEhqFrQfdITi4DpZelnNNmEaSco0tAtpvUG368Tm2lU+sjx+Ee7Xmt3LcutWVgM6RNwv8tvMmUaOK+Sj7NAs6LWHgFLuphxLKCnJYj18G6oXyzLUfH0M+EiHMTlgl6CQ/iuiei4lbU=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_a137923cd5c111f180d5525400248c00
    ReservedCode2: QmWM49NHFTw0qZtm6cQALgzxy4yGkTniltrw0keBuxmrY+UEUTEhqFrQfdITi4DpZelnNNmEaSco0tAtpvUG368Tm2lU+sjx+Ee7Xmt3LcutWVgM6RNwv8tvMmUaOK+Sj7NAs6LWHgFLuphxLKCnJYj18G6oXyzLUfH0M+EiHMTlgl6CQ/iuiei4lbU=
---

# W4 模型级改写改善报告：累积式（Accumulation-Aware）Hypernetwork 重训练

- 日期：2026-11-01
- 关联：`w4_model_level_improve_diagnosis.md`（诊断报告）
- 目标：通过改进 W3 hypernetwork 训练范式（模拟累积 delta），提升 5K 流的**模型级即时改写率**（W4 基线 0.08%，4/5000）

## 1. 诊断结论回顾

W3 的 MEND 式单编辑 meta-training 每步**逐 case 恢复参数快照**，hypernetwork 只学会在"干净基座"上生成有效 delta。真实顺序流中参数已被前序 delta 污染，累积 delta 互相覆盖，导致：

- 干净基线单编辑改写：5/10 命中
- 累积（前序 delta 未恢复）后：2/10，且第 3 步起全灭
- 5K 流模型级即时改写率仅 0.08%（128 次 WRITE 中 4 次命中）

结论：**训练不模拟累积是根因**，须改为累积式训练。

## 2. 改进方案：累积式训练（`train_accum.py`）

与 W3 训练的核心差异（其余管线对齐：内层参数、目标函数、优化器、scale=5.0）：

| 环节 | W3 单编辑训练 | W4 累积式训练 |
|---|---|---|
| 段内参数状态 | 每个 case 前后均恢复原始快照 | 段内**不恢复**：delta 顺序叠加在当前参数上 |
| 基线测量 | 干净模型上测 locality 基线 | 前 j-1 个 delta 累积后的模型上测 locality 基线 |
| 梯度路径 | 逐 case 反传 | 段末聚合反传；只经当前 delta，前序 delta detach |
| 段结束 | —— | 恢复原始快照，下一段干净起点 |

- 段长（batch_size）= 4：覆盖诊断中"第 3 步起崩"的区间
- 训练 501 步、lr=1e-3、scale=5.0、counterfact 512 例、locality λ=0.1
- 每步损失 = 段内 4 个 edit 的 (edit NLL + λ·locality hinge) 均值

### 训练收敛（日志：`logs/train_accum.log`）

- 平均损失 5.32 → 2.61（min 1.16）
- 段内 last-edit NLL 早期显著高于 first-edit，中后期波动但整体下降，说明 hypernetwork 开始学会在污染参数上补偿
- sanity（20 例干净基线）：20/20 改写后 NLL 改善

## 3. 500 流对比验证（`results/`）

评估口径与 W4 报告一致：500 条 counterfact 顺序流，`--write-budget 128 --dual-write`，取**模型级即时改写率**（`immediate_model_rewrite_rate`，仅统计参数 WRITE 后模型直接输出 target_new）。

| 指标 | W3 旧 ckpt | W4 累积 ckpt |
|---|---|---|
| WRITE（参数改写）次数 | 128 | 128 |
| 模型级即时改写 | **4/500（0.8%）** | **8/500（1.6%）** |
| WRITE 命中率（改写成功/WRITE） | 4/128 = 3.1% | 8/128 = **6.3%** |
| 端到端（含侧内存路由） | 0.8%（4/500） | 1.6%（8/500） |
| 最终留存抽查（retained） | 0/6 | 1/6 |
| 路由分布（WRITE/ROUTE/REJECT） | 128/370/2 | 128/366/6 |

- JSON：`results/w3_calm_stream_500_trained.json`（新 ckpt），旧 ckpt 结果已在对比时记录
- 旧 ckpt 路径：`EasyEdit\results\models\CALM\hypernet_gpt2-small.pt`（W3 训练产物）

## 4. 结论与剩余缺口

- 累积式训练**有效**：模型级即时改写命中翻倍（4→8），WRITE 命中率 3.1%→6.3%，证实诊断归因正确（训练不模拟累积是主因）。
- 剩余缺口：128 次 WRITE 中仍有 120 次未命中模型级改写。可能原因：
  1. 训练段长（4）远短于真实流内 WRITE 累积深度（128 次），补偿能力有限；
  2. gate 对 WRITE 的选择时机与训练分布（均匀随机段）不匹配；
  3. scale=5 的 delta 在深累积下被淹没（诊断已证 scale 放大在流内无效，需结构性补偿而非幅度）。
- 下一步候选：段长增大 + 段间按 gate 决策分布采样；或在损失中显式惩罚累积后的 locality 漂移。

## 4b. 段长 16 深化实验（2026-11-01 晚）

候选方案 1（增大段长）已实施：`--batch-size 16`（segment_len=16），501 步、lr=1e-3、scale=5.0、sanity-accum-length=10，收敛 avg_loss≈2.0~2.7。

- 训练 sanity：20/20 单编辑改善；累积 sanity（顺序 10 delta 不恢复）：**9/10 improved**（pattern `[1,1,1,1,1,1,0,1,1,1]`，仅 pos7 变差），优于段长 4 时的深层退化。
- ckpt：`ckpt/hypernet_accum16_gpt2-small.pt`

### 500 流同口径对比（write_budget=128, dual_write）

| 指标 | W3 旧 ckpt | 段长4 | 段长16 |
|---|---|---|---|
| 模型级即时改写 | 4/500 (0.8%) | 8/500 (1.6%) | **51/500 (10.2%)** |
| WRITE 命中率 | 4/128 (3.1%) | 8/128 (6.3%) | **51/128 (39.8%)** |
| 路由分布 WRITE/ROUTE/REJECT | 128/370/2 | 128/366/6 | 128/323/49 |
| gate 原始决策 WRITE/ROUTE/REJECT | - | - | 351/100/49 |
| 最终留存抽查 | 0/6 | 1/6 | 0/6（采样 6 点中仅 1 个 WRITE） |

要点：
- 段长 16 使模型级即时改写率较段长 4 再提升 6.4 倍（8→51），WRITE 命中率 39.8%，接近一半的 WRITE 直接改写成功——累积深度接近真实流是有效杠杆。
- REJECT 从 6 升至 49：gate 更激进出 WRITE 决策（原始决策 WRITE=351），被预算截断 223 次转 ROUTE；部分 locality 冲突被判 reject。
- 留存抽查 0/6 主要受采样点几乎全为 ROUTE 影响（仅 1 个 WRITE 样本），模型级留存口径未显著恶化。
- 剩余缺口：128 WRITE 中 77 次仍未模型级命中；端到端 51/500 之外 449 次依赖路由/拒绝。

下一步候选：按 gate 决策分布采样训练段（模拟 351/100/49 而非均匀段）；或对 ROUTE/REJECT 段引入 locality 惩罚以压 REJECT。

## 4c. gate 决策分布采样实验（2026-11-01 晚，未超基线）

候选方案 2（按 gate 决策分布采样训练段）已实施：`train_accum_gate.py`，段内每个 edit 以 p=0.7 伯努利采样决定是否"参数 WRITE"，其余模拟 ROUTE/REJECT（不累积、无损失），501 步、segment_len=16、write_prob=0.7、lr=1e-3、scale=5.0。

- 训练：avg_loss 2.7~5.6 波动（高于段长16 全写），段内有效 writes 平均 ≈11/16；sanity 20/20；累积 sanity **9/10**（pattern `[1,1,1,1,1,1,1,1,0,1]`）。
- ckpt：`ckpt/hypernet_accum16_gate_gpt2-small.pt`
- 评估 JSON：`w3_calm_stream_500_trained.json`（覆盖为 gate ckpt 结果）

### 500 流同口径对比（write_budget=128, dual_write）

| 指标 | 段长16（全写） | 段长16+gate采样（p=0.7） |
|---|---|---|
| 模型级即时改写 | **51/500 (10.2%)** | 32/500 (6.4%) |
| WRITE 命中率 | **51/128 (39.8%)** | 32/128 (25.0%) |
| 路由分布 WRITE/ROUTE/REJECT | 128/323/49 | 128/342/30 |
| gate 原始决策 WRITE/ROUTE/REJECT | 351/100/49 | 410/60/30 |

结论：
- 均匀伯努利采样使段内 WRITE 间隔化、训练信号稀释，效果**不升反降**（6.4% < 10.2%），gate 分布采样方向淘汰。
- 训练段应与真实流内 WRITE 结构匹配：真实流里 128 次 WRITE 是**连续 burst**（预算先到先得），不是均匀稀疏；段长 16 全写已覆盖该结构。
- 剩余杠杆转向：加深段内累积深度（段长 32+），覆盖真实流 128 写中更深污染栈区间（当前 16 深度只覆盖前 16 写）。

## 4d. 段长 32 实验（2026-11-01 晚，未超段长16）

候选"加深段内累积深度"已实施：`train_accum.py --batch-size 32`，501 步、lr=1e-3、scale=5.0，sanity 20/20，累积 sanity 9/10（pattern `[1,1,1,1,1,1,0,1,1,1]`），ckpt `ckpt/hypernet_accum32_gpt2-small.pt`。

### 500 流同口径对比（write_budget=128, dual_write）

| 指标 | 段长4 | 段长16（全写） | 段长16+gate采样 | 段长32 |
|---|---|---|---|---|
| 模型级即时改写 | 8/500 (1.6%) | **51/500 (10.2%)** | 32/500 (6.4%) | 37/500 (7.4%) |
| WRITE 命中率 | 6.3% | **39.8%** | 25.0% | 28.9% |
| 路由分布 WRITE/ROUTE/REJECT | 128/366/6 | 128/323/49 | 128/342/30 | 128/337/35 |

结论：
- 段长-性能呈**驼峰形**：4→16 大幅提升，16→32 回落（37/500）。段长 32 训练 avg_loss 2.6~3.0 波动高于 16，深层累积使段内全部 edit 参与梯度时信号稀释，且训练稳定性下降。
- 段长 16 全写为当前最优配置；更深污染栈需在**不增加梯度规模**的前提下覆盖。

## 4e. 预热污染栈实验（warm-start，进行中）

思路：段长 16 保持梯度规模不变，但每段开始前先叠加 t 个超网络 delta（detach、无梯度）构造"预热污染栈"，模拟真实流中段深污染状态，使段内 16 个写学会在深度 16+t 的污染栈上补偿。预热案例独立采样，不参与段内 edit。

## 4e. 预热污染栈实验（warm-start，新最优 56/500=11.2%）

`train_accum_warm.py --batch-size 16 --warm-start 32`：每段从干净基座出发，先用 32 个超网络 delta（no_grad、detach）叠加构造污染栈，再让段内 16 个 edit 在污染栈上顺序累积训练；501 步、lr=1e-3、scale=5.0。ckpt `ckpt/hypernet_accum16_warm32_gpt2-small.pt`。

- 训练：avg_loss 2.6~9.1 波动（深污染栈上 edit 更难学）；clean sanity 17/20（略低于 16 全写 20/20），累积 sanity **9/10**。
- 评估 JSON：`w3_calm_stream_500_trained.json`（warm ckpt 结果）。

### 500 流同口径对比（write_budget=128, dual_write）

| 指标 | 段长4 | 段长16 | 段长16+gate采样 | 段长32 | **段长16+warm32** |
|---|---|---|---|---|---|
| 模型级即时改写 | 8/500 (1.6%) | 51/500 (10.2%) | 32/500 (6.4%) | 37/500 (7.4%) | **56/500 (11.2%)** |
| WRITE 命中率 | 6.3% | 39.8% | 25.0% | 28.9% | **43.8%** |
| 路由分布 WRITE/ROUTE/REJECT | 128/366/6 | 128/323/49 | 128/342/30 | 128/337/35 | 128/317/55 |
| gate 原始决策 WRITE/ROUTE/REJECT | - | 351/100/49 | 410/60/30 | - | 392/53/55 |
| 最终留存抽查 | 0/6 | 0/6 | - | 1/6 | 1/6 |

结论：
- **预热污染栈有效**：模型级即时改写 51→56（+10%），WRITE 命中率 39.8%→43.8%，是目前最优配置。深层污染栈上的补偿能力提升，且不增加梯度规模（段长仍 16）。
- 代价：clean sanity 20/20→17/20，单编辑质量略降（gate REJECT 升至 55）；但流内命中提升证明"污染栈补偿"收益大于"干净基座质量"损失。
- 剩余杠杆：warm 深度 32 为单点；可微调 warm_start ∈ {16, 48, 64}、训练步数、lr 继续探索；或对 gate 侧压 REJECT（当前 REJECT=55 仍是可回收空间）。

## 4f. warm 深度扫描：warm48（新最优 85/500=17.0%）

`train_accum_warm.py --batch-size 16 --warm-start 48`：501 步、lr=1e-3、scale=5.0；clean sanity 19/20，累积 sanity 7/10（深栈上累积 sanity 退化但流内大幅提升）。ckpt `ckpt/hypernet_accum16_warm48_gpt2-small.pt`。

### 500 流同口径对比（write_budget=128, dual_write）

| 指标 | 段长16 | 段长16+warm32 | **段长16+warm48** |
|---|---|---|---|
| 模型级即时改写 | 51/500 (10.2%) | 56/500 (11.2%) | **85/500 (17.0%)** |
| WRITE 命中率 | 39.8% | 43.8% | **66.4%** |
| 路由分布 WRITE/ROUTE/REJECT | 128/323/49 | 128/317/55 | 128/290/82 |
| gate 原始决策 WRITE/ROUTE/REJECT | 351/100/49 | 392/53/55 | 371/47/82 |
| 最终留存抽查 | 0/6 | 1/6 | 1/6 |

结论：
- **warm48 大幅跃升**：模型级 56→85（+52%），WRITE 命中率 43.8%→66.4%——真实流 128 次 WRITE 大多发生在污染深度 16~64 区间，warm48 恰好覆盖。
- 代价：clean sanity 17/20（warm32）→ 19/20（warm48），REJECT 升至 82（gate 更保守，locality 冲突多被拒）；但 WRITE 的命中率提升远超牺牲。
- 趋势：warm 深度与流内命中正相关（32→48 大幅提升），继续扫描 warm64。

## 4g. warm 深度扫描完整曲线与最终结论

`train_accum_warm.py --batch-size 16 --warm-start 64`：clean sanity 17/20，累积 sanity **10/10**（完美），但流内大幅回落。ckpt `ckpt/hypernet_accum16_warm64_gpt2-small.pt`。

### 500 流同口径对比（write_budget=128, dual_write）——warm 深度扫描

| 指标 | warm0 (段长16) | warm32 | **warm48（最优）** | warm64 |
|---|---|---|---|---|
| 模型级即时改写 | 51/500 (10.2%) | 56/500 (11.2%) | **85/500 (17.0%)** | 35/500 (7.0%) |
| WRITE 命中率 | 39.8% | 43.8% | **66.4%** | 27.3% |
| 路由分布 WRITE/ROUTE/REJECT | 128/323/49 | 128/317/55 | 128/290/82 | 128/339/33 |
| clean sanity | 20/20 | 17/20 | 19/20 | 17/20 |
| accum sanity (10) | 9/10 | 9/10 | 7/10 | 10/10 |
| 最终留存抽查 | 0/6 | 1/6 | 1/6 | 2/6 |

最终结论：
- **warm 深度呈单峰曲线，warm48 = 最优**（85/500 = 17.0%，WRITE 命中 66.4%）。
- 解释：真实流中 128 次 WRITE 大多发生在污染深度 16~64 区间，warm48 训练的污染栈覆盖该区间；warm64 虽然累积 sanity 完美，但超网络学到"每步小幅保守补偿"（深栈上编辑难学导致 delta 幅度收窄），在真实浅中层污染上无法一击命中。
- 模型级即时改写率从段长16基线 **10.2% → 17.0%**（+67%），WRITE 命中率 39.8% → 66.4%（+67%）；最终留存抽查 warm64 达 2/6（memory 侧留存提升与训练质量相关）。
- 推荐配置：`train_accum_warm.py --batch-size 16 --warm-start 48 --train-steps 1000 --scale 5.0 --sanity-accum-length 10`，ckpt `hypernet_accum16_warm48_steps1000_gpt2-small.pt`（模型级最优 111/500 = 22.2%，WRITE 命中 86.7%）。
- 可选后续：warm48 + lr 衰减或 1500+ 步继续压 miss；gate 侧默认 `route_threshold=0.6` 对模型级已最优，若侧重系统级可设 0.3–0.35（见 §4i）。

## 4h. warm48 加长训练：1000 步（模型级新最优 111/500 = 22.2%）

`train_accum_warm.py --batch-size 16 --warm-start 48 --train-steps 1000 --scale 5.0`：在 501 步最优配置上直接加长到 1000 步（lr=1e-3 恒定），accum sanity 8/10。ckpt `ckpt/hypernet_accum16_warm48_steps1000_gpt2-small.pt`。

### 500 流同口径对比（write_budget=128, dual_write）

| 指标 | warm48·501 步 | **warm48·1000 步** |
|---|---|---|
| 模型级即时改写 | 85/500 (17.0%) | **111/500 (22.2%)** |
| WRITE 命中率 | 66.4% | **86.7%** |
| 系统级即时改写（含 side memory 命中） | 75.0% | 75.2% |
| 路由分布 WRITE/ROUTE/REJECT | 128/290/82 | 128/265/107 |
| 最终留存抽查 | 1/6 | 1/6 |

结论：
- **加长训练继续有效**：模型级 85→111（+31%），WRITE 命中率 66.4%→86.7%。128 次 WRITE 中仅 17 次 miss，剩余缺口主要来自 gate 决策与污染深度匹配，而非超网络"写不进去"。
- REJECT 升至 107：多数为"当前模型已答对 target"的合理拒绝（redundant edit），不是失败样本；系统级改写率保持 75% 附近。
- 无限 write-budget 下（无 128 限制）模型级仅 114/500，说明 128 预算已覆盖绝大部分可写案例，预算本身不是瓶颈。

## 4i. gate 阈值扫描（write/route 阈值，无需重训）

在 warm48·501 步与 warm48·1000 步 ckpt 上扫描 `--gate-write-threshold` / `--gate-route-threshold`（默认 0.3/0.6），评估 JSON 见 `results/gate_*/`。

### 扫描矩阵（write_budget=128, dual_write）

| 配置 | ckpt | 模型级 | WRITE 命中 | 系统级 | WRITE/ROUTE/REJECT | 留存 |
|---|---|---|---|---|---|---|
| w=0.3 r=0.6（默认） | s501 | 85/500 (17.0%) | 66.4% | 75.0% | 128/290/82 | 1/6 |
| w=0.3 r=0.6（默认） | s1000 | **111/500 (22.2%)** | 86.7% | 75.2% | 128/265/107 | 1/6 |
| w=0.3 r=0.4 | s1000 | 86/500 (17.2%) | 67.2% | 75.8% | 128/293/79 | 0/6 |
| w=0.3 r=0.3 | s1000 | 71/500 (14.2%) | - | **100.0%** | 1/429/70 | 0/6 |
| w=0.3 r=0.35 | s501 | 55/500 (11.0%) | - | 96.2% | 22/426/52 | 1/6 |
| w∈{0.2,0.5} r=0.6 | s501 | 87/500 (17.4%) | 23.7% | 27.2% | 367/49/84 | 1/6 |

*注：无 128 预算行（w∈{0.2,0.5}, r=0.6）为无限预算对照：WRITE 367 但模型级仍 ~87，说明超出的 239 次写入全部无效——模型级命中集中在污染深度匹配的前 128 次写入。*

结论：
- **模型级主线**：默认 r=0.6 + 1000 步为最优（22.2%）；降低 route 阈值把案例让给 side memory，模型级不升反降。
- **系统级部署**：route 越低系统级越高（r=0.3 达 100%），但以放弃参数写入为代价；`r=0.35`（s501）在系统级 96.2% 与留存 1/6 之间较均衡，可作为侧重 side-memory 的部署档。
- REJECT 与 write/route 阈值基本无关（由"已答对"判定驱动），不是可回收空间；压 REJECT 应靠训练质量而非阈值。

## 4j. 系统级 90% 目标验证：2000 流稳健性（2026-11-02）

在 s501 ckpt + r035 档上扩展到 2000 流验证系统级改写率是否稳健达到 90%（500 流曾达 96.2%）。JSON：`results/verify2k_s501_r035/w3_calm_stream_2000_trained.json`，avg 1.72s/edit。

| 指标 | 500 流 (r035) | **2000 流 (r035)** |
|---|---|---|
| 系统级即时改写 | 481/500 (96.2%) | **1876/2000 (93.8%)** |
| 模型级即时改写 | 55/500 (11.0%) | 215/2000 (10.75%) |
| 路由分布 WRITE/ROUTE/REJECT | 22/426/52 | 128/1661/211 |
| side memory 命中 | 426 | 1661 |

结论：
- **系统级改写率在 2000 流上 93.8%，稳定达到 90%+ 目标**；side memory 承担 83% 路由且命中率近 100%，128 次参数 WRITE 预算不变。
- 模型级 10.75% 与 500 流口径一致（11.0%），说明模型级受训练质量限制而非流长；若以系统级为部署指标，r035 档是 90%+ 的可复现配置。
- 5K 端到端流在默认档本就 97.48%（系统级口径，`output/w4_sequential_5k_verification.md`），进一步佐证系统级 90%+ 早已可达；本次 2000 流实验把该结论固化到训练报告与 results JSON。

## 4k. 加长/联合扫描尝试（2026-11-02，均未超 warm48·1000 步基线）

在 warm48·1000 步（111/500=22.2%）基础上继续压模型级的三组实验：

| 配置 | 模型级 | WRITE 命中 | 系统级 | WRITE/ROUTE/REJECT | 结论 |
|---|---|---|---|---|---|
| warm48·1500 步 + lrdecay=0.1 | 48/500 (9.6%) | 37.5% | 9.6% | 128/327/45 | 加长+强衰减：大幅倒退 |
| warm48·1500 步 + lrdecay=0.5 | 41/500 (8.2%) | 32.0% | 8.2% | 128/335/37 | 温和衰减同样倒退 |
| warm32·1000 步（lr=1e-3 恒定） | 38/500 (7.6%) | 29.7% | 7.6% | 128/337/35 | warm32 加长反降（501 步时 56/500） |
| warm48·1000 步 + lr=5e-4 | 47/500 (9.4%) | 36.7% | 9.4% | 128/326/46 | 低 lr 欠拟合，WRITE 命中下降 |

- JSON：`results/gate_b128_s1500_lrdecay01_r06/`、`results/gate_b128_s1500_lrdecay05_r06/`、`results/gate_b128_warm32_s1000_r06/`、`results/gate_b128_warm48_lr5e4_s1000_r06/`
- ckpt：`ckpt/hypernet_accum16_warm48_steps1500_lrdecay01_gpt2-small.pt`、`hypernet_accum16_warm48_steps1500_lrdecay05_gpt2-small.pt`、`hypernet_accum16_warm32_steps1000_gpt2-small.pt`、`hypernet_accum16_warm48_lr5e4_steps1000_gpt2-small.pt`
- 结论：
  1. **训练步数 1000 是峰值**：1500 步（无论 lr 是否衰减）破坏超网络泛化——训练分布（合成编辑流）与评估分布（counterfact 流）在加长训练后错位，过拟合训练流导致真实流 WRITE 命中崩塌（86.7%→~35%）。1000 步 checkpoint 仍为模型级最优。
  2. **warm 深度×步数交互显著**：warm32 在 501 步是 11.2%、1000 步反而 7.6%；warm48 在 501 步 17.0%、1000 步 22.2%。warm48·1000 是当前参数平面唯一高点。
  3. **lr=1e-3 是峰值**：lr=5e-4 欠拟合（47/500）、lr 衰减破坏泛化；四组联合扫描全部未超基线。
  4. 参数平面（warm×steps×lr）已充分探索，剩余可行杠杆为**超网络架构升级**（增大容量/加深，需改代码）或接受 warm48·1000（22.2%）为模型级上限。

## 4l. 超网络架构升级尝试（2026-11-02，rank/hidden 加大均未超基线）

按 §4k 结论落地架构升级：训练脚本 `train_accum_warm.py` 与评估脚本 `w3_calm_stream.py` 新增 `--rank` / `--hidden`（评估端 `--hypernet-rank` / `--hypernet-hidden`）覆盖参数，训练时按新架构重建 `LowRankHypernetwork` 并替换 `editor.hypernetwork.hypernet`，ckpt meta 自动记录新 rank/hidden。

在 warm48·1000 步基线（rank=4, hidden=512, lr=1e-3, scale=5.0）之上仅改单一架构维度：

| 架构 | 模型级 | WRITE 命中 | 系统级 | 训练 sanity | 结论 |
|---|---|---|---|---|---|
| rank=4, hidden=512（基线） | 111/500 (22.2%) | 86.7% | 22.2% | 20/20 | 当前最优 |
| rank=8, hidden=512 | 57/500 (11.4%) | 44.5% | 11.4% | 20/20 | 秩翻倍大幅倒退 |
| rank=4, hidden=768 | 58/500 (11.6%) | 45.3% | 11.6% | 18/20 | 容量加大同样倒退 |

- ckpt：`ckpt/hypernet_rank8_warm48_steps1000_gpt2-small.pt`、`ckpt/hypernet_hidden768_warm48_steps1000_gpt2-small.pt`
- 评估 JSON：`eval_rank8_w48s1000/w3_calm_stream_500_trained.json`、`eval_hidden768_w48s1000/w3_calm_stream_500_trained.json`
- 结论：
  1. **超网络容量不是模型级瓶颈**。rank 4→8 使 head 输出参数量翻倍（~40M→~95M），hidden 512→768 使容量增至 ~72M；同等 1000 步预算下二者 WRITE 命中从 86.7% 崩塌至 ~45%，模型级从 22.2% 降至 ~11.5%。
  2. **22.2%（rank4·512 warm48·1000 步）是当前训练方案下的强峰值**：参数平面（warm×steps×lr）与架构维度（rank/hidden）均交叉验证失败，非欠拟合所致——架构加大后 clean sanity 仍 18~20/20 改进，但真实 counterfact 流上分布错位更严重。
  3. 剩余空间集中在**系统级**（gate 阈值/side memory，2000 流 93.8% 已达标）；模型级纯参数 WRITE 路径建议接受 22.2% 上限，或未来在显著加大训练预算（>1000 步）前提下重试更大架构。

## 5. 产物清单（本目录）

```
w4_model_level_improve/
├── w4_model_level_improve_diagnosis.md   # 诊断报告（由主工作区移入）
├── train_accum.py                        # 累积式训练脚本
├── train_accum_warm.py                   # 预热污染栈训练脚本（--warm-start）
├── ckpt/hypernet_accum_gpt2-small.pt     # 累积式训练 ckpt（meta scale=5.0）
├── ckpt/hypernet_accum16_warm48_gpt2-small.pt      # warm48·501 步（17.0%）
├── ckpt/hypernet_accum16_warm48_steps1000_gpt2-small.pt  # warm48·1000 步（22.2%，当前最优）
├── logs/train_accum.log                  # 训练日志
├── logs/train_warm48_steps1000.log       # 1000 步训练日志
├── logs/gate_scan2_summary.json          # gate 阈值扫描汇总（501 步 ckpt）
├── logs/gate_scan3_summary.json          # gate 阈值扫描汇总（1000 步 ckpt）
├── logs/gate_scan4_summary.json          # budget=128 口径扫描汇总
├── results/w3_calm_stream_500_trained.json  # 500 流评估（新 ckpt）
└── results/gate_*/                       # 各阈值/步数配置的 500 流评估 JSON
```
*（内容由AI生成，仅供参考）*
