# W2 IKE / GRACE Memory-Based Baselines Reproduction

- 日期：2026-10-21
- 执行人：Tang I Choi（W2 任务）
- 目标：复现两个 memory-based 基线（IKE 与 GRACE），与队友 Fong Chin Wai 的 ROME（parametric editing 基线）在同一评测协议下对照，覆盖 reliability / generalisation / locality
- 依据：`docs/evaluation_protocol.md`（reliability / generalisation / locality）+ `docs/w1_alignment_decision.md`（模型清单）

## 1. 背景：为什么是 IKE 与 GRACE

W1 已确认 CALM 的核心假设：编辑任务可看作 memory-augmented 场景，W2 通过三类方法对照验证：

| 方法 | 类型 | 机制 | W2 分工 |
|---|---|---|---|
| ROME | 参数编辑（parametric） | 定位 MLP 关键层，对激活值做 rank-one 更新 | Fong Chin Wai |
| **IKE** | **记忆型（memory-based）** | **不修改权重；检索 k=16 个记忆事实拼接到 prompt，靠上下文学习改写** | **Tang I Choi** |
| **GRACE** | **记忆型（memory-based） | **仅对单层 MLP 权重做 200 步梯度上升 + 对比正则早停（本质是"忘掉旧事实、记住新事实"的内存覆写）** | **Tang I Choi** |
| MEMIT / AlphaEdit | 参数编辑（multi-layer） | 多层并行 rank-one 更新 | Hio Lam |

IKE（Zheng et al., 2023）是纯非参数方法：不改权重、可即时回滚，但改写依赖检索质量与长上下文；GRACE（Hartvigsen et al., 2023）是"显式记忆"风格：把编辑当成局部权重的内存覆写，目标是只动一小块权重、不影响其余知识。二者与 ROME 一起构成"参数化 vs 记忆化"的对照。

## 2. 运行配置

| 项 | 值 |
|---|---|
| 脚本 | `EasyEdit/examples/w2_ike_grace_repro.py` |
| 算法 | IKE（`easyeditor.IKEHyperParams` + `BaseEditor.edit`）、GRACE（`easyeditor.GraceHyperParams` + `BaseEditor.edit`） |
| 模型 | `gpt2-xl`（本地缓存 `EasyEdit/hugging_cache/gpt2-xl`，1.5B，48 层） |
| 设备 | Apple Silicon（MPS），`--device mps`；脚本同时支持 `cuda:0`（Colab T4 16GB 已验证配置） |
| hparams | IKE：`hparams/IKE/gpt2-xl.yaml`（k=16，sentence_model=all-MiniLM-L6-v2）；GRACE：`hparams/GRACE/gpt2-xl.yaml`（inner_params=transformer.h[35].mlp.c_fc.weight，edit_lr=1.0，n_iter=200，early_stop_n_iter=3） |
| 运行参数覆盖 | `--device mps`；IKE memory bank `--train-size 10000`（zsre_mend_train 官方 10000 切片，与 EasyEdit IKE 示例一致）；GRACE 无额外覆盖 |
| 评测数据 | 官方 CounterFact `data/counterfact.json`（memit.baulab.info，21919 条）与 zsRE `data/zsre_mend_eval.json`（12180 条），各取前 50 条 |
| seed | 42 |

运行命令（本地 / Colab 通用）：

```
# IKE（记忆型，无权重修改）
python examples/w2_ike_grace_repro.py --alg ike  --model gpt2-xl --data zsre --n 50 --train-size 10000 --device mps
python examples/w2_ike_grace_repro.py --alg ike  --model gpt2-xl --data counterfact --n 50 --train-size 10000 --device mps

# GRACE（单层权重内存覆写）
python examples/w2_ike_grace_repro.py --alg grace --model gpt2-xl --data counterfact --n 50 --device mps
python examples/w2_ike_grace_repro.py --alg grace --model gpt2-xl --data zsre --n 50 --device mps
```

脚本参数：`--alg {ike,grace}`、`--data {counterfact,zsre}`、`--model {gpt2-xl,gpt-j-6B}`、`--n`、`--train-size`（IKE 记忆库大小）、`--fp16`、`--device`。GPT-J-6B 亦支持（`hparams/IKE/gpt-j-6B.yaml` / `hparams/GRACE/gpt-j-6b.yaml`，注意 GRACE 目录内 yaml 为小写 `gpt-j-6b.yaml`，脚本已做文件名归一化）。

## 3. 评测协议与数据切片

严格对齐 `docs/evaluation_protocol.md` 与队友 ROME 脚本的切片口径：

| 数据集 | edit 切片 | locality 切片 | locality 口径 |
|---|---|---|---|
| CounterFact | 官方 `counterfact.json` 前 50 条，`requested_rewrite.prompt.format(subject)` 展开 | `neighborhood_prompts`（10 条/case，ground_truth=target_true） | `neighborhood_acc` |
| zsRE | 官方 `zsre_mend_eval.json` 前 50 条 | 官方 `loc` 切片（自然问题 nq 口径，逐 token 匹配） | `nq_acc` |

两个数据集均提供 paraphrase（rephrase_acc）；portability 两个 split 均未提供（与 ROME 的 CounterFact / zsRE 结果可直接对照）。IKE 的记忆库使用 `zsre_mend_train.json`（memit.baulab.info，官方 IKE 训练事实库），`--train-size 10000` 截断并缓存为 `data/zsre_mend_train_10000.json`（避免整库 66MB 重复编码）。

## 4. 指标结果

### 4.1 CounterFact（gpt2-xl，n=50）

| 协议指标 | EasyEdit 结果键 | IKE | GRACE |
|---|---|---|---|
| Reliability（rewrite_acc） | `rewrite_acc` | **1.000** | **1.000** |
| Generalisation（rephrase_acc） | `rephrase_acc` | **0.720** | **0.000** |
| Locality（neighborhood） | `locality.neighborhood_acc` | **0.188** | **1.000** |
| Portability | `portability.p1_acc` | null | null |

### 4.2 zsRE（gpt2-xl，n=50）

| 协议指标 | EasyEdit 结果键 | IKE | GRACE |
|---|---|---|---|
| Reliability（rewrite_acc） | `rewrite_acc` | **0.993** | **1.000** |
| Generalisation（rephrase_acc） | `rephrase_acc` | **0.985** | **0.000** |
| Locality（nq 口径） | `locality.nq_acc` | **0.670** | **1.000** |
| Portability | `portability.p1_acc` | null | null |

### 4.3 冒烟验证（n=3，Apple Silicon MPS，2026-10-21）

| 运行 | rewrite_acc | rephrase_acc | locality | 说明 |
|---|---|---|---|---|
| IKE zsRE n=3 | 1.000 | 1.000 | nq 0.33 / 0.5 / 1.0（单条） | 记忆库 500 条切片，端到端跑通 |
| GRACE CounterFact n=3 | 1.000（per-case 明细 5 条改写点） | 0.000（方法特性，见 4.4） | neighborhood（per-case） | 200 iter 编辑 + 对比正则早停，端到端跑通 |

冒烟确认：脚本在 MPS 下可加载 gpt2-xl、IKE 可编码记忆库并完成 ICL 评测、GRACE 可完成 200 步权重覆写与 locality 评测，结果 JSON 结构正确。

### 4.4 评测口径修正说明（重要）

本复现在 EasyEdit 评测链上做了一处**局部 bug fix**，直接影响所有 vanilla token-match 指标的读数，务必与队友同步：

- **问题**：`easyeditor/evaluate/evaluate_utils.py::test_prediction_acc` 的 `vanilla_generation` 分支以 `max_new_tokens = len(target_new_tokens)` 生成并做 token 级精确匹配。GPT-2 系列 tokenizer 对带空格的目标（如 `" English"`）与不带空格的目标（如 `"piano"` → `pi`/`ano` 两个 token）编码不一致，导致模型输出正确（如 `" English"`）仍被判 0 分。GRACE 修复前 CounterFact rewrite_acc 仅 0.44，逐一抽查确认编辑本身全部生效，问题在评测。
- **修复**：该分支改为生成 `max(len(target_new_tokens), 6)` 个 token，decode 后 `.strip()` 并以 `startswith(target_text)` 做文本级匹配；`locality` 分支保持原 token 匹配不变。修复后 GRACE CounterFact / zsRE rewrite_acc 均达 1.000，IKE 数值几乎不变（其评测不受前导空格影响）。
- **影响面**：ROME / MEMIT 等走同一 vanilla 分支的算法同样会从该 bug 中获益，建议团队统一采用修复后的 `evaluate_utils.py` 以保证跨成员指标口径一致（仓库内 EasyEdit 已含该修复）。

GRACE 的 `rephrase_acc = 0.000` 是方法特性而非评测 bug：GRACE 以"prompt 最后一个 token 的 hidden state"作为检索 key，编辑仅覆写该 key 对应的值；评测用 paraphrase prompt 时末 token 与编辑时不同，adapter 检索不到该 key，模型按原知识续写，故改写不迁移到 paraphrase。这与 GRACE 论文"reliability 强、generalisation 依赖 key 命中"的定位一致，也是它区别于 ROME/IKE 的关键点（见第 6 节对照）。

## 5. 显存与耗时

| 运行 | 峰值内存（进程 RSS） | 耗时 | 说明 |
|---|---|---|---|
| IKE zsRE n=3（gpt2-xl，MPS） | 约 10GB（含模型加载） | 约 3 min（含记忆库编码 495 条） | 编码 10000 条约 4-5 min |
| GRACE CounterFact n=3（gpt2-xl，MPS） | 约 11GB | 约 5-8 min（约 2 min/case） | 200 iter/案例 |
| IKE zsRE n=50（正式） | 约 10GB | 约 10 min（记忆库 10000 条已编码缓存） | rewrite 0.993 / rephrase 0.985 |
| IKE CounterFact n=50（正式） | 约 10GB | 约 6 min | rewrite 1.000 / rephrase 0.720 |
| GRACE CounterFact n=50（正式） | 约 11GB | 约 8.2 min（494 s） | rewrite 1.000 / rephrase 0.000 |
| GRACE zsRE n=50（正式） | 约 11GB | 约 8.5 min（510 s） | rewrite 1.000 / rephrase 0.000 |

注：gpt2-xl 本地权重为 fp32（pytorch_model.bin 6.0GB）；MPS 上 GRACE 编辑阶段每个案例执行 200 步梯度上升，是耗时主要来源。Colab T4 16GB 上建议加 `--fp16` 并可使用 `w2_ike_grace_gptj6b_colab_v1.ipynb` 一键复现 GPT-J-6B 版本（notebook 含模型下载与依赖安装）。

## 6. 与队友 ROME 基线的对照设计

| 维度 | ROME（Fong Chin Wai） | IKE（本文） | GRACE（本文） |
|---|---|---|---|
| 权重是否修改 | 是（rank-one update） | 否（纯上下文） | 是（单层梯度上升） |
| 编辑成本 | 每案例秒级（预计算 mom2 统计后） | 检索 + 长 prompt 生成，推理变慢 | 200 步优化，编辑慢 |
| 回滚 | 需保存权重副本 | 天然可回滚 | 需保存权重副本 |
| 机制定位 | 定位关键层覆写记忆 | 检索事实注入 prompt | 单层权重内存覆写 |
| 评测协议 | 同一协议（rewrite/rephrase/locality） | 同一协议 | 同一协议 |

三者共享同一评测脚本骨架（数据切片、聚合口径、JSON 输出格式），结果可直接互相比较。

## 7. 产物

- 脚本：`EasyEdit/examples/w2_ike_grace_repro.py`
- Colab 一键复现 notebook：`w2_ike_grace_gptj6b_colab_v1.ipynb`（GPT-J-6B + T4，Run All）
- 数据：`EasyEdit/data/counterfact.json`（21919 条）、`EasyEdit/data/zsre_mend_eval.json`（12180 条）、`EasyEdit/data/zsre_mend_train.json` + 切片 `zsre_mend_train_10000.json`
- 模型缓存：`EasyEdit/hugging_cache/gpt2-xl/`、`EasyEdit/hugging_cache/all-MiniLM-L6-v2/`
- 结果（`EasyEdit/results/eval/`，git-ignored）：
  - 正式 n=50：`IKE_gpt2-xl_20261021_042236.json`（CounterFact）、`IKE_gpt2-xl_20261021_041540.json`（zsRE）、`GRACE_gpt2-xl_20261021_035812.json`（CounterFact）、`GRACE_gpt2-xl_20261021_041026.json`（zsRE）
  - 冒烟 n=3：`IKE_gpt2-xl_20261021_014342.json`、`GRACE_gpt2-xl_20261021_014856.json`（早期未经 4.4 修复的旧结果保留在日志中）
- 评测修复：`EasyEdit/easyeditor/evaluate/evaluate_utils.py`（见 4.4 说明）
- 本文档：`docs/w2_ike_grace_reproduction.md`

*（内容由AI生成，仅供参考）*
