---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_f265abcccae811f196f9525400248c00
    ReservedCode1: 0O/W0WBaZelUYPspyZU23H/Y5yS0It5Ba94EvtanksLxPeKDb6+bd3eFK4gajjS0m/uLdPxLht6Ib3Xb/pSQhomaSwEfvvnt0R7uQm1Ys1kFfg4mDaJgi5sNiBxPRjK96lIvyJwmn92WmcWWZefMO2sJN97CUispx2BKpOj/EXIlTkll/UVcXnvYQog=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_f265abcccae811f196f9525400248c00
    ReservedCode2: 0O/W0WBaZelUYPspyZU23H/Y5yS0It5Ba94EvtanksLxPeKDb6+bd3eFK4gajjS0m/uLdPxLht6Ib3Xb/pSQhomaSwEfvvnt0R7uQm1Ys1kFfg4mDaJgi5sNiBxPRjK96lIvyJwmn92WmcWWZefMO2sJN97CUispx2BKpOj/EXIlTkll/UVcXnvYQog=
---



# W2 ROME Single-Fact Baseline Reproduction

- 日期：2026-10-18
- 执行人：Fong Chin Wai（W2 任务）
- 目标：复现 ROME 单事实编辑基线，验证在 RTX 4070 8GB 环境可运行，并产出三类指标
- 依据：`docs/evaluation_protocol.md`（reliability / generalisation / locality）+ `docs/w1_alignment_decision.md`（模型清单）

## 1. 运行配置

| 项 | 值 |
|---|---|
| 脚本 | `EasyEdit/examples/w2_rome_repro.py` |
| 算法 | ROME（`easyeditor.ROMEHyperParams` + `BaseEditor.edit`） |
| 模型 | `gpt2-xl`（本地缓存 `EasyEdit/hugging_cache/gpt2-xl`，1.5B，48 层） |
| 设备 | RTX 4070 Laptop 8GB（CUDA） |
| hparams | `hparams/ROME/gpt2-xl.yaml`（layers=[17]，fact_token=subject_last，v_num_grad_steps=20，v_lr=5e-1，v_loss_layer=47，rewrite_module=transformer.h.17.mlp.c_proj） |
| 运行参数覆盖 | `--fp16`（脚本设置 `hparams.fp16=True`）；mom2_dataset=wikitext2，mom2_n_samples=10000（替代 yaml 默认 wikipedia/100000，控制下载与显存） |
| 评测数据 | 官方 CounterFact `data/counterfact.json`（memit.baulab.info，21919 条）前 50 条 |
| seed | 42 |

运行命令：

```
python examples/w2_rome_repro.py --data counterfact --n 50 --mom2-dataset wikitext2 --mom2-n-samples 10000 --fp16
python examples/w2_rome_repro.py --data zsre --n 50 --fp16
python examples/w2_rome_repro.py --data mquake --n 50 --fp16
python examples/w2_rome_repro.py --data zsre --model gpt-j-6B --n 10 --fp16   # 仅可行性尝试
```

脚本参数：`--data {counterfact,zsre,mquake}`、`--model {gpt2-xl,gpt-j-6B}`、`--n`、`--fp16`、`--mom2-dataset`、`--mom2-n-samples`。

## 2. 数据集规模

- 编辑案例：50 条（官方 CounterFact 前 50 条，`requested_rewrite.prompt.format(subject)` 展开）
- 每条编辑含：prompt / subject / ground_truth(target_true) / target_new / 1 条 paraphrase（`paraphrase_prompts[0]`）/ 10 条 neighborhood locality prompt（ground_truth 按官方 `run_AlphaEdit_editing.py` 约定填充 target_true）
- mom2 统计语料：wikitext2（Salesforce/wikitext wikitext-2-raw-v1），10000 样本，float32
- 补充数据源：
  - zsRE：官方 `data/zsre_mend_eval.json`（memit.baulab.info，12180 条），locality 用官方 `loc` 切片（nq 口径），映射 src/subject/alt/loc
  - MQuAKE：官方 `data/MQuAKE-CF-3k-v2.json`（princeton-nlp/MQuAKE，3000 条），取 `requested_rewrite[0]` 作编辑事实，`questions`（3 条多跳问题）+ `new_answer` 构造 portability p1 切片（无 rephrase/locality）

## 3. 指标结果

### 3.1 CounterFact（gpt2-xl，n=50）

| 协议指标 | EasyEdit 结果键 | 数值 |
|---|---|---|
| Reliability（rewrite_acc） | `rewrite_acc` | **1.000** |
| Generalisation（rephrase_acc） | `rephrase_acc` | **0.700** |
| Locality（neighborhood） | `locality.neighborhood_acc` | **0.842** |
| Portability | `portability.p1_acc` | null（该 split 未提供） |

说明：
- locality 使用官方 CounterFact 的 `neighborhood_prompts` 切片（评测协议允许 CounterFact 或 zsRE）。
- 每个案例的 rewrite / rephrase / locality 明细见结果 JSON 的 `per_case`。

### 3.2 zsRE（gpt2-xl，n=50）

| 协议指标 | EasyEdit 结果键 | 数值 |
|---|---|---|
| Reliability（rewrite_acc） | `rewrite_acc` | **0.997** |
| Generalisation（rephrase_acc） | `rephrase_acc` | **0.807** |
| Locality（nq 口径） | `locality.nq_acc` | **0.966** |
| Portability | `portability.p1_acc` | null（该 split 未提供） |

说明：zsRE 的 locality 官方切片为 `loc`（自然问题 nq 口径，pre_post_argmax_token_match 逐 token 匹配）。rephrase 高于 CounterFact 属数据集特性（zsRE 多轮重复事实更易泛化）。

### 3.3 MQuAKE portability（gpt2-xl，n=50）

| 协议指标 | EasyEdit 结果键 | 数值 |
|---|---|---|
| Reliability（rewrite_acc） | `rewrite_acc` | **0.947** |
| Portability（多跳 p1） | `portability.p1_acc` | **0.069** |

说明：p1 度量编辑后模型在多跳问题（`questions`，ground_truth 设为 `new_answer`）上的 token 匹配率。0.069 表明 ROME 单跳编辑几乎不向多跳推理转移（与 ROME 论文观察一致：单点编辑泛化到多跳组合场景的能力很弱）。此 split 无 rephrase/locality，`per_case` 为 null。

## 4. 显存与耗时

| 运行 | 峰值 CUDA 显存（fp16） | 耗时 | 说明 |
|---|---|---|---|
| CounterFact n=50（gpt2-xl） | 4632 MB | 389.7 s（约 7.5 s/案例） | 8GB 上限内安全 |
| zsRE n=50（gpt2-xl） | 5110 MB | 365.4 s（约 7.0 s/案例） | 含 nq locality 评测 |
| MQuAKE n=50（gpt2-xl） | 4789 MB | 343.1 s（约 6.8 s/案例） | 含 3×50 条多跳 p1 评测 |
| GPT-J-6B n=10（zsRE） | 峰值 11555 MB（超 8GB，依赖系统内存回填） | 编辑 10 条约 23 s；评测约 570 s/案例 | 见第 6 节结论 |

冒烟验证：CounterFact n=3 时 21.4s；zsRE n=3 时 rewrite=1.0/rephrase=1.0/nq=1.0；MQuAKE n=3 时 rewrite=0.833/p1=0.135。

备注：fp32 冒烟峰值约 8722 MB（超出 8GB 物理显存，有 OOM 风险），正式运行一律使用 fp16。

## 5. 与 W1 demo 的差异

| 维度 | W1 `w1_rome_demo.py` | W2 `w2_rome_repro.py` |
|---|---|---|
| 案例数 | 6（手工构造冒烟） | 50（官方 CounterFact） |
| 数据来源 | 手工 dict | `data/counterfact.json`（官方下载，21919 条可选） |
| 评测 | 仅演示 | reliability + generalisation + locality 全指标聚合 |
| locality | 未覆盖 | neighborhood 切片（10 prompt/case，pre_post_argmax_token_match） |
| 输出 | 控制台 | `results/eval/ROME_gpt2-xl_<ts>.json`（JSON 结构化） |
| mom2 语料 | wikitext（w1 若运行则使用） | wikitext2（小语料加速，代价是与官方 wikipedia 统计略有差异） |
| 显存策略 | 无 | fp16 默认开启（峰值 4.6GB vs fp32 8.7GB） |
| 调用参数 | ground_truth / subject 手工 | 自动从 requested_rewrite 提取 subject、target_true 等 |

本次更新还扩展了：`--data zsre`（nq locality 口径）、`--data mquake`（MQuAKE-CF portability p1，`questions`×`new_answer` 构造）、`--model gpt-j-6B`（读 `hparams/ROME/gpt-j-6B.yaml` 并覆盖 model_name/stats_dir/device/mom2/fp16）。

## 6. GPT-J-6B 可行性结论（8GB 显存环境）

尝试过程：
1. 下载 `ainize/gpt-j-6B-float16`（fp16 权重 `pytorch_model.bin` 11.2GB + 分词器/配置，已过滤 README 等冗余文件）到 `EasyEdit/hugging_cache/gpt-j-6B`；官方 `EleutherAI/gpt-j-6b` 仅 fp32（约 24GB）且无 safetensors，故采用 fp16 镜像。
2. 用 ROME 跑 zsRE 前 10 条（fp16，layers=[5]，mom2 wikitext2）：编辑阶段成功（10 条约 23s，样例 rewrite=1.0），评测阶段实测约 **570 s/案例**（10 条预计 1.5 小时+）。
3. 设备验证：`AutoModelForCausalLM.from_pretrained(..., fp16).to('cuda')` 可加载，峰值显存 11555 MB——**超出 8GB 物理显存，依赖 Windows 显存扩展/系统内存回填**，单次前向 1.1s 但整体吞吐不可接受；`device_map="auto"` 因非 safetensors 权重报错（需 offload_folder，磁盘 offload 后仍无法直接编辑）；4-bit（bitsandbytes）未安装且 ROME 需在 fp16/float 权重上更新 `fc_out` 层，4-bit 权重不可直接编辑。

**结论**：GPT-J-6B（6B 参数，fp16 权重 11.2GB）在 RTX 4070 Laptop 8GB 上**不可行**做完整 ROME 复现——权重+激活超物理显存，虽可借系统内存勉强加载运行，但评测速度不可接受（570s/案例），且 4-bit 量化与 ROME 权重更新不兼容。

**适配方案**（若后续需要 6B 级验证）：
- 升级至 ≥16GB 显存 GPU（4090/A6000 等）直接 fp16 跑通；
- 或采用 CPU offload（`device_map` + `offload_folder`），代价是更慢；
- 或换用 2.7B 及以下模型（如 gpt2-xl 已完成）验证方法可行性；
- 若要追求 ROME 在多跳（portability）上的真实表现，推荐用官方论文配套的 7B 级硬件配置（≥24GB 显存）。

## 7. 产物

- 脚本：`EasyEdit/examples/w2_rome_repro.py`（支持 counterfact/zsre/mquake + gpt2-xl/gpt-j-6B）
- 数据：`EasyEdit/data/counterfact.json`（21919 条）、`EasyEdit/data/zsre_mend_eval.json`（12180 条）、`EasyEdit/data/MQuAKE-CF-3k-v2.json`（3000 条）
- GPT-J 权重：`EasyEdit/hugging_cache/gpt-j-6B/`（fp16，11.2GB）
- 结果：
  - `EasyEdit/results/eval/ROME_gpt2-xl_20261018_180857.json`（CounterFact n=50）
  - `EasyEdit/results/eval/ROME_gpt2-xl_20261018_190605.json`（zsRE n=50）
  - `EasyEdit/results/eval/ROME_gpt2-xl_20261018_191619.json`（MQuAKE n=50）
*（内容由AI生成，仅供参考）*
*（内容由AI生成，仅供参考）*
