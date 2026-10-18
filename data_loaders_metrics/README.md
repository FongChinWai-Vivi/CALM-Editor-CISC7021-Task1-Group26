---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 3948123ca1e8ab71efaa265617bdd19e_d349baabcaca11f196f9525400248c00
    ReservedCode1: 9EMyjCU3yfK6zmRV/m+NHr1cuv5MiGw89eXJOI9WMehIpyu2/dtqALmNdy6eHzLyoHgUhtD2YQuQ2v7VLgiVJDpX5XvzD7l7WnxvnE9UC+B7JiqWZ+yv7QJopExfj2BwtLaI8C9VLsU2fSYWyLs7wFLG0ZCTnW1mkVN4YnNwXgfAGZoHCRMo9uyq2fQ=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 3948123ca1e8ab71efaa265617bdd19e_d349baabcaca11f196f9525400248c00
    ReservedCode2: 9EMyjCU3yfK6zmRV/m+NHr1cuv5MiGw89eXJOI9WMehIpyu2/dtqALmNdy6eHzLyoHgUhtD2YQuQ2v7VLgiVJDpX5XvzD7l7WnxvnE9UC+B7JiqWZ+yv7QJopExfj2BwtLaI8C9VLsU2fSYWyLs7wFLG0ZCTnW1mkVN4YnNwXgfAGZoHCRMo9uyq2fQ=
---

# CALM W1 — Fong Hio Lam (MC654953)：数据加载器与评测指标

本工作包交付 CISC7021 Group26 CALM 项目 W1 阶段中 Fong Hio Lam 负责的
**数据管线**：CounterFact、zsRE、MQuAKE 三个数据集的数据加载器，以及
rewrite_acc / paraphrase / locality / portability 一套指标脚本。
代码自包含、无第三方运行时依赖，对齐 EasyEdit `easyeditor/dataset`
的字段约定与生成式评测协议，支持小规模 smoke test，满足 Gate 1
"三条数据管线与指标脚本均可运行"的验收要求。

---

## 1. 目录结构

```
w1_fonghiolam/
├── README.md                    # 本说明文档
├── loaders/
│   ├── __init__.py              # 加载器注册表 + build_loader()
│   ├── base.py                  # EditCase 数据类 + BaseDatasetLoader 基类
│   ├── counterfact.py           # CounterFactLoader
│   ├── zsre.py                  # ZsreLoader
│   └── mquake.py                # MQuaKELoader（兼容原始 MQuAKE 格式）
├── metrics.py                   # 指标实现：rewrite_acc/paraphrase/locality/portability
├── evaluate_pipeline.py         # CLI 端到端管线（loader + 指标）
└── sample_data/
    ├── counterfact_sample.json  # CounterFact 示例数据（3 条）
    ├── zsre_sample.json         # zsRE 示例数据（3 条）
    └── mquake_sample.json       # MQuAKE 示例数据（2 条，两种输入形态各 1）
```

## 2. 环境要求

- Python 3.9+（仅标准库，无第三方依赖）。
- 如需接入真实模型进行打分，需自行安装模型运行依赖
  （如 `transformers`、`torch`），见第 6 节示例。

## 3. 数据格式（与 EasyEdit 对齐）

统一中间结构为 `EditCase`（`loaders/base.py`）。原始 JSON 为记录列表，
每条记录字段遵循 EasyEdit 约定：

| 字段 | 含义 | 说明 |
|---|---|---|
| `request.prompt` | 编辑请求提示词 | 例如 `What is the official language of Spain?` |
| `request.subject` | 编辑主体 | 例如 `Spain` |
| `request.target_new` | 编辑后的新答案 | 可为字符串或 `{"str": ..., "id": ...}` |
| `request.target_true` | 编辑前的原始答案 | 同上 |
| `paraphrasess` | 改写提示词列表 | 兼容旧字段名 `paraphrase_prompts` |
| `locality_prompts.neighborhood_prompts` | 邻近无关提示词 | locality 评测用 |
| `locality_prompts.attribute_prompts` | 其他属性无关提示词 | locality 评测用 |
| `portability_prompts` | 多跳衍生问题（MQuAKE） | 亦可缺省，见下 |

**MQuAKE 兼容性**：`MQuaKELoader` 同时支持两种输入形态——

1. EasyEdit 风格（含显式 `portability_prompts`）；
2. 原始 MQuAKE 风格（`requested_rewrite` + `questions` 列表），加载器会
   自动把 `questions[].question` 提取为 `portability_prompts`。

顶层若为 `{"data": [...]}` / `{"instances": [...]}` 等包装对象也会被
自动解包。

## 4. 指标定义（评估协议）

所有指标采用 **归一化精确/包含匹配**（`metrics.normalize`：小写、去标点、
压缩空白），生成文本包含目标答案即计为命中。这与 EasyEdit 生成式评测
(exact-or-containment) 协议一致。

| 指标 | 定义 | 判定目标 |
|---|---|---|
| `rewrite_acc` | 编辑可靠性：编辑提示词 `request.prompt` 的生成 | 命中 `target_new` |
| `paraphrase_acc` | 泛化性：所有 `paraphrasess` 的生成 | 命中 `target_new` |
| `locality_neighborhood_acc` | 局部性：`neighborhood_prompts` 的生成 | 命中 `target_true` |
| `locality_attribute_acc` | 局部性：`attribute_prompts` 的生成 | 命中 `target_true` |
| `portability_acc` | 可移植性（多跳）：`portability_prompts` 的生成 | 命中 `target_new` |

计算公式：`accuracy = 命中条数 / 生成条数`；某组无数据时该指标返回 `0.0`
（`evaluate_cases` 同时返回 `n_cases` 供参考）。

## 5. 运行方式

### 5.1 Smoke test（无需模型，验证管线可跑）

```bash
python evaluate_pipeline.py --dataset counterfact --mock --limit 3
python evaluate_pipeline.py --dataset zsre      --mock --limit 3
python evaluate_pipeline.py --dataset mquake    --mock --limit 3
```

`--mock` 使用内置示例数据（`sample_data/`）与规则生成器，三条数据集
均应输出类似：

```json
{
  "rewrite_acc": 1.0,
  "paraphrase_acc": 1.0,
  "locality_neighborhood_acc": 1.0,
  "locality_attribute_acc": 1.0,
  "portability_acc": 1.0,
  "n_cases": 3
}
```

> 注意：`--mock` 的规则生成器只是格式/链路自检，**不得**用作正式评测结果。

### 5.2 对预生成预测打分（无需模型）

先由模型（或任何生成器）按下列 JSON 结构产出预测：

```json
{
  "rewrite":      {"0": ["<生成文本>"]},
  "paraphrase":   {"0": ["<改写1生成>", "<改写2生成>"]},
  "neighborhood": {"0": ["<邻近提示生成>"]},
  "attribute":    {"0": ["<属性提示生成>"]},
  "portability":  {"0": ["<多跳问题生成>"]}
}
```

然后：

```bash
python evaluate_pipeline.py --dataset counterfact \
    --data-path data/counterfact.json --predictions-json preds.json
```

`--limit N` 可只加载前 N 条用于小规模冒烟；缺省的组在结果中计 `0.0`。

### 5.3 在 Python 中接入真实模型

```python
from evaluate_pipeline import run_with_generator

def generate(prompt: str) -> str:
    # 例：使用 transformers 加载 Llama-3-8B / GPT-J-6B（W1 锁定模型清单）
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # out = model.generate(**inputs, max_new_tokens=32)
    # return tokenizer.decode(out[0], skip_special_tokens=True)
    ...

result = run_with_generator(
    dataset_name="counterfact",
    data_path=r"C:\path\to\counterfact.json",
    generate_fn=generate,
    limit=100,
)
print(result)
```

`generate_fn` 是纯函数式回调，可直接替换为 CALMEditor 编辑后模型的
生成接口，实现对编辑前后输出的统一打分。

### 5.4 直接使用加载器

```python
from loaders import build_loader

loader = build_loader("counterfact", data_path="counterfact.json", limit=100)
cases = loader.load()          # list[EditCase]
print(loader.size, cases[0].target_new)
```

## 6. 与 EasyEdit / CALMEditor 的对接说明

- **数据层**：`EditCase` 的字段与 EasyEdit `easyeditor/dataset`
  （`rome_datasets.py` / `mend_datasets.py` 等）使用的 JSON 约定一致
  （`request` / `paraphrasess` / `locality_prompts` /
  `portability_prompts`），团队按 `w1_alignment_decision.md` 采用
  `Llama-3-8B`、`GPT-J-6B` 正式评测，`gpt2-xl/gpt2-small` 仅快速验证。
- **指标层**：指标走原生 evaluate 层，不侵入 EasyEdit 组件；`metrics.py`
  可直接被 EasyEdit 的 `evaluate` 逻辑复用（同样归一化匹配口径）。
- **零侵入**：本包不修改 EasyEdit 源码，仅作为外部数据管线提供
  统一 `EditCase` 与打分函数，供后续 W2-W6 的 CALMEditor 评测循环调用。

## 7. Gate 1 验收对照

| 验收项 | 交付位置 | 状态 |
|---|---|---|
| CounterFact 数据加载器可运行 | `loaders/counterfact.py` + sample | ✅ |
| zsRE 数据加载器可运行 | `loaders/zsre.py` + sample | ✅ |
| MQuAKE 数据加载器可运行 | `loaders/mquake.py` + sample | ✅ |
| 指标脚本可运行（rewrite_acc/paraphrase/locality/portability） | `metrics.py` | ✅ |
| 端到端冒烟（小规模 smoke test） | `evaluate_pipeline.py --mock/--predictions-json` | ✅ |
| 使用说明（运行方式、输入输出格式） | 本 README | ✅ |
*（内容由AI生成，仅供参考）*
