# CALM editor — 圖形介面用法與輸入／預期輸出對照

> 新版白底藍色網頁介面見 `docs/demo_web_ui.md`（`EasyEdit/examples/demo_web.py`），外觀更好且同樣走這套推理流程；本文件保留給 tkinter 視窗版。

本文件對應 `EasyEdit/examples/demo_gui.py`：把原本要在 PowerShell 逐行敲的 `demo_repl.py` 改成點選式視窗，不需要記指令。

---

## 0. 啟動

在專案根目錄執行一行：

```powershell
.\.venv\Scripts\python.exe EasyEdit\examples\demo_gui.py
```

常用參數：

| 參數 | 作用 | 預設 |
|---|---|---|
| `--model gpt2-small` / `gpt2-xl` | 換模型 | `gpt2-small` |
| `--read-template prefix` | 記憶讀取模板，**必須保持 prefix**，其他模板記憶注入不生效 | `prefix` |
| `--pause 0.4` | `Run scripted demo` 每步之間的停頓秒數 | `0.4` |
| `--selftest` | 自動跑完兩段演示後印出記錄並關窗（驗證用，錄屏不用） | 關 |
| `--screenshot <png>` | 存一張介面截圖 | 關 |

開窗後先載入模型（約 3 秒），標題列右側狀態由 `loading model ...` 變成 `ready` 才可操作。
終端機出現的 `timm` FutureWarning、`torch_dtype` deprecated、`attention_mask not set` 屬已知雜訊，不影響結果。

---

## 1. 介面分區

左欄五個區塊，右欄是記錄面板（等寬字體、彩色標籤）：

| 區塊 | 內容 | 用途 |
|---|---|---|
| 1. ASK THE MODEL | 多行輸入框 + `Ask (Ctrl+Enter)` | 只提問，看模型原本的答案與信心 |
| 2. EDIT ONE FACT | `Subject` / `Relation` / `New answer` / `Ground truth (optional)` 四格 + `Apply edit (Enter)` | 改一個事實（側記憶寫入） |
| 3. PRESETS | `Macau (flip via memory)` / `Macau language (gate says REJECT)` | 一鍵把 prompt 與四個欄位填好 |
| 4. SESSION | `Run scripted demo` / `Clear` / `Reset memory` | 自動演兩段、清空記錄、清空記憶 |

右欄每段輸出的標籤固定為：`PROMPT` → `BEFORE` → `EDIT` → `GATE` → `MEMORY` → `AFTER` → 結論。

---

## 2. 你要輸入什麼 → 預期看到什麼

### A. 只提問（不寫記憶）

| 位置 | 你要輸入的內容 |
|---|---|
| 1. ASK THE MODEL | `The capital of China is Beijing. Hong Kong is a SAR of China. Macau is a SAR of` |
| 動作 | 按 `Ask (Ctrl+Enter)` |

預期輸出：

```
PROMPT
The capital of China is Beijing. Hong Kong is a SAR of China. Macau is a SAR of
BEFORE   'Macau.'
         confidence 0.68
         5-token mean 0.68 · first token 0.56 · answer-only 0.75 over 3 tok
         tokens ' Mac' 0.56 'au' 0.88 '.' 0.81 '\n' 0.15 '\n' 1.00
```

> 兩段式 lead（China→Beijing、Hong Kong→China）是必要的：只有這樣置信度才會越過門檻 0.6、讓 gate 判 ROUTE。單獨一句 `Macau is a SAR of` 只有 0.44（`Hong Kong is a SAR of China. Macau is a SAR of` 也才 0.44），gate 會判 WRITE，記憶寫不進去、前後答案一致。
> 這 0.68 是五個貪心 token 機率的平均，句尾兩個換行（0.15、1.00）把均值從首 token 的 0.56 抬上來；所以 `BEFORE` 是「模型沒回答、只把主語回聲成答案」，屬提示格式失敗，答辯時不要說成「模型自信地答錯」。

---

### B. 改一個事實（會 flip）

| 位置 | 你要輸入的內容 |
|---|---|
| 1. ASK THE MODEL | `The capital of China is Beijing. Hong Kong is a SAR of China. Macau is a SAR of` |
| Subject | `Macau` |
| Relation | `is a SAR of` |
| New answer | `China` |
| Ground truth (optional) | 留空（留空＝自動取編輯前的答案 `Macau.`） |
| 動作 | 按 `Apply edit (Enter)` |

預期輸出（逐字）：

```
EDIT     (Macau, is a SAR of)  'Macau.'  ->  China
GATE     ROUTE   route=side_memory   reason=default_moderate
MEMORY   stored note = 'China'
AFTER    'China.\nThe capital'
         channel memory   flipped by side memory
```

判讀：`GATE ROUTE` + `MEMORY stored note` + `AFTER channel memory` + `flipped by side memory` 四項同時出現，才算演示成功。
`reason` 應為 `default_moderate`；若顯示 `conflicts_in_memory(1)`，代表沒有先按 `Reset memory`，gate 還記著上一輪的同鍵編輯。

---

### C. 改一個模型已經答對的事實（會被 gate 拒絕）

| 位置 | 你要輸入的內容 |
|---|---|
| 1. ASK THE MODEL | `The capital of Portugal is Lisbon. The official language of Macau is` |
| Subject | `Macau` |
| Relation | `language` |
| New answer | `Portuguese` |
| 動作 | 按 `Apply edit (Enter)` |

預期輸出（逐字）：

```
EDIT     (Macau, language)  'Portuguese.\n\nThe'  ->  Portuguese
GATE     REJECT   route=gate_reject   reason=current_answer_matches_target
MEMORY   nothing written, the edit carried no new fact
AFTER    'Portuguese.\n\nThe'
         channel parameter   unchanged, already correct
```

判讀：`REJECT` 且 `nothing written` 是**正確行為**，代表 gate 沒有亂寫；不要期待這裡出現 flip。此題 `BEFORE` 的 conf 是 0.50（first token 0.57），本來就低於門檻，但決定 REJECT 的是「當前答案等於目標」，不是置信度。

---

### D. 主體拼錯（你上次踩到的坑，現已攔截）

| 位置 | 你要輸入的內容 |
|---|---|
| Prompt | `... Macau is a SAR of`（把 Macau 打成 `Macua`） |
| Subject | `Macau` |
| 動作 | 按 `Apply edit (Enter)` |

預期輸出：

```
BLOCKED  'Macau' does not appear in the prompt (close match: 'Macua'). Fix the spelling first.
```

左欄下方提示列同步顯示同一句。**編輯不會送出、記憶不會被汙染**，把 prompt 的 `Macua` 改成 `Macau` 再按一次即可。

> 這是原本 REPL 的靜默失敗來源：主體拼錯 → subject 對不上 → gate 仍判 ROUTE 但側記憶查不到 key → `channel none` → `still wrong`。現在在送出前就擋掉。

---

### E. 一鍵按鈕

| 按鈕 | 效果 | 預期 |
|---|---|---|
| `Macau (flip via memory)` | 填入 Macau 的 prompt 與三個欄位 | 提示列 `Fields filled. Press Apply edit.`，再按 `Apply edit` 得到 B 段結果 |
| `Macau language (gate says REJECT)` | 填入澳門語言題的 prompt 與三個欄位 | 同上，再按 `Apply edit` 得到 C 段結果 |
| `Run scripted demo` | 自動依序跑 B、C 兩段 | 右欄依次出現兩段記錄，最後一行 `end of scripted demo` |
| `Clear` | 只清空右欄記錄 | 記憶保留，可用 `Ask` 驗證先前的 flip 仍在 |
| `Reset memory` | 清空側記憶（64 shards）＋gate 的 `recent_edits` 與決策 log | 提示列顯示 `memory cleared`；重建議重跑時不會再出現 `conflicts_in_memory(1)` |

`Clear` 與 `Reset memory` 的差別是答辯常被問到的點：前者只清畫面，後者回到冷啟動——側記憶與 gate 的編輯歷史一起清掉。只清記憶不清歷史會讓下一次同鍵編輯被判成「與記憶衝突」而直接 ROUTE，等於沿用上一輪的判斷。

---

## 3. 完整兩段演示的預期記錄（`Run scripted demo` 按下去後右欄全文）

```
model loaded, side memory empty (shards=64, max_per_shard=200)

Session 2 of the demo: the model answers as it is, then one fact is edited.

PROMPT
The capital of China is Beijing. Hong Kong is a SAR of China. Macau is a SAR of
BEFORE   'Macau.'
         confidence 0.68
         5-token mean 0.68 · first token 0.56 · answer-only 0.75 over 3 tok
         tokens ' Mac' 0.56 'au' 0.88 '.' 0.81 '\n' 0.15 '\n' 1.00

EDIT     (Macau, is a SAR of)  'Macau.'  ->  China
GATE     ROUTE   route=side_memory   reason=default_moderate
MEMORY   stored note = 'China'
AFTER    'China.\nThe capital'
         channel memory   flipped by side memory

PROMPT
The capital of Portugal is Lisbon. The official language of Macau is
BEFORE   'Portuguese.\n\nThe'
         confidence 0.50
         5-token mean 0.50 · first token 0.57 · answer-only 0.50 over 5 tok
         tokens ' Portuguese' 0.57 '.' 0.57 '\n' 0.24 '\n' 1.00 'The' 0.13

EDIT     (Macau, language)  'Portuguese.\n\nThe'  ->  Portuguese
GATE     REJECT   route=gate_reject   reason=current_answer_matches_target
MEMORY   nothing written, the edit carried no new fact
AFTER    'Portuguese.\n\nThe'
         channel parameter   unchanged, already correct

end of scripted demo
```

---

## 4. 已知限制（口徑一致，別在錄屏裡改口）

1. 置信度數字（0.68 / 0.50）與硬體、模型、模板綁定，換 `gpt2-xl` 或改模板會變；不要為了好看手改。
2. `conf` 是五個貪心 token 機率的平均（長度固定 5，含句尾標點與換行），同一句改成 `max_new_tokens=10` 會掉到 0.55；`first token` 才更接近「模型對這題有多確定」。報告用平均口徑，所以介面只把三種算法並列，不改口徑。
3. `Macau is a SAR of` 這一題的 `BEFORE 'Macau.'` 是主語回聲，不是模型答錯某個事實；答辯要照這個說法講（同樣實測：`The capital of Myanmar is` 只有 0.27、`The official language of Macau is` 0.26、`Macau was a colony of` 0.35，gpt2-small 在門檻 0.6 下幾乎湊不出「自信答錯」）。
4. 只有 `--read-template prefix` 能讓記憶注入生效；`instruct` / `qa` / `repeat` 會得到 `channel none`、答案不變。
5. 任何 `still wrong` 都是真實結果，照實呈現，不補一個假 flip。
6. 每次重啟程式記憶為空；同一 session 內按 `Reset memory` 才回到冷啟動（記憶與 gate 歷史一起清）。

---

## 5. 與舊版 REPL 的對照

| 舊（PowerShell） | 新（GUI） |
|---|---|
| `prompt> 打字提問` | 輸入框 + `Ask (Ctrl+Enter)` |
| `/edit Italy capital Milan` | 四格表單 + `Apply edit (Enter)` |
| `/state`、`/clear`、`/help`、`/quit` | `Reset memory`、`Clear`、左欄標籤即說明、關窗即退出 |
| 卡片在終端用 `+---+` 手繪，寬字元會錯位 | Tk 原生排版，無邊框錯位問題 |
*（内容由AI生成，仅供参考）*
