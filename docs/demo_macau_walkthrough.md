---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_15b6d8fee00611f189bc525400248c00
    ReservedCode1: CxlCVXJyTjH3yfxfGCvuwH4Owntvm2O3tB2Yw347I1EE4p+EcOc6af9DRn02k4EtblmrjmrFW2Trd5Ei4/h0jEDKRo3roHNUaqo1d3uzgjlyn/+WXjv2HbMbpPnepFolcEE1b4YV/BsS27Ai0XpbIczrgSz+4v3mCjf7UKR15OFlZcnhPESIlJFCpUY=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_15b6d8fee00611f189bc525400248c00
    ReservedCode2: CxlCVXJyTjH3yfxfGCvuwH4Owntvm2O3tB2Yw347I1EE4p+EcOc6af9DRn02k4EtblmrjmrFW2Trd5Ei4/h0jEDKRo3roHNUaqo1d3uzgjlyn/+WXjv2HbMbpPnepFolcEE1b4YV/BsS27Ai0XpbIczrgSz+4v3mCjf7UKR15OFlZcnhPESIlJFCpUY=
---

﻿# 澳門題材示範：逐步操作腳本（按哪個按鈕 → 看到什麼）

對應程式：`EasyEdit/examples/demo_web.py`（網頁版）與 `demo_gui.py`（tkinter 備用版），兩者 preset 相同。
模型：`gpt2-small` + `prefix` 讀取模板。以下數值為實測值。

---

## 0. 開場前 30 秒

| # | 動作 | 螢幕上會看到 |
|---|---|---|
| 0.1 | 在專案根目錄執行 `.\.venv\Scripts\python.exe EasyEdit\examples\demo_web.py` | 終端機印 `building editor ...`，約 3–20 秒後印 `serving http://127.0.0.1:8765/`，瀏覽器自動打開 |
| 0.2 | 看右上角狀態膠囊 | 由 `loading model` 變 `ready`；仍是 `loading model` 就不要按任何按鈕 |
| 0.3 | 按一下左欄 `Reset memory` | 下方提示變成 `side memory cleared (total_stored = 0)`，右欄記錄清空 |

---

## 1. 第一段：澳門（答錯 → gate 判 ROUTE → 側記憶把答案翻成 China）

| # | 按哪裡 | 會發生什麼 |
|---|---|---|
| 1.1 | `presets` 區第一顆 `Macau → China (flips)` | 輸入框自動填好：Prompt = `The capital of China is Beijing. Hong Kong is a SAR of China. Macau is a SAR of`；Subject `Macau`、Relation `is a SAR of`、New answer `China`、Ground truth 空白；提示 `Fields filled. Press Apply edit.` |
| 1.2 | 按 `Apply edit`（或在任一輸入框按 `Enter`） | 狀態變 `editing`，右欄先出現一張 `query` 卡，再出現一張 `edit one fact` 卡；約 3–8 秒後狀態回 `ready` |
| 1.3 | 看第一張卡 | `Prompt` 是你輸入的句子；`Before` = `Macau.`，conf `0.68` —— 模型把主語回聲成答案，明顯答錯 |
| 1.4 | 看第二張卡 | 標題 `edit one fact  (Macau, is a SAR of)`；`Result` 顯示 `Macau → China`，gate 徽章 `ROUTE`（綠色），`route=side_memory`，`reason=default_moderate`（第二次跑同一事實時顯示 `conflicts_in_memory(1)`），memory 行 `stored note = China` |
| 1.5 | 看第二張卡的 `After` | `China.`，徽章 `flipped by side memory`，`channel memory` —— 這就是要在報告裡強調的「側記憶翻轉」 |
| 1.6 | 再按一次同一顆 preset，再按 `Apply edit` | 這次 `Before` 仍是 `Macau.`（原始模型沒被改），gate 顯示 `ROUTE` 且 `reason=conflicts_in_memory(1)`，`After` 仍是 `China.`。用來解釋「非參數式編輯、基底權重不動」 |

> 講解一句話：*「模型本身會把 Macau 答成 Macau，gate 判斷這是它不確定的事實，於是不浪費參數，把 China 寫進側記憶；下一次讀取時由記憶補上，答案就翻了。」*

---

## 2. 第二段：澳門官方語言（模型已答對 → gate 判 REJECT，不寫記憶）

| # | 按哪裡 | 會發生什麼 |
|---|---|---|
| 2.1 | `presets` 第二顆 `Macau language → Portuguese (REJECT)` | Prompt 換成 `The capital of Portugal is Lisbon. The official language of Macau is`；Subject `Macau`、Relation `language`、New answer `Portuguese`、Ground truth 空白 |
| 2.2 | 按 `Apply edit` | `Before` = `Portuguese.`，conf `0.50` |
| 2.3 | 看結果卡 | gate 徽章 `REJECT`（紅色），`route=gate_reject`，`reason=current_answer_matches_target`，memory 行 `nothing written` |
| 2.4 | 看 `After` | 仍是 `Portuguese.`，徽章 `unchanged, already correct` —— 沒有寫入、沒有翻轉，正是預期結果 |

> 講解一句話：*「模型已經答對，gate 直接拒絕，省下一次參數寫入；這說明守門器不是無條件接受編輯。」*

---

## 3. 一次跑完兩段（錄屏用）

| # | 按哪裡 | 會發生什麼 |
|---|---|---|
| 3.1 | 按 `session` 區的 `Run scripted demo` | 狀態變 `scripted demo`，自動依序填欄位並 `Apply edit` 兩次，中間停 260/420 毫秒，右欄陸續出現兩段共四張卡；結束時提示 `end of scripted demo`，狀態回 `ready` |
| 3.2 | 或直接在網址後加 `?demo=1` 再刷新 | 頁面 `ready` 後延遲 400 毫秒自動跑完同樣兩段，適合不想手動按的展示 |

---

## 4. 其他按鈕的作用（被問到時用得上）

| 按鈕 | 作用 |
|---|---|
| `Ask the model`（或 `Ctrl+Enter`） | 只提問、不編輯；`Before` 卡的數值就是它產生 |
| `Clear` | 只清右欄記錄，側記憶不動，狀態膠囊不變 |
| `Reset memory` | 清掉 64 個 shard 的側記憶，回到冷啟動；示範前務必先按 |
| `Ground truth`（留空） | 留空時後端拿當前答案當 ground truth，所以不用手填；填了會改變 gate 的比對基準 |

---

## 5. 若被問「為什麼 Relation 要寫 `is a SAR of`」

prefix 模板會拼成一句英文注記：`[Macau] [is a SAR of] is [China]`。
- 寫 `is a SAR of` → 注記是通順句子，模型會把 `China` 抄出來，翻轉成立。
- 只寫 `SAR` → 注記變成 `[Macau] [SAR] is [China]`，模型仍答 `Macau.`，看不到翻轉。
- 同理，前兩句 lead（`The capital of China is Beijing.` / `Hong Kong is a SAR of China.`）把置信度推到 `0.68`，高於 `gate_route_threshold = 0.6`，gate 才會判 ROUTE 去寫側記憶；單獨一句 `Macau is a SAR of` 置信度不足，會落到 WRITE 分支，不會產生記憶注記。

---

## 6. 截圖對照

| 檔案 | 內容 |
|---|---|
| `docs/images/web_ui_macau_empty.png` | 空狀態：左欄四段表單、右欄 `Nothing yet` |
| `docs/images/web_ui_macau_filled.png` | `?demo=1` 跑完兩段後：四張卡、`ROUTE` / `REJECT` 徽章、`China.` 與 `Portuguese.` |

---

## 7. 數值速查（gpt2-small + prefix，實測）

| 段落 | Subject / Relation / New answer | Before（conf） | gate | After |
|---|---|---|---|---|
| 澳門地位 | `Macau` / `is a SAR of` / `China` | `Macau.`（0.68） | `ROUTE` → `side_memory` | `China.`（flipped） |
| 澳門語言 | `Macau` / `language` / `Portuguese` | `Portuguese.`（0.50） | `REJECT` → `gate_reject` | `Portuguese.`（unchanged） |
*（内容由AI生成，仅供参考）*
