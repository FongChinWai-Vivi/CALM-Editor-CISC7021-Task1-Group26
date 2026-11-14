---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_18071fb0e00611f189bc525400248c00
    ReservedCode1: xZSRGtzv6ZQuYYzKyieVFce84McGf3YuKwXNWGfhQF2JNmmCwgVTrmpmwdwfR7hKeWC1rtJJZoGEaF6hrIg8iPxHPqP1dCAiugM2AMiwzhKlyv0dUPEFSjnFaQpbu7Wx+0zfDwytd1hVrpMHYBKOTFOZqsB6ND3MEZ7HN3W2DzclSHymP+5fkNmDRQQ=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_18071fb0e00611f189bc525400248c00
    ReservedCode2: xZSRGtzv6ZQuYYzKyieVFce84McGf3YuKwXNWGfhQF2JNmmCwgVTrmpmwdwfR7hKeWC1rtJJZoGEaF6hrIg8iPxHPqP1dCAiugM2AMiwzhKlyv0dUPEFSjnFaQpbu7Wx+0zfDwytd1hVrpMHYBKOTFOZqsB6ND3MEZ7HN3W2DzclSHymP+5fkNmDRQQ=
---

﻿# CALM editor — 網頁介面用法（白底＋藍色主題，澳門題材）

本文件對應 `EasyEdit/examples/demo_web.py`。同一套推理流程，只換掉畫面。

## 0. 為什麼不是 Java
- Java/Swing：需額外安裝JDK；Swing控件受限，要做出乾淨白藍觀感需自己畫；推論在Python裡，Java只能跨進程呼叫Python，錄屏多一層啟動與逾時風險。
- 現有tkinter：控件外觀受限，配色只能逐個widget指定，調整成本高。
- 本機網頁（採用）：只用標準庫`http.server`，無新依賴；Python後端照舊，瀏覽器負責渲染，CSS可完全控制；改樣式只改一段CSS，錄屏時瀏覽器全屏即可。

## 1. 啟動
專案根目錄：
```
.\.venv\Scripts\python.exe EasyEdit\examples\demo_web.py
```
終端機會印出 `serving http://127.0.0.1:8765/` 並自動開瀏覽器（用 `--no-open` 可關掉自動開啟）。

參數：
- `--model gpt2-small` / `gpt2-xl`：換模型，預設`gpt2-small`。
- `--read-template prefix`：記憶讀取模板，**必須保持prefix**，其他模板記憶注入不生效，預設`prefix`。
- `--port 8765`：連接埠，被佔用時換一個，預設`8765`。
- `--host 127.0.0.1`：只綁本機，不對外開放，預設`127.0.0.1`。
- `--no-open`：不自動開瀏覽器，預設`關`。

模型啟動時載入一次（約3–19秒），標題列右側狀態由`loading model`變成`ready`才可操作。之後每個請求共用同一個編輯器物件，用鎖排隊，行為與終端機版本一致。

## 2. 介面分區
左欄由上到下四段，右欄是記錄區：
- `1 · ask the model`：多行輸入框、`Ask the model`按鈕，只提問不動記憶。
- `2 · edit one fact`：Subject/Relation/New answer/Ground truth、`Apply edit`，對上方prompt做一次編輯。
- `3 · ask again (reads side memory)`：`Ask again with memory`（或輸入框按 `Shift+Enter`），對同一條prompt再問一次，這次走記憶讀取通道，用來現場證明翻轉不是編輯回應裡的一次性假象。
- `presets`：`Macau → China (flips)`、`Macau language → Portuguese (REJECT)`，一鍵填好欄位，不用手打。
- `session`：`Run scripted demo` / `Clear` / `Reset memory`，自動跑兩段演示、清記錄、清側記憶。

快捷鍵：`Ctrl+Enter`提問；在四個輸入框裡按`Enter`直接套用編輯。

顏色含義（記錄區卡片標籤）：`Prompt`藍、`Before`灰、`Result`綠；gate判定`WRITE`橙、`ROUTE`綠、`REJECT`紅；答案由紅（原答案）轉綠（改後）表示成功翻轉。
`Clear`只清右側記錄，側記憶不動；`Reset memory`回冷啟動：清64個shard，同時清gate的`recent_edits`與決策log（提示列會印`gate history cleared = N`）。若只清shard不清gate歷史，重置後第一次編輯會沿用上一輪的衝突計數，出現`conflicts_in_memory(1)`，而衝突路徑會繞過置信度門檻。

## 3. 輸入與預期輸出
兩段範例都以澳門為題材，數值為 gpt2-small + prefix 模板的實測結果。

### Macau — 經過側記憶翻轉（答錯 → 修正）
Prompt：
```
The capital of China is Beijing. Hong Kong is a SAR of China. Macau is a SAR of
```
欄位：Subject `Macau`、Relation `is a SAR of`、New answer `China`、Ground truth留空。
預期：
| 卡片 | 內容 |
|---|---|
| Before | `Macau.`，conf `0.68`（5個token機率平均）；同一張卡另印 `first token 0.56`、`answer-only 0.75 over 3 tok` 與逐token機率 `" Mac" 0.56 · "au" 0.88 · "." 0.81 · "\n" 0.15 · "\n" 1.00` |
| Result | edit `(Macau, is a SAR of)  Macau → China`；gate `ROUTE`，`route=side_memory`，`reason=default_moderate`；memory `stored note = China`。重置後再跑一次不會再出現`conflicts_in_memory(1)` |
| After | `China.`，標籤 `flipped by side memory`，`channel memory` |
| Recall | 同一 prompt 再問一次（`Ask again with memory`）：`China.`，`channel memory`、`memory injected yes`、`stored note = China`，標籤 `flip persists on a fresh query`；連按兩次結果一致 |

Relation 寫成 `is a SAR of` 而不是 `SAR`：prefix 模板會拼成 `[Macau] [is a SAR of] is [China]`，讀起來是一句完整英文，模型才會把 `China` 抄出來；用 `SAR` 時模型仍答 `Macau.`。
前面兩句 lead（China→Beijing、Hong Kong→China）是必要的：它把置信度推過 `gate_route_threshold`0.6，gate 才會判 ROUTE 並把注記寫進側記憶；單獨一句 `Macau is a SAR of` 置信度不足，只會走 WRITE 分支、不產生記憶注記，答案也就不會翻轉。

**答辯要守住的說法**：這個`0.68`是5個貪心token機率的平均，不是答案的機率。句尾兩個換行貢獻了0.15與1.00，把均值從0.56抬到0.68；首token（真正決定答案的` Mac`）只有0.56，本來不過門檻。所以此題的性質是「模型沒回答，只把主語回聲成答案」，屬提示格式失敗，不是「模型自信地答錯」。Before卡上那句`Before is a verbatim echo of the subject`就是為此而印，看到它就不要把它講成反事實注入。
`gate_route_threshold`與「5token平均」的口徑都沿用報告裡的定義，示範端不改；要改的是敘述，不是量測。

### Macau language — gate直接拒絕，不寫入
Prompt把 lead 換成葡萄牙語背景，最後一句問語言：
```
The capital of Portugal is Lisbon. The official language of Macau is
```
欄位：Subject `Macau`、Relation `language`、New answer `Portuguese`、Ground truth留空。
預期：
| 卡片 | 內容 |
|---|---|
| Before | `Portuguese.`，conf `0.50`；同卡另印 `first token 0.57`、`answer-only 0.50 over 5 tok` |
| Result | gate `REJECT`，`route=gate_reject`，`reason=current_answer_matches_target`；memory `nothing written`；After 不變，標籤 `unchanged, already correct` |
| Recall | 同一 prompt 再問一次：`Portuguese.`，`channel parameter`、`memory injected no`，標籤 `no memory needed, the model already answered this` |

這一段演示的是「模型已經答對就不要浪費參數」：gate 先用當前答案跟目標比對，一致就拒絕寫入。`Mcau`/`Macao` 這類拼寫變體會被當成不同實體，所以欄位裡統一只寫 `Macau`。

### 拼字防護
Subject沒出現在prompt裡時，後端不執行編輯，直接回一張橙色`Blocked`卡，例如把`Macau`打成不出現於prompt的詞時提示close match。避免錄屏時因為打錯字而出現無法解釋的結果。

### 已知限制（demo的邊界，答辯先講明）
1. **置信度口徑**：`conf`是5個貪心token機率平均，長度固定為5，句尾標點與換行會計入。同一句prompt改成`max_new_tokens=10`，衡量值會掉到0.55。「首token」更接近「模型對這一題有多確定」，但報告的數字用平均口徑，所以示範不換算，只把拆解一起印出來。
2. **gpt2-small上「自信答錯」很難湊**：實測14條澳門／常識prompt，凡不是主語回聲的錯誤答案，5token平均普遍落在0.13–0.50（`The capital of Myanmar is`→0.27、`The official language of Macau is`→0.26、`Macau was a colony of`→0.35、`Mount Everest is located in`→0.50）。在門檻0.6下，能判成ROUTE的基本只有回聲型。要做真正的反事實示範，得換更大的模型或調門檻，兩者都會讓本文件裡的數字對不上報告，因此維持現狀。
3. **衝突路徑不看門檻**：`recent_edits`已有同鍵時，即使conf只有0.55也會判`ROUTE`、`reason=conflicts_in_memory(n)`。這是gate的原設計（編輯過的事實優先），`Reset memory`現在會一併清掉這段歷史，避免重置後還帶著上一輪的衝突計數。
4. **`Ask the model` 與 `Ask again with memory` 不是同一件事**：前者直接問底座模型、不查側記憶，所以編輯後按它仍然得到 `Macau.`（`Before` 卡的數值就是它產生）；後者才重跑完整讀取流程（先答、再查記憶、必要時注入重答），編輯後按它得到 `China.`。要證明編輯生效，按的是 `Ask again with memory`，不是 `Ask the model`。

## 4. 錄屏建議
1. 先`Run scripted demo`跑一次確認狀態，然後`Reset memory`、重新整理頁面，從乾淨狀態錄。
2. 瀏覽器`F11`全屏，縮放100%，視窗寬度1400以上時左欄392px、右欄自動伸展。
3. 想跳過等待，可用`http://127.0.0.1:8765/?demo=1`：頁面載入後自動跑完兩段演示，適合純展示。
4. 對照截圖：`web_ui_macau_empty.png`（空狀態）、`web_ui_macau_filled.png`（兩段演示跑完）。

## 5. 疑難排解
| 現象 | 原因與處理 |
|---|---|
| `Address already in use` | 8765被佔用，加`--port 8766`或先關掉舊的python程序 |
| 頁面空白、狀態顯示`offline` | 後端沒起來或已被關掉，看終端機是否有traceback |
| 卡片遲遲不出現 | 正在算，`Ready`以外的狀態表示後端仍在處理；gpt2-small單次編輯約3–8秒 |
| preset 不是 Macau 題 | 舊的python程序還在跑舊程式碼，關掉重啟即可 |
| 想改配色 | 只改`PAGE`裡`:root`的變數，`--blue`、`--bg`、`--card`三個就決定整體觀感 |

## 6. 與舊介面的關係
`EasyEdit/examples/demo_gui.py`（tkinter視窗）保留為備用，配色已同步改成白底藍色，preset 亦已換成同一組澳門題材；兩者共用同樣的`Engine`邏輯與預期輸出。
*（内容由AI生成，仅供参考）*
