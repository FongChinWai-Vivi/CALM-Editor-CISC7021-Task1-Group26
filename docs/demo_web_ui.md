# CALM editor — 網頁介面用法（白底＋藍色主題）

本文件對應 `EasyEdit/examples/demo_web.py`。它把點選式 demo 換成本機網頁介面：同一個 `CALMEditor` 物件、同一套推理流程，只換掉畫面。白底、藍色重點色，卡片式記錄區，字體與間距由 CSS 控制，不受 Windows 控件主題影響。

## 0. 為什麼不是 Java

| 方案 | 問題 |
|---|---|
| Java / Swing 重寫 | 需要額外安裝 JDK；Swing 原生控件同樣受限於系統外觀，要做出乾淨的白藍觀感得自己畫一堆；而且推論在 Python（EasyEdit）裡，Java 只能透過跨進程呼叫 Python，錄屏時多一層啟動與逾時風險 |
| 現有 tkinter | 已經能用，但控件外觀受限，配色只能逐個 widget 指定，調整成本高 |
| 本機網頁（採用） | 只用標準庫 `http.server`，無新依賴；Python 後端照舊，瀏覽器負責渲染，CSS 可完全控制；改樣式只改一段 CSS，錄屏時瀏覽器全屏即可 |

## 1. 啟動

在專案根目錄：

```powershell
.\.venv\Scripts\python.exe EasyEdit\examples\demo_web.py
```

終端機會印出 `serving http://127.0.0.1:8765/` 並自動開瀏覽器（用 `--no-open` 可關掉自動開啟）。

| 參數 | 作用 | 預設 |
|---|---|---|
| `--model gpt2-small` / `gpt2-xl` | 換模型 | `gpt2-small` |
| `--read-template prefix` | 記憶讀取模板，**必須保持 prefix**，其他模板記憶注入不生效 | `prefix` |
| `--port 8765` | 連接埠，被佔用時換一個 | `8765` |
| `--host 127.0.0.1` | 只綁本機，不對外開放 | `127.0.0.1` |
| `--no-open` | 不自動開瀏覽器 | 關 |

模型在啟動時載入一次（約 3–19 秒，視快取），標題列右側狀態由 `loading model` 變成 `ready` 才可操作。之後每個請求共用同一個編輯器物件，用鎖排隊，行為與終端機版本一致。

## 2. 介面分區

左欄由上到下四段，右欄是記錄區：

| 區塊 | 內容 | 用途 |
|---|---|---|
| `1 · ask the model` | 多行輸入框、`Ask the model` 按鈕 | 只提問，不動記憶 |
| `2 · edit one fact` | Subject / Relation / New answer / Ground truth、`Apply edit` | 對上方 prompt 做一次編輯 |
| `presets` | `Italy → Milan (flips)`、`Spain → Madrid (REJECT)` | 一鍵填好欄位，不用手打 |
| `session` | `Run scripted demo` / `Clear` / `Reset memory` | 自動跑兩段演示、清記錄、清側記憶 |

快捷鍵：`Ctrl+Enter` 提問；在四個輸入框裡按 `Enter` 直接套用編輯。

顏色含義（記錄區卡片標籤）：`Prompt` 藍、`Before` 灰、`Result` 綠；gate 判定 `WRITE` 橙、`ROUTE` 綠、`REJECT` 紅；答案由紅（原答案）轉綠（改後）表示成功翻轉。

`Clear` 只清右側記錄，側記憶不動（再問 Italy 仍然答 Milan）；`Reset memory` 清 64 個 shard。

## 3. 輸入與預期輸出

同一份對照，顏色與卡片格式見截圖。

### Italy — 經過側記憶翻轉

Prompt：

```
The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is
```

欄位：Subject `Italy`、Relation `capital`、New answer `Milan`、Ground truth 留空。

預期：

| 卡片 | 內容 |
|---|---|
| Before | `Rome. The capital of`，conf `0.74` |
| Result | edit `(Italy, capital)  Rome. The capital of → Milan`；gate `ROUTE`，`route=side_memory`，`reason=default_moderate`（記憶裡已有同一事實時顯示 `conflicts_in_memory(1)`）；memory `stored note = Milan` |
| After | `Milan. The capital of`，標籤 `flipped by side memory`，`channel memory` |

### Spain — gate 直接拒絕，不寫入

Prompt 把最後一句換成 `The capital of Spain is`，其餘同上。

欄位：Subject `Spain`、Relation `capital`、New answer `Madrid`、Ground truth 留空。

預期：

| 卡片 | 內容 |
|---|---|
| Before | `Madrid. The capital of`，conf `0.75` |
| Result | gate `REJECT`，`route=gate_reject`，`reason=current_answer_matches_target`；memory `nothing written`；After `Madrid. The capital of`，標籤 `unchanged, already correct` |

`0.74` / `0.75` 是 gpt2-small + prefix 模板下這兩個 prompt 的實測值，換模型或換模板會變，不要手改數字。`still wrong` 若出現，照實呈現，不要修飾。

### 拼字防護

Subject 沒出現在 prompt 裡時，後端不執行編輯，直接回一張橙色 `Blocked` 卡，例如把 `Italy` 打成不出現在 prompt 的詞時提示 close match。避免錄屏時因為打錯字而出現無法解釋的結果。

## 4. 錄屏建議

1. 先 `Run scripted demo` 跑一次確認狀態，然後 `Reset memory`、重新整理頁面，從乾淨狀態錄。
2. 瀏覽器 `F11` 全屏，縮放 100%，視窗寬度 1400 以上時左欄 392px、右欄自動伸展。
3. 想跳過等待，可用 `http://127.0.0.1:8765/?demo=1`：頁面載入後自動跑完兩段演示，適合純展示。
4. 對照截圖：`output/web_ui_empty.png`（空狀態）、`output/web_ui_filled.png`（兩段演示跑完）。

## 5. 疑難排解

| 現象 | 原因與處理 |
|---|---|
| `Address already in use` | 8765 被佔用，加 `--port 8766` 或先關掉舊的 python 程序 |
| 頁面空白、狀態顯示 `offline` | 後端沒起來或已被關掉，看終端機是否有 traceback |
| 卡片遲遲不出現 | 正在算，`Ready` 以外的狀態表示後端仍在處理；gpt2-small 單次編輯約 3–8 秒 |
| 想改配色 | 只改 `PAGE` 裡 `:root` 的變數，`--blue`、`--bg`、`--card` 三個就決定整體觀感 |

## 6. 與舊介面的關係

`EasyEdit/examples/demo_gui.py`（tkinter 視窗）保留為備用，配色已同步改成白底藍色；兩者共用同樣的 `Engine` 邏輯與預期輸出，`docs/demo_gui_quickstart.md` 仍適用於它。
