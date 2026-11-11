# gpt2-xl on CPU: the backbone check is worth ~1 custom core, not 6.3 GB of a
# card that two live streams are sharing.  Slower, but it cannot stall them.
$ErrorActionPreference = "Continue"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location $repo
& $py -u w6_rev3\w6_recall_readpath.py --offsets 0 --n 40 --templates prefix,fewshot `
    --device cpu `
    --model-dir "$repo\EasyEdit\hugging_cache\gpt2-xl" --tag _xl `
    *>> w6_rev3\readpath_xl.log 2>> w6_rev3\readpath_xl.err.log
Write-Output "XL CPU DONE"
