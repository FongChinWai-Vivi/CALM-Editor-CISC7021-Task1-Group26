$py = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26\.venv\Scripts\python.exe"
Set-Location "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
& $py -u w6_rev3\w6_recall_readpath.py --offsets 800,900,1000,1100,1200 --n 100 --templates prefix,fewshot *>> w6_rev3\readpath_fresh.log 2>> w6_rev3\readpath_fresh.err.log
& $py -u w6_rev3\w6_recall_readpath.py --offsets 0 --n 40 --templates prefix,fewshot --model-dir "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26\EasyEdit\hugging_cache\gpt2-xl" --tag _xl *>> w6_rev3\readpath_xl.log 2>> w6_rev3\readpath_xl.err.log
Write-Output "READPATH2 DONE"
