param()
$repo = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot"
$python = "C:\Users\Owner\AppData\Local\Programs\Python\Python311\python.exe"
$dotenv = "C:\Users\Owner\OneDrive\Desktop\GAME_PLAN_2K28\.env"
Set-Location $repo
& $python "scripts/trade_top3.py" --ensure-scan --cap 900 --dotenv $dotenv | Out-Null
