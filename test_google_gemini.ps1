# Test if your Google API key works with Gemini

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TESTING GOOGLE GEMINI API KEY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if key is loaded
if (-not $env:GOOGLE_API_KEY) {
    Write-Host "[ERROR] GOOGLE_API_KEY not loaded" -ForegroundColor Red
    Write-Host "Run: reload" -ForegroundColor Yellow
    exit 1
}

$apiKey = $env:GOOGLE_API_KEY
$keyPreview = $apiKey.Substring(0, [Math]::Min(15, $apiKey.Length)) + "..."

Write-Host "[INFO] Found key: $keyPreview" -ForegroundColor Green
Write-Host ""
Write-Host "[TEST] Testing Gemini API..." -ForegroundColor Yellow

try {
    $uri = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$apiKey"

    $body = @{
        contents = @(
            @{
                parts = @(
                    @{
                        text = "Reply with just: Gemini OK"
                    }
                )
            }
        )
    } | ConvertTo-Json -Depth 10

    $response = Invoke-RestMethod -Uri $uri -Method Post -Body $body -ContentType "application/json" -ErrorAction Stop

    $reply = $response.candidates[0].content.parts[0].text

    Write-Host ""
    Write-Host "[SUCCESS] Gemini API is working!" -ForegroundColor Green
    Write-Host "Response: $reply" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "YOUR GOOGLE KEY WORKS WITH GEMINI!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Free Tier: 1,500 requests per day" -ForegroundColor Yellow
    Write-Host "Cost: FREE" -ForegroundColor Yellow
    Write-Host "Quality: Very smart (GPT-4 level)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Next step: Let me integrate Gemini into Mamba AI" -ForegroundColor Cyan
    Write-Host ""

} catch {
    $errorMsg = $_.Exception.Message

    Write-Host ""
    Write-Host "[FAILED] Gemini API test failed" -ForegroundColor Red
    Write-Host ""

    if ($errorMsg -like "*API key not valid*" -or $errorMsg -like "*invalid*") {
        Write-Host "Your Google key is NOT enabled for Gemini" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Solution:" -ForegroundColor Cyan
        Write-Host "  1. Visit: https://aistudio.google.com/app/apikey" -ForegroundColor Gray
        Write-Host "  2. Create a NEW API key for Gemini (free)" -ForegroundColor Gray
        Write-Host "  3. Copy the new key" -ForegroundColor Gray
        Write-Host "  4. Add to .env:" -ForegroundColor Gray
        Write-Host "     GOOGLE_API_KEY=your_new_gemini_key" -ForegroundColor Gray
        Write-Host "  5. Run: reload" -ForegroundColor Gray
    } elseif ($errorMsg -like "*quota*") {
        Write-Host "You hit the daily limit (1,500 requests)" -ForegroundColor Yellow
        Write-Host "Try again tomorrow or use Ollama (unlimited local AI)" -ForegroundColor Cyan
    } else {
        Write-Host "Error: $errorMsg" -ForegroundColor Gray
    }
    Write-Host ""
}

Write-Host ""
