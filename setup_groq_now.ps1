# GROQ SETUP - GET AI WORKING NOW (2 MINUTES)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GROQ SETUP - FREE AI (2 MINUTES)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1: Get Groq API Key" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Open: https://console.groq.com/" -ForegroundColor Gray
Write-Host "  2. Sign up (free, no credit card)" -ForegroundColor Gray
Write-Host "  3. Click 'API Keys'" -ForegroundColor Gray
Write-Host "  4. Click 'Create API Key'" -ForegroundColor Gray
Write-Host "  5. Copy the key (starts with gsk_...)" -ForegroundColor Gray
Write-Host ""
Write-Host "Press ENTER when you have the key..." -ForegroundColor Yellow
Read-Host

Write-Host ""
Write-Host "Step 2: Enter Your Groq API Key" -ForegroundColor Yellow
Write-Host ""
$groqKey = Read-Host "Paste your Groq API key here"

if (-not $groqKey -or $groqKey.Length -lt 20) {
    Write-Host ""
    Write-Host "[ERROR] Invalid key. Please run this script again." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[OK] Key received!" -ForegroundColor Green
Write-Host ""

# Add to .env file
$envPath = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\.env"

Write-Host "Step 3: Adding to .env file..." -ForegroundColor Yellow

$envContent = Get-Content $envPath -Raw

if ($envContent -match "GROQ_API_KEY") {
    # Replace existing
    $envContent = $envContent -replace "GROQ_API_KEY=.*", "GROQ_API_KEY=$groqKey"
} else {
    # Add new
    $envContent += "`n`n# GROQ API (FREE AI)`nGROQ_API_KEY=$groqKey`n"
}

Set-Content -Path $envPath -Value $envContent
Write-Host "[OK] Added to .env file!" -ForegroundColor Green
Write-Host ""

# Test the key
Write-Host "Step 4: Testing Groq API..." -ForegroundColor Yellow

try {
    $headers = @{
        "Authorization" = "Bearer $groqKey"
        "Content-Type" = "application/json"
    }

    $body = @{
        model = "llama-3.1-70b-versatile"
        messages = @(
            @{
                role = "user"
                content = "Reply with just: Groq OK"
            }
        )
        max_tokens = 10
    } | ConvertTo-Json -Depth 10

    $response = Invoke-RestMethod -Uri "https://api.groq.com/openai/v1/chat/completions" `
                                   -Method Post `
                                   -Headers $headers `
                                   -Body $body `
                                   -ErrorAction Stop

    $reply = $response.choices[0].message.content

    Write-Host "[OK] Groq API is working!" -ForegroundColor Green
    Write-Host "Response: $reply" -ForegroundColor Cyan
    Write-Host ""

} catch {
    Write-Host "[ERROR] Groq API test failed" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Check your API key at: https://console.groq.com/" -ForegroundColor Yellow
    exit 1
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "GROQ IS READY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "What you have now:" -ForegroundColor Yellow
Write-Host "  - FREE unlimited AI" -ForegroundColor Cyan
Write-Host "  - Llama 3.1 70B (very smart)" -ForegroundColor Cyan
Write-Host "  - FASTEST inference in the world" -ForegroundColor Cyan
Write-Host "  - No billing, no limits" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. I'll add Groq support to Mamba AI (5 min)" -ForegroundColor Gray
Write-Host "  2. Run: reload" -ForegroundColor Gray
Write-Host "  3. Use: talk hello, test my Groq AI" -ForegroundColor Gray
Write-Host ""
