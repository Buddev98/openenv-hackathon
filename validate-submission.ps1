$baseUrl = $args[0]
if (-not $baseUrl) { $baseUrl = "http://localhost:7860" }

Write-Host "------------------------------------------------" -ForegroundColor Cyan
Write-Host "🚀 Validating OpenEnv Submission at $baseUrl" -ForegroundColor Cyan
Write-Host "------------------------------------------------" -ForegroundColor Cyan

# 1. Check Health
Write-Host -NoNewline "Checking /health... "
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    if ($health.status -eq "ok") {
        Write-Host "✅ PASS (200 OK)" -ForegroundColor Green
    } else {
        Write-Host "❌ FAIL (Status not ok)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ FAIL ($($_.Exception.Message))" -ForegroundColor Red
    exit 1
}

# 2. Check Reset
Write-Host -NoNewline "Checking /reset?task=easy... "
try {
    $reset = Invoke-RestMethod -Uri "$baseUrl/reset?task=easy" -Method Post
    if ($reset.emails -and $reset.emails.Count -gt 0) {
        Write-Host "✅ PASS (Emails received)" -ForegroundColor Green
    } else {
        Write-Host "❌ FAIL (No emails in response)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ FAIL ($($_.Exception.Message))" -ForegroundColor Red
    exit 1
}

# 3. Check Step
Write-Host -NoNewline "Checking /step (classify)... "
try {
    $body = @{
        action_type = "classify"
        email_id = "email_1"
        category = "spam"
    } | ConvertTo-Json
    $step = Invoke-RestMethod -Uri "$baseUrl/step" -Method Post -Body $body -ContentType "application/json"
    if ($null -ne $step.reward) {
        Write-Host "✅ PASS (Reward received: $($step.reward))" -ForegroundColor Green
    } else {
        Write-Host "❌ FAIL (No reward in response)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ FAIL ($($_.Exception.Message))" -ForegroundColor Red
    exit 1
}

Write-Host "------------------------------------------------" -ForegroundColor Cyan
Write-Host "🎉 ALL LOCAL CHECKS PASSED! " -ForegroundColor Green
Write-Host "Ready for Hugging Face Spaces deployment." -ForegroundColor Cyan
Write-Host "------------------------------------------------" -ForegroundColor Cyan
