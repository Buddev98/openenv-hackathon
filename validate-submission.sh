#!/bin/bash

# OpenEnv Submission Validator
BASE_URL=${1:-"http://localhost:7860"}

echo "------------------------------------------------"
echo "🚀 Validating OpenEnv Submission at $BASE_URL"
echo "------------------------------------------------"

# 1. Check Health
echo -n "Checking /health... "
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
if [ "$HEALTH" == "200" ]; then
    echo "✅ PASS (200 OK)"
else
    echo "❌ FAIL ($HEALTH)"
    exit 1
fi

# 2. Check Reset
echo -n "Checking /reset?task=easy... "
RESET=$(curl -s -X POST "$BASE_URL/reset?task=easy")
if echo "$RESET" | grep -q "emails"; then
    echo "✅ PASS (Emails received)"
else
    echo "❌ FAIL (No emails in response)"
    exit 1
fi

# 3. Check Step
echo -n "Checking /step (classify)... "
STEP=$(curl -s -X POST "$BASE_URL/step" \
     -H "Content-Type: application/json" \
     -d '{"action_type": "classify", "email_id": "email_1", "category": "spam"}')

if echo "$STEP" | grep -q "reward"; then
    echo "✅ PASS (Reward received)"
else
    echo "❌ FAIL (No reward in response)"
    exit 1
fi

echo "------------------------------------------------"
echo "🎉 ALL LOCAL CHECKS PASSED! "
echo "Ready for Hugging Face Spaces deployment."
echo "------------------------------------------------"
