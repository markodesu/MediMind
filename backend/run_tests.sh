#!/bin/bash
# Quick test runner for MediMind chatbot
# Usage: ./run_tests.sh

echo "🧪 Running MediMind Chatbot Tests..."
echo ""

# Check if backend is running
if ! curl -s http://localhost:8000/api/v1/ > /dev/null 2>&1; then
    echo "❌ Backend is not running!"
    echo "   Start it with: uvicorn app.main:app --reload"
    exit 1
fi

echo "✅ Backend is running"
echo ""

# Run tests
cd "$(dirname "$0")"
python test_chatbot_responses.py

echo ""
echo "📋 Check logs in: backend/logs/chatbot_$(date +%Y%m%d).log"

