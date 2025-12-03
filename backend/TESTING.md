# MediMind Testing and Logging Guide

This guide explains how to test chatbot responses and review logs after making changes.

## Quick Start

### Run Tests After Changes

```bash
cd backend
python test_chatbot_responses.py
```

Or use the convenience script:
```bash
cd backend
./run_tests.sh
```

## What Gets Tested

The test suite covers:

1. **Basic Health Questions** - General health information queries
2. **Urgent/Safety Scenarios** - Nose bleeding, chest pain, severe symptoms
3. **Women's Health** - Period cramps, menstrual pain, severe cases
4. **Doctor Contact Information** - Requests for Dr. Kyal's contact details
5. **Common Symptoms** - Cold, stomach ache, stress, etc.
6. **Low Confidence Scenarios** - Gibberish, unclear questions
7. **Human-like Interaction** - Follow-up questions and conversational flow

## Logging

### Automatic Logging

All chatbot interactions are automatically logged to:
```
backend/logs/chatbot_YYYYMMDD.log
```

Each log entry includes:
- Timestamp
- User message
- Bot response
- Confidence score
- Safety flag
- Response time
- Conversation history length
- Additional metadata

### View Recent Logs

```python
from app.utils.logger import get_recent_logs

logs = get_recent_logs(limit=50)
for log in logs:
    print(f"{log['timestamp']}: {log['user_message']} -> {log['bot_response'][:50]}...")
```

### Log Format

Logs are stored as JSON lines for easy parsing:

```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "user_message": "I have a headache",
  "bot_response": "For a mild headache, rest...",
  "confidence": 0.7,
  "safe": true,
  "history_length": 0,
  "response_time_ms": 234.5,
  "metadata": {}
}
```

## Test Results

After running tests, you'll see:
- ✅ Pass/Fail status for each test
- Confidence scores
- Expected vs actual behavior
- Summary statistics

## Adding New Tests

Edit `test_chatbot_responses.py` and add test cases:

```python
self.test_case(
    "Test name",
    "User message",
    expected_contains=["keyword1", "keyword2"],
    expected_not_contains=["bad_word"],
    min_confidence=0.5,
    should_redirect=False
)
```

## Best Practices

1. **Run tests after every change** to ensure nothing broke
2. **Check logs** if a test fails to understand what happened
3. **Add tests** for new features or edge cases
4. **Review logs regularly** to spot patterns or issues

## Troubleshooting

### Backend not running
```bash
cd backend
uvicorn app.main:app --reload
```

### Tests failing
- Check logs in `backend/logs/`
- Verify backend is running on port 8000
- Check that model is loaded correctly

### No logs appearing
- Ensure `backend/logs/` directory exists (created automatically)
- Check file permissions
- Verify logger is imported correctly

