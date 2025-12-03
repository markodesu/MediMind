"""
Logging utility for MediMind chatbot interactions.
Logs all user messages and bot responses for quality assurance and debugging.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Setup logging directory
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Setup Python logger
logger = logging.getLogger("medimind")
logger.setLevel(logging.INFO)

# Create file handler with daily rotation
log_file = LOG_DIR / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers if not already added
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def log_interaction(
    user_message: str,
    bot_response: str,
    confidence: float,
    safe: bool = True,
    history_length: int = 0,
    response_time_ms: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log a chatbot interaction with all relevant details.
    
    Args:
        user_message: User's input message
        bot_response: Bot's response
        confidence: Confidence score (0.0-1.0)
        safe: Whether response was deemed safe
        history_length: Number of previous messages in conversation
        response_time_ms: Response time in milliseconds
        metadata: Additional metadata dictionary
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "bot_response": bot_response,
        "confidence": confidence,
        "safe": safe,
        "history_length": history_length,
        "response_time_ms": response_time_ms,
        "metadata": metadata or {}
    }
    
    # Log as JSON for structured logging
    logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    # Also log a human-readable summary
    summary = (
        f"User: {user_message[:100]}{'...' if len(user_message) > 100 else ''} | "
        f"Bot: {bot_response[:100]}{'...' if len(bot_response) > 100 else ''} | "
        f"Confidence: {confidence:.2f} | Safe: {safe}"
    )
    logger.info(f"INTERACTION: {summary}")


def log_test_result(
    test_name: str,
    passed: bool,
    user_message: str,
    bot_response: str,
    expected_behavior: str,
    actual_behavior: str,
    confidence: Optional[float] = None
):
    """
    Log test results for quality assurance.
    
    Args:
        test_name: Name of the test case
        passed: Whether test passed
        user_message: Test input message
        bot_response: Bot's response
        expected_behavior: What was expected
        actual_behavior: What actually happened
        confidence: Confidence score if available
    """
    status = "✅ PASS" if passed else "❌ FAIL"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "test_result",
        "test_name": test_name,
        "status": status,
        "user_message": user_message,
        "bot_response": bot_response,
        "expected_behavior": expected_behavior,
        "actual_behavior": actual_behavior,
        "confidence": confidence
    }
    
    logger.info(json.dumps(log_entry, ensure_ascii=False))
    logger.info(f"TEST {status}: {test_name} | Expected: {expected_behavior} | Actual: {actual_behavior}")


def get_recent_logs(limit: int = 50) -> list:
    """
    Get recent log entries from today's log file.
    
    Args:
        limit: Maximum number of entries to return
        
    Returns:
        List of log entries (most recent first)
    """
    log_file = LOG_DIR / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"
    
    if not log_file.exists():
        return []
    
    entries = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Get last N lines
            for line in lines[-limit:]:
                if line.strip():
                    try:
                        # Try to parse as JSON
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError:
                        # If not JSON, skip
                        continue
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
    
    return entries

