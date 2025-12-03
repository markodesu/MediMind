#!/usr/bin/env python3
"""
Quick test script to see the improved responses.
Tests various question types to verify concise, clear answers.
"""

import requests
import json

API_URL = "http://127.0.0.1:8000/api/v1/chat"

test_questions = [
    "What should I do if I have mild food poisoning?",
    "What is a headache?",
    "How to treat a cold?",
    "I have severe chest pain",
    "What disease do I have?",
    "What is flu?",
    "I got flu, what to do?",
]

print("=" * 70)
print("Testing Improved MediMind Responses")
print("=" * 70)
print()

for i, question in enumerate(test_questions, 1):
    print(f"Test {i}: {question}")
    print("-" * 70)
    
    try:
        response = requests.post(
            API_URL,
            json={"message": question, "history": []},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data['answer']}")
            print(f"Confidence: {data['confidence']:.2f}")
            print(f"Safe: {data.get('safe', 'N/A')}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    print("=" * 70)
    print()

print("✅ Testing complete!")

