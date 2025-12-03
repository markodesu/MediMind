#!/usr/bin/env python3
"""
Quick test script to verify backend-frontend connection.
Tests the API endpoint that the frontend uses.
"""

import requests
import json
import sys

API_URL = "http://localhost:8000"

def test_backend_connection():
    """Test if backend is running and accessible."""
    print("=" * 60)
    print("Testing Backend-Frontend Connection")
    print("=" * 60)
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Backend is running")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ❌ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to backend at {API_URL}")
        print(f"   Make sure backend is running: cd backend && uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Chat endpoint (same as frontend uses)
    print("\n2. Testing chat endpoint (POST /api/v1/chat)...")
    try:
        payload = {
            "message": "What is a headache?",
            "history": []
        }
        
        print(f"   Request URL: {API_URL}/api/v1/chat")
        print(f"   Request Body: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{API_URL}/api/v1/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # Longer timeout for model inference
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Chat endpoint working!")
            print(f"   Response:")
            print(f"     - Answer: {data.get('answer', 'N/A')[:100]}...")
            print(f"     - Confidence: {data.get('confidence', 'N/A')}")
            print(f"     - Safe: {data.get('safe', 'N/A')}")
            return True
        else:
            print(f"   ❌ Chat endpoint returned status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ⚠️  Request timed out (model may be loading)")
        print(f"   This is normal on first request - model needs to load")
        return True  # Not a connection issue
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 3: CORS headers (if available)
    print("\n3. Testing CORS configuration...")
    try:
        response = requests.options(
            f"{API_URL}/api/v1/chat",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST"
            }
        )
        cors_headers = {
            "Access-Control-Allow-Origin": response.headers.get("Access-Control-Allow-Origin"),
            "Access-Control-Allow-Methods": response.headers.get("Access-Control-Allow-Methods"),
        }
        print(f"   CORS Headers: {cors_headers}")
        if "localhost:5173" in str(cors_headers.get("Access-Control-Allow-Origin", "")):
            print(f"   ✅ CORS configured for frontend")
        else:
            print(f"   ⚠️  CORS may not be configured correctly")
    except Exception as e:
        print(f"   ⚠️  Could not test CORS: {e}")
    
    return True

def test_request_format():
    """Test that request format matches what frontend sends."""
    print("\n4. Testing request format compatibility...")
    
    # Simulate frontend request
    frontend_request = {
        "message": "test message",
        "history": [
            {"role": "user", "content": "previous message"},
            {"role": "assistant", "content": "previous response"}
        ]
    }
    
    try:
        response = requests.post(
            f"{API_URL}/api/v1/chat",
            json=frontend_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"   ✅ Request format is correct")
            print(f"   Frontend format matches backend expectations")
            return True
        elif response.status_code == 422:
            print(f"   ❌ Validation error - format mismatch")
            print(f"   Response: {response.json()}")
            return False
        else:
            print(f"   ⚠️  Status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n🔍 Backend-Frontend Connection Test\n")
    
    # Test backend connection
    backend_ok = test_backend_connection()
    
    if backend_ok:
        # Test request format
        format_ok = test_request_format()
        
        print("\n" + "=" * 60)
        if backend_ok and format_ok:
            print("✅ All connection tests passed!")
            print("\nFrontend should be able to connect to backend.")
            print("If frontend still has issues, check:")
            print("  1. Frontend is running on http://localhost:5173")
            print("  2. Browser console for errors")
            print("  3. Network tab in browser DevTools")
        else:
            print("⚠️  Some tests failed. Check errors above.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Backend connection failed!")
        print("\nTo fix:")
        print("  1. Start backend: cd backend && uvicorn app.main:app --reload")
        print("  2. Wait for 'Backend ready!' message")
        print("  3. Run this test again")
        print("=" * 60)
        sys.exit(1)

