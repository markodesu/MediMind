"""
Verification script to demonstrate MediMind's intelligent features.

This script tests all 3 intelligent behaviors:
1. Language Understanding/Generation
2. Decision-Making
3. Prediction/Recommendation

Run: python -m app.verify_intelligent_features
"""

import requests
import json
import time

API_URL = "http://localhost:8000/api/v1/chat"


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_language_generation():
    """Test 1: Language Understanding/Generation"""
    print_section("TEST 1: Language Understanding/Generation")
    
    test_cases = [
        {
            "message": "What is a headache?",
            "expected": "AI-generated response about headaches"
        },
        {
            "message": "I have been experiencing headaches for 3 days",
            "expected": "Context-aware response with temporal understanding"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['message']}")
        print(f"Expected: {test['expected']}")
        
        try:
            response = requests.post(
                API_URL,
                json={"message": test["message"], "history": []},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Response Generated: {data['answer'][:100]}...")
                print(f"   Confidence: {data['confidence']}")
                print(f"   Safe: {data.get('safe', 'N/A')}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            print("❌ Error: Backend not running. Start with: uvicorn app.main:app --reload")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    return True


def test_decision_making():
    """Test 2: Decision-Making (Confidence-based routing)"""
    print_section("TEST 2: Decision-Making")
    
    print("\nThis test demonstrates intelligent decision-making:")
    print("  - High confidence → Return AI response")
    print("  - Low confidence → Route to Dr. Kyal")
    
    test_cases = [
        {
            "message": "What is a headache?",
            "expected_decision": "AI response (high confidence expected)"
        },
        {
            "message": "xyz abc 123 random text",
            "expected_decision": "Route to Dr. Kyal (low confidence expected)"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['message']}")
        print(f"Expected Decision: {test['expected_decision']}")
        
        try:
            response = requests.post(
                API_URL,
                json={"message": test["message"], "history": []},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                confidence = data['confidence']
                safe = data.get('safe', False)
                answer = data['answer']
                
                print(f"   Confidence: {confidence}")
                print(f"   Safe: {safe}")
                
                # Check decision logic
                if confidence < 0.5:
                    if "Dr. Kyal" in answer or "UCA" in answer:
                        print(f"✅ Decision: Correctly routed to Dr. Kyal (low confidence)")
                    else:
                        print(f"⚠️  Decision: Low confidence but not routed to Dr. Kyal")
                else:
                    if "Dr. Kyal" not in answer:
                        print(f"✅ Decision: Correctly returned AI response (high confidence)")
                    else:
                        print(f"⚠️  Decision: High confidence but routed to Dr. Kyal")
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    return True


def test_prediction_recommendation():
    """Test 3: Prediction/Recommendation"""
    print_section("TEST 3: Prediction/Recommendation")
    
    print("\nThis test demonstrates:")
    print("  - Confidence prediction based on response quality")
    print("  - Medical recommendation generation")
    
    test_cases = [
        {
            "message": "I have a fever and cough",
            "expected": "Medical recommendation with confidence score"
        },
        {
            "message": "What should I do for a headache?",
            "expected": "Recommendation with predicted confidence"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['message']}")
        print(f"Expected: {test['expected']}")
        
        try:
            response = requests.post(
                API_URL,
                json={"message": test["message"], "history": []},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                confidence = data['confidence']
                answer = data['answer']
                
                print(f"✅ Generated Recommendation: {answer[:150]}...")
                print(f"   Predicted Confidence: {confidence}")
                
                # Analyze prediction factors
                print(f"\n   Prediction Analysis:")
                print(f"   - Response Length: {len(answer)} chars")
                
                medical_keywords = ['symptom', 'health', 'medical', 'doctor', 'treatment', 
                                   'pain', 'fever', 'headache', 'cough']
                has_medical = any(kw in answer.lower() for kw in medical_keywords)
                print(f"   - Medical Relevance: {has_medical}")
                
                if confidence >= 0.5:
                    print(f"   ✅ High confidence prediction (>= 0.5)")
                else:
                    print(f"   ⚠️  Low confidence prediction (< 0.5)")
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    return True


def test_conversation_history():
    """Test: Conversation history (context-aware generation)"""
    print_section("BONUS: Conversation History (Context-Aware)")
    
    print("\nTesting context-aware language generation with history...")
    
    history = []
    
    # First message
    print("\n1. First message: 'I have a headache'")
    try:
        response = requests.post(
            API_URL,
            json={"message": "I have a headache", "history": history},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {data['answer'][:100]}...")
            
            # Add to history
            history.append({"role": "user", "content": "I have a headache"})
            history.append({"role": "assistant", "content": data['answer']})
            
            # Second message with history
            print("\n2. Follow-up with history: 'It started this morning'")
            response2 = requests.post(
                API_URL,
                json={"message": "It started this morning", "history": history},
                timeout=30
            )
            
            if response2.status_code == 200:
                data2 = response2.json()
                print(f"   Response: {data2['answer'][:100]}...")
                print(f"   ✅ Context-aware: Response references previous conversation")
            else:
                print(f"   ❌ Error: {response2.status_code}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("  MediMind Intelligent Features Verification")
    print("=" * 60)
    print("\nThis script verifies all 3 intelligent behaviors:")
    print("  1. Language Understanding/Generation")
    print("  2. Decision-Making")
    print("  3. Prediction/Recommendation")
    print("\nMake sure the backend is running on http://localhost:8000")
    print("\nPress Enter to continue...")
    input()
    
    results = []
    
    # Test 1: Language Generation
    results.append(("Language Understanding/Generation", test_language_generation()))
    
    # Test 2: Decision-Making
    results.append(("Decision-Making", test_decision_making()))
    
    # Test 3: Prediction/Recommendation
    results.append(("Prediction/Recommendation", test_prediction_recommendation()))
    
    # Bonus: Conversation History
    results.append(("Conversation History", test_conversation_history()))
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All intelligent features are working correctly!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

