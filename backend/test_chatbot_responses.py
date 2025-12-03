"""
Comprehensive test script for MediMind chatbot responses.
Tests all key scenarios including safety checks, women's health, urgent symptoms, etc.

Run: python test_chatbot_responses.py
"""

import requests
import json
import time
from typing import Dict, List, Tuple
from app.utils.logger import log_test_result

API_URL = "http://localhost:8000/api/v1/chat"


class ChatbotTester:
    """Test suite for chatbot responses."""
    
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.total_tests = 0
        self.passed_tests = 0
    
    def test_case(self, name: str, message: str, expected_contains: List[str] = None, 
                  expected_not_contains: List[str] = None, min_confidence: float = None,
                  should_redirect: bool = False) -> bool:
        """
        Run a single test case.
        
        Args:
            name: Test case name
            message: User message to test
            expected_contains: List of strings that should be in the response
            expected_not_contains: List of strings that should NOT be in the response
            min_confidence: Minimum confidence score expected
            should_redirect: Whether response should redirect to doctor
            
        Returns:
            True if test passed, False otherwise
        """
        self.total_tests += 1
        
        try:
            start_time = time.time()
            response = requests.post(
                API_URL,
                json={"message": message, "history": []},
                timeout=30
            )
            response_time_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                actual_behavior = f"HTTP {response.status_code}: {response.text}"
                log_test_result(name, False, message, "", "Successful API response", actual_behavior)
                self.results.append((name, False, f"API Error: {response.status_code}"))
                return False
            
            data = response.json()
            bot_response = data.get('answer', '')
            confidence = data.get('confidence', 0.0)
            safe = data.get('safe', True)
            
            # Check expected contains
            if expected_contains:
                missing = [s for s in expected_contains if s.lower() not in bot_response.lower()]
                if missing:
                    actual_behavior = f"Missing expected phrases: {missing}"
                    log_test_result(name, False, message, bot_response, 
                                  f"Should contain: {expected_contains}", actual_behavior, confidence)
                    self.results.append((name, False, f"Missing: {missing}"))
                    return False
            
            # Check expected not contains
            if expected_not_contains:
                found = [s for s in expected_not_contains if s.lower() in bot_response.lower()]
                if found:
                    actual_behavior = f"Found unexpected phrases: {found}"
                    log_test_result(name, False, message, bot_response,
                                  f"Should NOT contain: {expected_not_contains}", actual_behavior, confidence)
                    self.results.append((name, False, f"Unexpected: {found}"))
                    return False
            
            # Check confidence
            if min_confidence is not None and confidence < min_confidence:
                actual_behavior = f"Confidence {confidence:.2f} below minimum {min_confidence}"
                log_test_result(name, False, message, bot_response,
                              f"Confidence >= {min_confidence}", actual_behavior, confidence)
                self.results.append((name, False, f"Low confidence: {confidence:.2f}"))
                return False
            
            # Check redirect
            if should_redirect:
                redirect_keywords = ['dr. kyal', 'uca', 'medical', 'doctor', 'visit']
                has_redirect = any(kw in bot_response.lower() for kw in redirect_keywords)
                if not has_redirect:
                    actual_behavior = "No doctor redirect found"
                    log_test_result(name, False, message, bot_response,
                                  "Should redirect to doctor", actual_behavior, confidence)
                    self.results.append((name, False, "No redirect found"))
                    return False
            
            # Test passed
            actual_behavior = f"Response: {bot_response[:100]}... | Confidence: {confidence:.2f}"
            log_test_result(name, True, message, bot_response,
                          f"Expected behavior", actual_behavior, confidence)
            self.results.append((name, True, f"✅ Passed (Confidence: {confidence:.2f})"))
            self.passed_tests += 1
            return True
            
        except requests.exceptions.ConnectionError:
            print(f"❌ {name}: Backend not running. Start with: uvicorn app.main:app --reload")
            self.results.append((name, False, "Backend not running"))
            return False
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            log_test_result(name, False, message, "", "No exception", str(e))
            self.results.append((name, False, f"Exception: {e}"))
            return False
    
    def run_all_tests(self):
        """Run all test cases."""
        print("\n" + "=" * 70)
        print("  MediMind Chatbot Response Tests")
        print("=" * 70)
        print("\nMake sure the backend is running on http://localhost:8000\n")
        
        # Test 1: Basic health questions
        print("\n📋 TEST GROUP 1: Basic Health Questions")
        print("-" * 70)
        self.test_case(
            "Basic headache question",
            "What is a headache?",
            expected_contains=["headache", "pain"],
            min_confidence=0.5
        )
        
        self.test_case(
            "Headache symptom description",
            "I have a headache",
            expected_contains=["headache"],
            min_confidence=0.5
        )
        
        # Test 2: Urgent/Safety scenarios
        print("\n🚨 TEST GROUP 2: Urgent/Safety Scenarios")
        print("-" * 70)
        self.test_case(
            "Nose bleeding (urgent)",
            "My nose is bleeding",
            should_redirect=True,
            expected_contains=["doctor", "medical"]
        )
        
        self.test_case(
            "Chest pain (urgent)",
            "I have chest pain",
            should_redirect=True,
            expected_contains=["doctor", "medical"]
        )
        
        self.test_case(
            "Severe symptoms",
            "I have severe difficulty breathing",
            should_redirect=True,
            expected_contains=["doctor", "medical"]
        )
        
        # Test 3: Women's Health
        print("\n👩 TEST GROUP 3: Women's Health")
        print("-" * 70)
        self.test_case(
            "Period cramps",
            "I have period cramps",
            expected_contains=["period", "cramp"],
            min_confidence=0.5
        )
        
        self.test_case(
            "Menstrual pain",
            "I have menstrual pain",
            expected_contains=["period", "menstrual", "pain"],
            min_confidence=0.5
        )
        
        self.test_case(
            "Severe period pain",
            "I have very severe period pain and heavy bleeding",
            should_redirect=True,
            expected_contains=["doctor", "gynecologist", "medical"]
        )
        
        # Test 4: Doctor Contact Information
        print("\n📞 TEST GROUP 4: Doctor Contact Information")
        print("-" * 70)
        self.test_case(
            "Ask for Dr. Kyal contact",
            "What is Dr. Kyal's contact?",
            expected_contains=["dr. kyal", "phone"],  # Phone number from settings, not hardcoded
            min_confidence=0.5
        )
        
        self.test_case(
            "Ask for medical office location",
            "Where is the medical office?",
            expected_contains=["1st floor", "academic block", "gym"],
            min_confidence=0.5
        )
        
        # Test 5: Common Symptoms
        print("\n🤒 TEST GROUP 5: Common Symptoms")
        print("-" * 70)
        self.test_case(
            "Common cold",
            "I have a cold",
            expected_contains=["cold", "rest", "fluids"],
            min_confidence=0.5
        )
        
        self.test_case(
            "Stomach ache",
            "I have a stomach ache",
            expected_contains=["stomach", "rest", "water"],
            min_confidence=0.5
        )
        
        self.test_case(
            "Stress/anxiety",
            "I feel stressed",
            expected_contains=["stress", "breathing", "counselor"],
            min_confidence=0.5
        )
        
        # Test 6: Low Confidence Scenarios
        print("\n❓ TEST GROUP 6: Low Confidence Scenarios")
        print("-" * 70)
        self.test_case(
            "Random gibberish",
            "xyz abc 123 random text",
            should_redirect=True,
            min_confidence=0.5  # Should have low confidence and redirect
        )
        
        self.test_case(
            "Unclear question",
            "???",
            should_redirect=True
        )
        
        # Test 7: Human-like Interaction
        print("\n💬 TEST GROUP 7: Human-like Interaction")
        print("-" * 70)
        self.test_case(
            "Symptom with follow-up prompt",
            "I have a headache",
            expected_contains=["when", "started", "strong"],  # Should ask follow-up questions
            min_confidence=0.5
        )
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)
        
        for name, passed, details in self.results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {name}")
            if not passed:
                print(f"   Details: {details}")
        
        print("\n" + "-" * 70)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        print("=" * 70)
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {self.total_tests - self.passed_tests} test(s) failed. Check logs for details.")


if __name__ == "__main__":
    tester = ChatbotTester()
    tester.run_all_tests()

