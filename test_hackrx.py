#!/usr/bin/env python3
"""
Test script for HackRx 6.0 LLM-Powered Query-Retrieval System
Tests the system with the provided sample data
"""

import asyncio
import requests
import json
import time
from typing import List, Dict
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "5198af4a74b3f28046d225858ffe5010789c03ac0a414b41e5fa08533884a424"

# Sample test data from the problem statement
SAMPLE_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf"
SAMPLE_QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Medic?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
]

class HackRxTester:
    """Test client for the HackRx system"""
    
    def __init__(self, base_url: str = API_BASE_URL, token: str = BEARER_TOKEN):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
    
    def test_health_endpoint(self) -> bool:
        """Test if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data.get('status')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_hackrx_endpoint(self, document_url: str, questions: List[str]) -> Dict:
        """Test the main HackRx endpoint"""
        payload = {
            "documents": document_url,
            "questions": questions
        }
        
        print(f"\n🚀 Testing HackRx endpoint with {len(questions)} questions...")
        print(f"📄 Document: {document_url}")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                headers=self.headers,
                json=payload,
                timeout=120  # 2 minutes timeout
            )
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Request successful (took {processing_time:.0f}ms)")
                return {
                    "success": True,
                    "data": data,
                    "processing_time": processing_time
                }
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Error: {response.text}")
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def print_results(self, result: Dict, questions: List[str]):
        """Pretty print the results"""
        if not result.get("success"):
            print(f"\n❌ Test failed: {result.get('error')}")
            return
        
        data = result["data"]
        answers = data.get("answers", [])
        processing_time = data.get("processing_time", 0)
        
        print(f"\n📊 Results Summary:")
        print(f"   • Total processing time: {processing_time}ms")
        print(f"   • Questions processed: {len(questions)}")
        print(f"   • Answers generated: {len(answers)}")
        print(f"   • Average time per question: {processing_time/len(questions):.0f}ms")
        
        print(f"\n📝 Question-Answer Pairs:")
        print("=" * 80)
        
        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            print(f"\n{i}. Q: {question}")
            print(f"   A: {answer}")
            print("-" * 80)
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🧪 Starting HackRx System Comprehensive Test")
        print("=" * 50)
        
        # Test 1: Health check
        if not self.test_health_endpoint():
            print("❌ System is not running. Please start the server first.")
            return False
        
        # Test 2: Small batch (first 3 questions)
        print(f"\n🔬 Test 2: Small batch (3 questions)")
        small_result = self.test_hackrx_endpoint(
            SAMPLE_DOCUMENT_URL, 
            SAMPLE_QUESTIONS[:3]
        )
        self.print_results(small_result, SAMPLE_QUESTIONS[:3])
        
        if not small_result.get("success"):
            print("❌ Small batch test failed. Stopping here.")
            return False
        
        # Test 3: Full batch (all questions)
        print(f"\n🔬 Test 3: Full batch ({len(SAMPLE_QUESTIONS)} questions)")
        full_result = self.test_hackrx_endpoint(
            SAMPLE_DOCUMENT_URL, 
            SAMPLE_QUESTIONS
        )
        self.print_results(full_result, SAMPLE_QUESTIONS)
        
        # Test 4: Performance analysis
        self.analyze_performance(small_result, full_result)
        
        return full_result.get("success", False)
    
    def analyze_performance(self, small_result: Dict, full_result: Dict):
        """Analyze performance metrics"""
        print(f"\n📈 Performance Analysis:")
        print("=" * 30)
        
        if small_result.get("success"):
            small_time = small_result["processing_time"]
            small_count = 3
            print(f"   • Small batch (3 Q): {small_time:.0f}ms ({small_time/small_count:.0f}ms/Q)")
        
        if full_result.get("success"):
            full_time = full_result["processing_time"]
            full_count = len(SAMPLE_QUESTIONS)
            print(f"   • Full batch ({full_count} Q): {full_time:.0f}ms ({full_time/full_count:.0f}ms/Q)")
            
            # Calculate efficiency metrics
            if small_result.get("success"):
                expected_time = (small_time / small_count) * full_count
                efficiency = expected_time / full_time
                print(f"   • Batch efficiency: {efficiency:.2f}x")
                print(f"   • Time saved: {expected_time - full_time:.0f}ms")

def main():
    """Main test function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "health":
            tester = HackRxTester()
            tester.test_health_endpoint()
            return
        elif sys.argv[1] == "quick":
            tester = HackRxTester()
            result = tester.test_hackrx_endpoint(SAMPLE_DOCUMENT_URL, SAMPLE_QUESTIONS[:2])
            tester.print_results(result, SAMPLE_QUESTIONS[:2])
            return
    
    # Run comprehensive test
    tester = HackRxTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print(f"\n🎉 All tests passed! System is ready for submission.")
    else:
        print(f"\n❌ Some tests failed. Please check the logs and fix issues.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
