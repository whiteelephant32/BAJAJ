"""
Test script for Gemini API integration
Tests API key, basic functionality, and integration with the main system
"""

import os
import sys
import asyncio
import json
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def test_environment_setup():
    """Test 1: Check environment variables and setup"""
    print("🧪 Test 1: Environment Setup")
    print("-" * 50)
    
    # Check if .env file exists
    env_file = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_file):
        print("✅ .env file found")
    else:
        print("❌ .env file not found - create one with your API keys")
        return False
    
    # Check Gemini API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print(f"✅ GEMINI_API_KEY found: {gemini_key[:10]}...{gemini_key[-4:]}")
    else:
        print("❌ GEMINI_API_KEY not found in environment")
        return False
    
    # Check OpenAI API key (for comparison)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OPENAI_API_KEY found: {openai_key[:10]}...{openai_key[-4:]}")
    else:
        print("⚠️ OPENAI_API_KEY not found in environment")
    
    # Check LLM provider setting
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    print(f"🔧 LLM_PROVIDER set to: {llm_provider}")
    
    print()
    return True

def test_gemini_library_import():
    """Test 2: Check if Gemini library is installed and can be imported"""
    print("🧪 Test 2: Gemini Library Import")
    print("-" * 50)
    
    try:
        import google.generativeai as genai
        print("✅ google.generativeai library imported successfully")
        print(f"📦 Library version: {genai.__version__ if hasattr(genai, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import google.generativeai: {e}")
        print("💡 Install with: pip install google-generativeai==0.3.1")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing library: {e}")
        return False
    
    print()

def test_gemini_basic_connection():
    """Test 3: Basic Gemini API connection and simple generation"""
    print("🧪 Test 3: Basic Gemini API Connection")
    print("-" * 50)
    
    try:
        import google.generativeai as genai
        
        # Configure API key
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        print("✅ Gemini API configured successfully")
        
        # Initialize model
        model = genai.GenerativeModel('gemini-pro')
        print("✅ Gemini Pro model initialized")
        
        # Test simple generation
        print("🔄 Testing simple text generation...")
        response = model.generate_content("Hello! Can you tell me what you are?")
        
        if response.text:
            print("✅ Gemini API connection successful!")
            print(f"📝 Response: {response.text[:100]}...")
            return True
        else:
            print("❌ Gemini API returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        if "API_KEY" in str(e).upper():
            print("💡 Check your GEMINI_API_KEY in the .env file")
        elif "QUOTA" in str(e).upper():
            print("💡 API quota exceeded - check your Gemini API limits")
        elif "PERMISSION" in str(e).upper():
            print("💡 Permission denied - verify your API key has correct permissions")
        return False
    
    print()

def test_gemini_advanced_features():
    """Test 4: Advanced Gemini features with generation config"""
    print("🧪 Test 4: Advanced Gemini Features")
    print("-" * 50)
    
    try:
        import google.generativeai as genai
        
        # Configure API
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-pro')
        
        # Test with generation config (similar to your main code)
        prompt = """You are an expert at analyzing documents. Based on the context below, answer the question.

Context: This is a sample insurance policy document. The waiting period for major surgeries is 24 months. The coverage amount is $100,000 per year.

Question: What is the waiting period for major surgeries?

Answer:"""
        
        print("🔄 Testing with generation config...")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=300,
            )
        )
        
        if response.text:
            print("✅ Advanced generation successful!")
            print(f"📝 Response: {response.text}")
            return True
        else:
            print("❌ Advanced generation failed - empty response")
            return False
            
    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        return False
    
    print()

async def test_gemini_service_integration():
    """Test 5: Integration with your GeminiService class"""
    print("🧪 Test 5: GeminiService Integration")
    print("-" * 50)
    
    try:
        # Add the current directory to Python path to import from main.py
        sys.path.insert(0, os.getcwd())
        
        from main import GeminiService, Config
        
        print("✅ Successfully imported GeminiService from main.py")
        
        # Test GeminiService initialization
        gemini_service = GeminiService()
        print("✅ GeminiService initialized successfully")
        print(f"🔧 Provider: {gemini_service.provider}")
        
        # Test generate_answer method
        mock_context_chunks = [
            {
                'content': 'The insurance policy covers medical expenses up to $50,000 annually.',
                'score': 0.95,
                'metadata': {'chunk_id': 'test123', 'document_id': 'test_doc'}
            },
            {
                'content': 'Waiting period for pre-existing conditions is 12 months.',
                'score': 0.87,
                'metadata': {'chunk_id': 'test456', 'document_id': 'test_doc'}
            }
        ]
        
        print("🔄 Testing generate_answer method...")
        result = await gemini_service.generate_answer(
            "What is the annual coverage limit?", 
            mock_context_chunks
        )
        
        if result and result.get('answer'):
            print("✅ GeminiService generate_answer successful!")
            print(f"📝 Answer: {result['answer']}")
            print(f"🔍 Provider: {result.get('provider', 'unknown')}")
            print(f"📊 Confidence: {result.get('confidence', 0):.2f}")
            return True
        else:
            print("❌ GeminiService generate_answer failed")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import from main.py: {e}")
        return False
    except Exception as e:
        print(f"❌ GeminiService integration test failed: {e}")
        return False
    
    print()

def test_environment_variables():
    """Test 6: Comprehensive environment variable check"""
    print("🧪 Test 6: Environment Variables Check")
    print("-" * 50)
    
    required_vars = {
        'GEMINI_API_KEY': 'Required for Gemini API',
        'OPENAI_API_KEY': 'Required for OpenAI API (optional)',
        'LLM_PROVIDER': 'Determines which LLM to use'
    }
    
    all_good = True
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            if 'API_KEY' in var:
                print(f"✅ {var}: {value[:10]}...{value[-4:]} - {description}")
            else:
                print(f"✅ {var}: {value} - {description}")
        else:
            if var == 'OPENAI_API_KEY':
                print(f"⚠️ {var}: Not set - {description}")
            else:
                print(f"❌ {var}: Not set - {description}")
                all_good = False
    
    print()
    return all_good

def create_sample_env_file():
    """Create a sample .env file if it doesn't exist"""
    print("🧪 Creating sample .env file")
    print("-" * 50)
    
    env_file = os.path.join(os.getcwd(), '.env')
    
    if os.path.exists(env_file):
        print("✅ .env file already exists")
        return True
    
    sample_content = """# HackRx 6.0 LLM System Configuration

# OpenAI Configuration
OPENAI_API_KEY=sk-your_openai_api_key_here

# Gemini Configuration  
GEMINI_API_KEY=AIzaSyYour_Gemini_API_Key_Here

# LLM Provider Selection (openai or gemini)
LLM_PROVIDER=openai

# Database Configuration
DATABASE_URL=sqlite:///hackrx.db

# Optional: Pinecone Configuration
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENV=us-east-1-aws
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(sample_content)
        print("✅ Sample .env file created successfully!")
        print("💡 Please edit the .env file and add your actual API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

async def run_all_tests():
    """Run all tests in sequence"""
    print("🚀 Starting Gemini API Integration Tests")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print("=" * 60)
    print()
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Gemini Library Import", test_gemini_library_import),
        ("Basic API Connection", test_gemini_basic_connection),
        ("Advanced Features", test_gemini_advanced_features),
        ("Service Integration", test_gemini_service_integration),
        ("Environment Variables", test_environment_variables),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("-" * 60)
    print(f"📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Gemini integration is ready!")
    elif passed >= total * 0.5:
        print("⚠️ Some tests failed. Check the issues above.")
    else:
        print("❌ Multiple tests failed. Check your configuration.")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 60)
    
    if not results.get("Environment Setup", False):
        print("• Create or update your .env file with API keys")
    
    if not results.get("Gemini Library Import", False):
        print("• Install google-generativeai: pip install google-generativeai==0.3.1")
    
    if not results.get("Basic API Connection", False):
        print("• Verify your GEMINI_API_KEY is correct and has permissions")
        print("• Check your internet connection")
        print("• Verify your Gemini API quota")
    
    if results.get("Basic API Connection", False) and not results.get("Service Integration", False):
        print("• Check your main.py file imports and class definitions")
    
    print(f"\n🔗 Next Steps:")
    print("1. Fix any failing tests")
    print("2. Run your main application: python main.py")
    print("3. Test the /switch-model endpoint")
    print("4. Try the /hackrx/run endpoint with Gemini")

if __name__ == "__main__":
    # Check if .env file exists, create sample if not
    if not os.path.exists('.env'):
        print("No .env file found. Creating sample...")
        create_sample_env_file()
        print("\n⚠️ Please edit the .env file with your API keys before running tests.\n")
    
    # Run all tests
    asyncio.run(run_all_tests())