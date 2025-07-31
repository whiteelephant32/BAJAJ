"""
Fixed Gemini API debug script with correct model names
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def debug_gemini_api_fixed():
    print("🔍 Debugging Gemini API Connection (FIXED)")
    print("=" * 50)
    
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"API Key: {api_key[:15]}...{api_key[-10:]}")
    
    try:
        import google.generativeai as genai
        print("✅ Library imported successfully")
        
        genai.configure(api_key=api_key)
        print("✅ API configured")
        
        # List all available models for generateContent
        print("\n📋 Available models for generateContent:")
        models = list(genai.list_models())
        generate_models = []
        
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                generate_models.append(model.name)
                print(f"  ✅ {model.name}")
        
        if not generate_models:
            print("❌ No models support generateContent")
            return False
        
        # Test with the first available model
        test_model_name = generate_models[0]
        print(f"\n🤖 Testing with model: {test_model_name}")
        
        model = genai.GenerativeModel(test_model_name)
        response = model.generate_content("Say hello")
        
        if response and hasattr(response, 'text') and response.text:
            print(f"✅ Generation successful!")
            print(f"Response: {response.text}")
            print(f"\n🎉 RECOMMENDED MODEL: {test_model_name}")
            return True
        else:
            print("❌ Empty response")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    debug_gemini_api_fixed()