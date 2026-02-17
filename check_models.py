import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

# Load API Key
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("❌ Error: GEMINI_API_KEY not found in .env file")
    exit()

genai.configure(api_key=API_KEY)

print(f"🔍 Scanning available models for your API key...")
print(f"{'-'*60}")
print(f"{'MODEL NAME':<30} | {'STATUS':<25}")
print(f"{'-'*60}")

try:
    # 1. Get the official list from Google
    all_models = list(genai.list_models())
    
    valid_models = []
    
    # Filter only models that generate text
    for m in all_models:
        if 'generateContent' in m.supported_generation_methods:
            valid_models.append(m.name)

    # 2. Test each model
    working_models = []
    
    for model_name in valid_models:
        # Strip the "models/" prefix if present for cleaner output
        short_name = model_name.replace("models/", "")
        
        try:
            model = genai.GenerativeModel(short_name)
            # Send a tiny prompt to test response
            response = model.generate_content("Hi", request_options={"timeout": 10})
            
            print(f"{short_name:<30} | ✅ AVAILABLE")
            working_models.append(short_name)
            
        except Exception as e:
            error = str(e).lower()
            if "429" in error or "quota" in error:
                print(f"{short_name:<30} | ⚠️ RATE LIMITED (Busy)")
            elif "not found" in error or "404" in error:
                print(f"{short_name:<30} | ❌ NOT FOUND (Deprecated)")
            else:
                # Shorten long error messages
                short_err = str(e)[:20] + "..."
                print(f"{short_name:<30} | ❌ ERROR: {short_err}")
        
        # Small sleep to prevent self-inflicted rate limits during the test
        time.sleep(1)

    print(f"{'-'*60}")
    print(f"\n✨ RECOMMENDATION:")
    if working_models:
        print(f"Update your script to use this list: {working_models}")
    else:
        print("❌ No models are working right now. You might be completely out of quota.")

except Exception as e:
    print(f"\n❌ Critical Error: {e}")