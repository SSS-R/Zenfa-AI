import asyncio
import os
from zenfa_ai.evaluator.llm_client import create_llm_client

# Test script to verify that search grounding is functioning and tools are attached
async def main():
    print("Testing LLM generation with grounding...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Skipping LLM test: No GEMINI_API_KEY provided")
        return
        
    client = create_llm_client(gemini_api_key=api_key)
    if not client:
        print("Failed to initialize LLM client.")
        return
        
    # We provide a prompt that requires searching for latest component news
    sys_prompt = "You are a tech AI evaluating a component. You must use the Google Search tool to find recent information."
    user_prompt = "Find a recent (last 30 days) hardware news or known issue regarding the Intel Core i9-14900K and summarize it briefly in json under the key 'summary'."
    
    try:
        # We invoke the internal _call_llm directly to bypass the rigid structured schema validation for evaluation responses
        # Because we're just doing a quick test of the mechanics.
        raw = await client._call_llm(sys_prompt, user_prompt)
        print("Raw response from LLM:")
        print(raw)
        
        if "google_search" in str(client._get_client()._tools):
            print("SUCCESS: Google Search tool is correctly mounted on the Gemini client model instance.")
        else:
            print("WARNING: google_search does not appear to be present in tools.")
    except Exception as e:
        print(f"Error calling LLM: {e}")

if __name__ == "__main__":
    asyncio.run(main())
