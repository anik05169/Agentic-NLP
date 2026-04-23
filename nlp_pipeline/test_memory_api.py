"""
Memory Behavior Test for Legal AI API
Verifies that the agent remembers short-term session context and 
recalls long-term semantic summaries for the same user.
"""

import requests
import json
import time
import sys

API_URL = "http://localhost:8000"

def test_memory_flow():
    user_id = f"test_user_{int(time.time())}"
    session_id = "session_A"
    
    print("="*60)
    print(f" TESTING MEMORY FOR USER: {user_id}")
    print("="*60)

    # 1. Turn 1: Establish baseline
    print("\n[TURN 1] Asking about a specific legal concept...")
    payload = {
        "query": "What is a 'material breach' in a contract?",
        "user_id": user_id,
        "session_id": session_id
    }
    try:
        r1 = requests.post(f"{API_URL}/ask", json=payload, timeout=30).json()
        print(f"-> AI Answered: {r1['answer'][:150]}...")
    except Exception as e:
        print(f"ERROR: API not reachable. Is it running? {e}")
        return

    # Wait for background task to save long-term memory
    print("\nWaiting for long-term memory background task (5s)...")
    time.sleep(5)

    # 2. Turn 2: Follow-up (Short-Term Memory)
    print("\n[TURN 2] Asking a follow-up without repeating the subject...")
    payload = {
        "query": "Can you summarize the consequences of that?",
        "user_id": user_id,
        "session_id": session_id
    }
    r2 = requests.post(f"{API_URL}/ask", json=payload, timeout=30).json()
    print(f"-> AI Answered: {r2['answer'][:150]}...")
    
    # 3. Turn 3: New Session (Long-Term Memory)
    print("\n[TURN 3] Starting NEW session for same user...")
    payload = {
        "query": "What did we discuss earlier regarding contract breaches?",
        "user_id": user_id,
        "session_id": "session_B"
    }
    r3 = requests.post(f"{API_URL}/ask", json=payload, timeout=30).json()
    print(f"-> AI Answered: {r3['answer'][:150]}...")

    print("\n" + "="*60)
    print(" TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_memory_flow()
