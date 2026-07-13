import os
import sys
import json
import time
import subprocess

def create_dummy_tasks(filename="input/tasks.json", count=50):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Mix of tasks to trigger Gemma and DeepSeek locally, plus some hard ones for remote
    tasks = []
    
    for i in range(1, count + 1):
        if i % 5 == 0:
            prompt = f"Solve {i}x + 7 = 35 step by step. Final answer in 10 words."
        elif i % 5 == 1:
            prompt = f"Analyze the sentiment of this review: 'The product {i} is terrible and broke immediately.' Exactly one word."
        elif i % 5 == 2:
            prompt = f"Name {i} historical figures who were scientists. Exactly {min(i, 5)} bullet points."
        elif i % 5 == 3:
            prompt = f"Write a python function to calculate the factorial of {i}. Under 20 words outside code."
        else:
            prompt = f"Summarize the benefits of walking {i} miles a day in exactly two sentences."
            
        tasks.append({
            "task_id": f"test_{i:03d}",
            "prompt": prompt
        })
        
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)
    print(f"[TEST HARNESS] Generated {count} tasks in {filename}")

def main():
    print("=== AMD Track 1 Local End-to-End Test ===")
    
    # 1. Get API Key
    api_key = input("Enter your Fireworks API Key (starts with 'fw_...'): ").strip()
    if not api_key:
        print("API Key required for full test!")
        sys.exit(1)
        
    # 2. Setup environment
    env = os.environ.copy()
    env["FIREWORKS_API_KEY"] = api_key
    env["FIREWORKS_BASE_URL"] = "https://api.fireworks.ai/inference/v1"
    env["ALLOWED_MODELS"] = "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/gemma-4-31b-it"
    
    env["INPUT_PATH"] = "input/tasks.json"
    env["OUTPUT_PATH"] = "output/results.json"
    
    # Point models to the downloaded folder
    env["LOCAL_MODEL_PATH"] = "models/gemma-2-2b-it-Q4_K_M.gguf"
    env["LOCAL_REASONING_MODEL_PATH"] = "models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
    
    # 3. Create dummy tasks
    create_dummy_tasks(env["INPUT_PATH"], count=50)
    os.makedirs("output", exist_ok=True)
    
    # 4. Run main.py
    print(f"\n[TEST HARNESS] Starting main.py with 50 tasks...")
    print(f"[TEST HARNESS] Simulating grading sandbox... Timer started.\n")
    print("-" * 60)
    
    start_time = time.monotonic()
    
    try:
        # Run main.py, piping stdout and stderr to the console
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        process.wait()
    except KeyboardInterrupt:
        print("\n[TEST HARNESS] Interrupted by user.")
        process.terminate()
        sys.exit(1)
        
    end_time = time.monotonic()
    elapsed = end_time - start_time
    
    print("-" * 60)
    print(f"[TEST HARNESS] main.py exited with code {process.returncode}")
    print(f"[TEST HARNESS] Total Execution Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    if elapsed > 600:
        print("[TEST HARNESS] ❌ FAIL: Execution exceeded 10 minutes (600s).")
    else:
        print("[TEST HARNESS] ✅ PASS: Execution finished under 10 minutes.")
        
    if os.path.exists(env["OUTPUT_PATH"]):
        with open(env["OUTPUT_PATH"], "r") as f:
            try:
                results = json.load(f)
                print(f"[TEST HARNESS] Successfully wrote {len(results)} results to {env['OUTPUT_PATH']}.")
            except json.JSONDecodeError:
                print(f"[TEST HARNESS] ❌ FAIL: Results file is invalid JSON.")

if __name__ == "__main__":
    main()
