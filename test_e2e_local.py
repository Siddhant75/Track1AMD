import os
import sys
import json
import subprocess
from pathlib import Path

def setup_and_run():
    # Setup paths
    base_dir = Path(__file__).parent
    input_dir = base_dir / "mock_input"
    output_dir = base_dir / "mock_output"
    
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    tasks_file = input_dir / "tasks.json"
    results_file = output_dir / "results.json"
    
    # 1. Mock some tasks. We'll include exact matches, regex matches, and some unseen tasks.
    mock_tasks = [
        {"task_id": "001", "prompt": "What are the three primary colors in the RGB color model?", "category": "factual"},
        {"task_id": "002", "prompt": "Solve for x: 5x + 3 = 18", "category": "math"},
        {"task_id": "003", "prompt": "A warehouse starts with 2,400 units in Q1. If sales reduce it by 37%, and they restock 800, then sell 640 in Q3, how many units are left?", "category": "math"},
        {"task_id": "004", "prompt": "Translate 'Hello World' to French.", "category": "factual"} # This one will fall back to local models since it's unseen
    ]
    
    with open(tasks_file, "w") as f:
        json.dump(mock_tasks, f)
        
    print(f"[TEST] Wrote {len(mock_tasks)} mock tasks to {tasks_file}")
    
    # 2. Inject environment variables
    env = os.environ.copy()
    env["INPUT_PATH"] = str(tasks_file)
    env["OUTPUT_PATH"] = str(results_file)
    env["LOCAL_MODEL_PATH"] = str(base_dir / "models" / "gemma-2-2b-it-Q4_K_M.gguf")
    
    # User must provide their own key or we assume it's already in the host env
    if "FIREWORKS_API_KEY" not in env or not env["FIREWORKS_API_KEY"]:
        print("[WARNING] FIREWORKS_API_KEY not found in environment. Remote escalation will fail.")
        # We don't exit so we can at least test the local & deterministic flow
    
    env["FIREWORKS_BASE_URL"] = "https://api.fireworks.ai"
    env["ALLOWED_MODELS"] = "minimax-m3,kimi-k2p7-code"
    
    # 3. Run the main agent
    print("\n" + "="*50)
    print("RUNNING AGENT PIPELINE")
    print("="*50)
    
    main_script = base_dir / "main.py"
    process = subprocess.Popen(
        [sys.executable, str(main_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Stream output
    for line in process.stdout:
        print(line.rstrip())
        
    process.wait()
    
    # 4. Check results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    
    if results_file.exists():
        with open(results_file, "r") as f:
            try:
                results = json.load(f)
                print(json.dumps(results, indent=2))
            except json.JSONDecodeError:
                print("[ERROR] Could not parse results.json")
    else:
        print("[ERROR] results.json was not generated.")

if __name__ == "__main__":
    setup_and_run()
