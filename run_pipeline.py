"""
FILE: run_pipeline.py
PURPOSE: Master orchestration script. Run this to execute any stage
         of the pipeline from one place, with guided prompts for novices.

HOW TO RUN:
  python run_pipeline.py

It will ask what you want to do and guide you step by step.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from glob import glob


# ============================================================
# MENU SYSTEM
# ============================================================

MENU = """
╔══════════════════════════════════════════════════════════════╗
║         EMOTIONAI-VR — MASTER CONTROL PANEL                 ║
╚══════════════════════════════════════════════════════════════╝

What would you like to do?

  [1] SIMULATE   — Run full simulation (no hardware needed)
                   Tests the hypothesis on synthetic data.
                   START HERE if you are a first-time user.

  [2] COLLECT    — Collect real session data (hardware required)
                   Arduino GSR + Polar H10 + Webcam must be connected.

  [3] PROCESS    — Process raw session data into features
                   Run after collecting one or more sessions.

  [4] TRAIN      — Train AI emotion prediction models
                   Run after processing at least 5 sessions.

  [5] SERVE      — Start inference server (bridges AI ↔ Unity VR)
                   Run this while Unity VR session is active.

  [6] ANALYZE    — Analyze results and get redesign recommendations
                   Run after training to evaluate model performance.

  [7] TEST       — Run automated test suite
                   Verifies everything is working correctly.

  [8] STATUS     — Show current project status (what data exists)

  [9] FULL RUN   — Run complete pipeline: simulate → process → train → analyze
                   Best for a quick end-to-end validation.

  [Q] QUIT

"""

SEPARATOR = "─" * 60


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(f"{SEPARATOR}\n")


def run_command(cmd, description=""):
    """Run a shell command and stream output."""
    if description:
        print(f"\n→ {description}")
    print(f"  Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def get_choice(prompt, valid_choices):
    while True:
        choice = input(prompt).strip().upper()
        if choice in [c.upper() for c in valid_choices]:
            return choice.upper()
        print(f"  Please enter one of: {', '.join(valid_choices)}")


# ============================================================
# STAGE FUNCTIONS
# ============================================================

def stage_simulate():
    print_header("STAGE 1: SIMULATION")
    print("This generates synthetic physiological data that mimics real")
    print("emotional states. It lets you test the full pipeline without")
    print("any physical hardware.\n")
    
    print("How many simulated participants?")
    print("  5  = Quick test (2-3 minutes)")
    print("  15 = Recommended for reliable results (5-8 minutes)")
    print("  30 = Full study power (15-20 minutes)")
    
    n = input("\nNumber of participants [15]: ").strip() or "15"
    sessions = input("Sessions per participant [3]: ").strip() or "3"
    
    print("\nAlso run quick pipeline validation? (y/n) [y]: ", end="")
    validate = input().strip().lower() or "y"
    
    if validate == "y":
        success = run_command(
            [sys.executable, "simulate_experiment.py", "--quick_validate"],
            "Quick validation (5 participants)"
        )
        if not success:
            print("\n✗ Quick validation FAILED. Debug before full simulation.")
            input("Press Enter to return to menu...")
            return
        print("\n✓ Quick validation passed.")
    
    run_command(
        [sys.executable, "simulate_experiment.py",
         "--n_participants", n, "--sessions_each", sessions,
         "--output_dir", "data/raw"],
        f"Generating {n} participants × {sessions} sessions"
    )
    
    print("\n✓ Simulation complete.")
    print("  Next step: Select [3] PROCESS to extract features.\n")
    input("Press Enter to continue...")


def stage_collect():
    print_header("STAGE 2: REAL DATA COLLECTION")
    
    print("HARDWARE CHECKLIST — verify before proceeding:")
    print()
    print("  [ ] Arduino Uno connected via USB")
    print("      - GSR sensor wired to A0, 5V, GND")
    print("      - Two finger electrodes attached to participant")
    print("      - arduino_gsr_reader.ino uploaded and running")
    print()
    print("  [ ] Polar H10 chest strap")
    print("      - Strap on participant's chest, sensor positioned on sternum")
    print("      - Moistened contacts for good conductance")
    print("      - Blue LED flashing = powered on")
    print()
    print("  [ ] Webcam (Logitech C920 or similar)")
    print("      - Positioned at eye level, ~50cm from participant's face")
    print("      - Good frontal lighting (avoid backlit windows)")
    print()
    print("  [ ] Meta Quest 3 connected via USB-C link cable")
    print("      - Developer mode enabled (Settings > About > tap Build 7x)")
    print("      - Quest shows 'Allow USB debugging' prompt → Accept")
    print()
    
    ready = input("All hardware ready? (y/n): ").strip().lower()
    if ready != 'y':
        print("\nSet up hardware first, then return to this option.")
        input("Press Enter to return to menu...")
        return
    
    participant = input("\nParticipant ID (e.g., P001): ").strip() or "P001"
    session = input("Session number (e.g., 1): ").strip() or "1"
    
    print("\nSelect condition:")
    conditions = ["BASELINE", "ANXIETY_EXPOSURE", "CRAVING_EXPOSURE",
                  "POSITIVE_AFFECT", "NEUTRAL", "SOCIAL_STRESS"]
    for i, c in enumerate(conditions, 1):
        print(f"  [{i}] {c}")
    
    c_idx = int(input("Choice [1]: ").strip() or "1") - 1
    condition = conditions[c_idx]
    
    duration = input("Duration in minutes [45]: ").strip() or "45"
    
    print(f"\nStarting session: {participant} / Session {session} / {condition}")
    print("Press Ctrl+C during collection to end early.\n")
    time.sleep(2)
    
    run_command(
        [sys.executable, "data_collector.py",
         "--participant", participant,
         "--session", session,
         "--condition", condition,
         "--duration", duration],
        "Running data collection"
    )
    
    input("\nPress Enter to continue...")


def stage_process():
    print_header("STAGE 3: SIGNAL PROCESSING")
    print("Converting raw sensor recordings into AI-ready feature matrices")
    print("at all time scales: micro, momentary, episodic, session.\n")
    
    raw_folders = glob("data/raw/*/")
    
    if not raw_folders:
        print("✗ No raw data folders found in data/raw/")
        print("  Run SIMULATE or COLLECT first.")
        input("Press Enter to return to menu...")
        return
    
    print(f"Found {len(raw_folders)} session folder(s).\n")
    
    # Check which are already processed
    already_processed = set()
    for folder in raw_folders:
        folder_name = os.path.basename(folder.rstrip('/'))
        processed_path = f"data/processed/{folder_name}"
        if os.path.exists(processed_path) and glob(f"{processed_path}/features_*.csv"):
            already_processed.add(folder)
    
    pending = [f for f in raw_folders if f not in already_processed]
    
    print(f"  Already processed: {len(already_processed)}")
    print(f"  Pending processing: {len(pending)}\n")
    
    if not pending:
        print("All sessions already processed. Proceed to TRAIN.")
        input("Press Enter to return to menu...")
        return
    
    print(f"Processing {len(pending)} session(s)...")
    
    success_count = 0
    for i, folder in enumerate(pending, 1):
        folder_name = os.path.basename(folder.rstrip('/'))
        print(f"\n[{i}/{len(pending)}] {folder_name}")
        
        success = run_command(
            [sys.executable, "signal_processor.py",
             "--session_folder", folder.rstrip('/')],
            f"Processing {folder_name}"
        )
        if success:
            success_count += 1
    
    print(f"\n✓ Processed {success_count}/{len(pending)} sessions successfully.")
    print(f"  Next step: Select [4] TRAIN to build AI models.\n")
    input("Press Enter to continue...")


def stage_train():
    print_header("STAGE 4: AI MODEL TRAINING")
    print("Training multi-scale emotion prediction models.\n")
    
    processed_folders = glob("data/processed/*/")
    n_sessions = len(processed_folders)
    
    if n_sessions < 3:
        print(f"✗ Only {n_sessions} processed sessions found. Need at least 5.")
        print("  Collect or simulate more data first.")
        input("Press Enter to return to menu...")
        return
    
    participants = set()
    for f in processed_folders:
        folder_name = os.path.basename(f.rstrip('/'))
        pid = folder_name.split('_')[0]
        participants.add(pid)
    
    print(f"  Processed sessions: {n_sessions}")
    print(f"  Unique participants: {len(participants)}\n")
    
    if len(participants) < 5:
        print("⚠ Warning: Fewer than 5 participants. Cross-validation will be limited.")
        print("  Results may be overly optimistic. Collect more participant data.")
    
    target = input("Classification target:\n  [1] Binary (positive/negative) [default]\n  [2] 4-class quadrant\nChoice [1]: ").strip() or "1"
    target_arg = "label_binary" if target != "2" else "label_quadrant"
    
    version = input("\nModel version name [v1]: ").strip() or "v1"
    output_dir = f"models/{version}"
    
    print(f"\nTraining models → {output_dir}")
    print("This may take 5-15 minutes depending on dataset size.\n")
    
    success = run_command(
        [sys.executable, "emotion_model.py",
         "--data_dir", "data/processed",
         "--output_dir", output_dir,
         "--target", target_arg],
        "Training emotion models"
    )
    
    if success:
        print(f"\n✓ Training complete. Models saved to: {output_dir}")
        print("  Next step: Select [6] ANALYZE to evaluate performance.\n")
    else:
        print("\n✗ Training encountered errors. Check output above.")
    
    input("Press Enter to continue...")


def stage_serve():
    print_header("STAGE 5: START INFERENCE SERVER")
    print("This starts the real-time prediction server that Unity connects to.\n")
    print("IMPORTANT: This must be running WHILE you use the Unity VR scene.\n")
    
    model_dirs = glob("models/*/")
    if not model_dirs:
        print("✗ No trained models found. Run TRAIN first.")
        input("Press Enter to return to menu...")
        return
    
    print("Available model versions:")
    for i, d in enumerate(model_dirs, 1):
        version = os.path.basename(d.rstrip('/'))
        model_files = glob(f"{d}model_*.pkl")
        print(f"  [{i}] {version} ({len(model_files)} scale models)")
    
    idx = int(input(f"\nSelect model version [1]: ").strip() or "1") - 1
    model_dir = model_dirs[idx].rstrip('/')
    
    port = input("Server port [5000]: ").strip() or "5000"
    
    # Get local IP
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "YOUR_PC_IP"
    
    print(f"\n{'─'*50}")
    print(f"  SERVER STARTING")
    print(f"  Local IP: {local_ip}")
    print(f"  URL: http://{local_ip}:{port}")
    print(f"{'─'*50}")
    print(f"\nIn Unity's EmotionAIBridge component, set:")
    print(f"  SERVER_URL = \"http://{local_ip}:{port}\"")
    print()
    print("Press Ctrl+C to stop the server.\n")
    
    run_command(
        [sys.executable, "inference_server.py",
         "--model_dir", model_dir,
         "--port", port,
         "--host", "0.0.0.0"],
        "Starting inference server"
    )


def stage_analyze():
    print_header("STAGE 6: RESULTS ANALYSIS")
    print("Analyzing model performance and generating redesign recommendations.\n")
    
    model_dirs = glob("models/*/")
    if not model_dirs:
        print("✗ No trained models. Run TRAIN first.")
        input("Press Enter to return to menu...")
        return
    
    print("Select model to analyze:")
    for i, d in enumerate(model_dirs, 1):
        print(f"  [{i}] {os.path.basename(d.rstrip('/'))}")
    
    idx = int(input(f"Choice [1]: ").strip() or "1") - 1
    model_dir = model_dirs[idx].rstrip('/')
    
    output_dir = "analysis_results"
    
    success = run_command(
        [sys.executable, "analyze_results.py",
         "--data_dir", "data/processed",
         "--model_dir", model_dir,
         "--output_dir", output_dir],
        "Running analysis"
    )
    
    if success:
        # Generate hypothesis report
        run_command(
            [sys.executable, "simulate_experiment.py", "--hypothesis_report"],
            "Generating hypothesis test report"
        )
        
        print(f"\n✓ Analysis complete. Charts saved to: {output_dir}/")
        print("  Open these files to review results:")
        for f in glob(f"{output_dir}/*.png"):
            print(f"    {f}")
        recs_path = f"{output_dir}/redesign_recommendations.json"
        if os.path.exists(recs_path):
            with open(recs_path) as f:
                recs = json.load(f)
            print(f"\n  OVERALL STATUS: {recs.get('overall_status', 'UNKNOWN')}")
    
    input("\nPress Enter to continue...")


def stage_test():
    print_header("STAGE 7: AUTOMATED TESTS")
    print("Running test suite to verify all components work correctly.\n")
    
    print("Test types:")
    print("  [1] Basic tests (no hardware/server needed) — recommended")
    print("  [2] Integration tests (inference server must be running)")
    print("  [3] All tests")
    
    choice = input("\nChoice [1]: ").strip() or "1"
    
    if choice == "1":
        cmd = [sys.executable, "-m", "pytest", "tests/test_pipeline.py",
               "-v", "-m", "not integration", "--tb=short"]
    elif choice == "2":
        cmd = [sys.executable, "-m", "pytest", "tests/test_pipeline.py",
               "-v", "-m", "integration", "--tb=short"]
    else:
        cmd = [sys.executable, "-m", "pytest", "tests/test_pipeline.py",
               "-v", "--tb=short"]
    
    run_command(cmd, "Running tests")
    input("\nPress Enter to continue...")


def stage_status():
    print_header("PROJECT STATUS")
    
    # Data
    raw_count = len(glob("data/raw/*/"))
    proc_count = len(glob("data/processed/*/"))
    
    participants_raw = set()
    for f in glob("data/raw/*/"):
        pid = os.path.basename(f.rstrip('/')).split('_')[0]
        participants_raw.add(pid)
    
    print("DATA:")
    print(f"  Raw sessions:        {raw_count} ({len(participants_raw)} participants)")
    print(f"  Processed sessions:  {proc_count}")
    
    # Features
    all_feature_files = glob("data/processed/**/features_*.csv", recursive=True)
    scales_found = set()
    for f in all_feature_files:
        scale = os.path.basename(f).replace('features_', '').replace('.csv', '')
        scales_found.add(scale)
    
    print(f"  Feature scales:      {sorted(scales_found) or 'None yet'}")
    
    # Models
    model_versions = glob("models/*/")
    print(f"\nMODELS:")
    if model_versions:
        for d in model_versions:
            version = os.path.basename(d.rstrip('/'))
            model_files = glob(f"{d}model_*.pkl")
            summary_path = f"{d}training_summary.json"
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
                n_parts = summary.get('n_participants', '?')
                n_samples = summary.get('total_samples', '?')
                print(f"  {version}: {len(model_files)} scale models "
                      f"({n_parts} participants, {n_samples} samples)")
            else:
                print(f"  {version}: {len(model_files)} scale models")
    else:
        print("  No trained models yet.")
    
    # Analysis
    recs_path = "analysis_results/redesign_recommendations.json"
    print(f"\nANALYSIS:")
    if os.path.exists(recs_path):
        with open(recs_path) as f:
            recs = json.load(f)
        status = recs.get('overall_status', 'UNKNOWN')
        color = "✓" if "READY" in status else "⚠" if "ITERATE" in status else "✗"
        print(f"  {color} Status: {status}")
        print(f"  Date: {recs.get('analysis_date', 'Unknown')[:19]}")
    else:
        print("  No analysis run yet.")
    
    # Recommended next step
    print(f"\nRECOMMENDED NEXT STEP:")
    if raw_count == 0:
        print("  → [1] SIMULATE — generate data to begin")
    elif proc_count < raw_count:
        print(f"  → [3] PROCESS — {raw_count - proc_count} sessions need processing")
    elif not model_versions:
        print("  → [4] TRAIN — build AI models from processed data")
    elif not os.path.exists(recs_path):
        print("  → [6] ANALYZE — evaluate model performance")
    else:
        status = recs.get('overall_status', '')
        if "READY" in status:
            print("  → [5] SERVE — start inference server for Unity VR integration")
        else:
            print("  → Review analysis_results/redesign_recommendations.json")
            print("     Then iterate based on recommendations.")
    
    input("\nPress Enter to continue...")


def stage_full_run():
    print_header("FULL PIPELINE RUN")
    print("This runs: Simulate → Process → Train → Analyze\n")
    print("Estimated time: 15-30 minutes\n")
    
    n = input("Number of simulated participants [15]: ").strip() or "15"
    sessions = input("Sessions per participant [3]: ").strip() or "3"
    
    confirm = input(f"\nRun full pipeline with {n} participants × {sessions} sessions? (y/n): ")
    if confirm.lower() != 'y':
        return
    
    steps = [
        ("Simulation", [sys.executable, "simulate_experiment.py",
                        "--n_participants", n, "--sessions_each", sessions]),
        ("Processing", None),  # Handled separately
        ("Training", None), # Handled separately
        ("Analysis", [sys.executable, "analyze_results.py",
                      "--data_dir", "data/processed", "--model_dir", "models/v1",
                      "--output_dir", "analysis_results"]),
        ("Hypothesis Report", [sys.executable, "simulate_experiment.py",
                                "--hypothesis_report"]),
    ]
    
    for step_name, cmd in steps:
        print(f"\n{'─'*50}")
        print(f"  STEP: {step_name}")
        print(f"{'─'*50}")
        
        if step_name == "Processing":
            # Process all raw sessions
            raw_folders = glob("data/raw/*/")
            for folder in raw_folders:
                run_command(
                    [sys.executable, "signal_processor.py",
                     "--session_folder", folder.rstrip('/')],
                    f"Processing {os.path.basename(folder.rstrip('/'))}"
                )
        elif step_name == "Training":
            model_files = glob("models/v1/model_*.pkl")
            if model_files:
                print(f"  Models already exist ({len(model_files)} found). Skipping training.")
                print("  Delete models/v1/ folder if you want to retrain.")
            else:
                run_command([sys.executable, "emotion_model.py",
                     "--data_dir", "data/processed",
                     "--output_dir", "models/v1"], "Training")
        else:
            success = run_command(cmd, step_name)
            if not success:
                print(f"\n✗ {step_name} failed. Stopping pipeline.")
                break
    
    print("\n" + "="*60)
    print("  FULL PIPELINE COMPLETE")
    print("="*60)
    print("\nResults are in: analysis_results/")
    print("Check: hypothesis_test_report.txt for formal results.\n")
    
    input("Press Enter to return to menu...")


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    while True:
        clear()
        print(MENU)
        
        choice = input("Enter choice: ").strip().upper()
        
        if choice == '1':
            stage_simulate()
        elif choice == '2':
            stage_collect()
        elif choice == '3':
            stage_process()
        elif choice == '4':
            stage_train()
        elif choice == '5':
            stage_serve()
        elif choice == '6':
            stage_analyze()
        elif choice == '7':
            stage_test()
        elif choice == '8':
            stage_status()
        elif choice == '9':
            stage_full_run()
        elif choice == 'Q':
            print("\nGoodbye!\n")
            break
        else:
            print("Invalid choice. Press Enter to try again.")
            input()


if __name__ == "__main__":
    main()
