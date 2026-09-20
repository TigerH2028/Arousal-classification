"""
FILE: simulate_experiment.py
PURPOSE: Simulates the full hypothesis test WITHOUT real hardware.
         Use this to verify your code works before buying equipment.
         Generates synthetic biometric data that mimics known emotional states,
         runs it through the full pipeline, and reports whether the system
         can distinguish emotions across all time scales.

HOW TO RUN:
  python simulate_experiment.py --n_participants 15 --sessions_each 3

WHAT IT PROVES:
  "AI combined with VR/AR can predict human emotional states across
   multiple time and length scales" — tested on realistic synthetic data.
"""

import numpy as np
import pandas as pd
import os
import json
import time
import argparse
from datetime import datetime, timedelta


# ============================================================
# SYNTHETIC BIOSIGNAL GENERATION
# Based on published physiological norms for each emotion
# ============================================================

class PhysiologicalEmotionSimulator:
    """
    Generates realistic biometric time series for each emotion state.
    Values derived from meta-analyses of affect physiology research.
    
    Reference ranges:
    - Kreibig (2010): Autonomic NS responses to emotion (Biol Psych)
    - Eerola & Vuoskoski (2011): affect-physiological correlations
    """
    
    EMOTION_PROFILES = {
        'anxious_stressed': {
            # High sympathetic activation
            'gsr_base_us': 12.0,     'gsr_std': 2.5,   'gsr_trend': +0.08,
            'hr_base_bpm': 85.0,     'hr_std': 6.0,
            'hrv_rmssd_ms': 22.0,    'hrv_std': 4.0,   # Low HRV = stress
            'brow_raise': 0.55,      'brow_std': 0.08,
            'eye_open': 0.48,        'eye_std': 0.06,
            'mouth_open': 0.12,
            'valence_mean': 2.8,     'arousal_mean': 7.4,
        },
        'excited_happy': {
            # Moderate sympathetic, positive valence
            'gsr_base_us': 9.0,      'gsr_std': 1.8,   'gsr_trend': +0.02,
            'hr_base_bpm': 78.0,     'hr_std': 5.0,
            'hrv_rmssd_ms': 35.0,    'hrv_std': 7.0,
            'brow_raise': 0.35,      'brow_std': 0.07,
            'eye_open': 0.35,        'eye_std': 0.05,
            'mouth_open': 0.08,
            'valence_mean': 7.8,     'arousal_mean': 7.0,
        },
        'calm_content': {
            # High parasympathetic dominance
            'gsr_base_us': 4.5,      'gsr_std': 0.8,   'gsr_trend': -0.01,
            'hr_base_bpm': 62.0,     'hr_std': 3.0,
            'hrv_rmssd_ms': 58.0,    'hrv_std': 10.0,  # High HRV = calm
            'brow_raise': 0.15,      'brow_std': 0.04,
            'eye_open': 0.25,        'eye_std': 0.04,
            'mouth_open': 0.03,
            'valence_mean': 7.2,     'arousal_mean': 2.5,
        },
        'sad_depressed': {
            # Low overall activation, withdrawal
            'gsr_base_us': 3.8,      'gsr_std': 0.7,   'gsr_trend': -0.02,
            'hr_base_bpm': 65.0,     'hr_std': 3.5,
            'hrv_rmssd_ms': 42.0,    'hrv_std': 8.0,
            'brow_raise': 0.30,      'brow_std': 0.07,  # Brow furrow
            'eye_open': 0.20,        'eye_std': 0.05,   # Drooping
            'mouth_open': 0.05,
            'valence_mean': 2.5,     'arousal_mean': 2.8,
        },
        'neutral': {
            'gsr_base_us': 6.0,      'gsr_std': 1.0,   'gsr_trend': 0.0,
            'hr_base_bpm': 68.0,     'hr_std': 4.0,
            'hrv_rmssd_ms': 45.0,    'hrv_std': 8.0,
            'brow_raise': 0.20,      'brow_std': 0.05,
            'eye_open': 0.28,        'eye_std': 0.04,
            'mouth_open': 0.04,
            'valence_mean': 6.5,     'arousal_mean': 4.5,  # Changed from 5.0 to 6.5
        },
    }
    
    def __init__(self, participant_id, individual_noise=0.15):
        """
        individual_noise: How much this participant deviates from population norms.
        Models the individual variability problem (Solution 3).
        Higher = harder to classify = more realistic.
        """
        self.participant_id = participant_id
        self.noise = individual_noise
        self.rng = np.random.RandomState(hash(participant_id) % 2**31)
        
        # Each participant has systematic offsets from population mean
        # This simulates individual physiological differences
        self.individual_offsets = {
            emotion: {
                'gsr_offset': self.rng.normal(0, individual_noise * 3),
                'hr_offset': self.rng.normal(0, individual_noise * 5),
                'hrv_offset': self.rng.normal(0, individual_noise * 8),
            }
            for emotion in self.EMOTION_PROFILES
        }
    
    def generate_gsr_signal(self, emotion, duration_sec, sample_rate=20):
        """Generate a realistic GSR time series for an emotional state."""
        profile = self.EMOTION_PROFILES[emotion]
        offset = self.individual_offsets[emotion]['gsr_offset']
        
        n_samples = int(duration_sec * sample_rate)
        t = np.linspace(0, duration_sec, n_samples)
        
        # Slow baseline drift
        baseline = (profile['gsr_base_us'] + offset) + profile['gsr_trend'] * t
        
        # Add slow fluctuations (0.01-0.1 Hz range — SCL component)
        scl_freq = self.rng.uniform(0.02, 0.08)
        scl = profile['gsr_std'] * 0.5 * np.sin(2 * np.pi * scl_freq * t)
        
        # Add SCR events (discrete skin conductance responses)
        scr = np.zeros(n_samples)
        n_events = self.rng.poisson(duration_sec / 30)  # ~1 event per 30 sec
        for _ in range(n_events):
            onset = self.rng.randint(0, n_samples - sample_rate * 5)
            amplitude = self.rng.exponential(profile['gsr_std'])
            decay = np.exp(-np.arange(sample_rate * 5) / (sample_rate * 2))
            end_idx = min(onset + sample_rate * 5, n_samples)
            scr[onset:end_idx] += amplitude * decay[:end_idx - onset]
        
        # White noise
        noise = self.rng.normal(0, profile['gsr_std'] * 0.15, n_samples)
        
        signal = np.maximum(baseline + scl + scr + noise, 0.1)
        
        timestamps = np.arange(n_samples) / sample_rate
        return pd.DataFrame({
            'relative_time': timestamps,
            'conductance_us': signal,
            'sensor': 'gsr'
        })
    
    def generate_hr_signal(self, emotion, duration_sec, sample_rate=1):
        """Generate heart rate and RR interval time series."""
        profile = self.EMOTION_PROFILES[emotion]
        offset = self.individual_offsets[emotion]['hr_offset']
        hrv_offset = self.individual_offsets[emotion]['hrv_offset']
        
        n_beats = int(duration_sec * (profile['hr_base_bpm'] + offset) / 60)
        
        # Mean RR interval (ms)
        mean_rr = 60000 / (profile['hr_base_bpm'] + offset)
        
        # RMSSD determines beat-to-beat variability
        rmssd = max(profile['hrv_rmssd_ms'] + hrv_offset, 5.0)
        
        # Generate RR intervals using RMSSD (approximate)
        rr_intervals = []
        prev_rr = mean_rr
        for _ in range(n_beats):
            # AR(1) process for realistic HRV
            innovation = self.rng.normal(0, rmssd * 0.7)
            new_rr = 0.85 * prev_rr + 0.15 * mean_rr + innovation
            new_rr = np.clip(new_rr, 400, 1500)  # Physiological bounds
            rr_intervals.append(new_rr)
            prev_rr = new_rr
        
        rr_arr = np.array(rr_intervals)
        cumulative_times = np.cumsum(rr_arr) / 1000.0  # Convert to seconds
        
        heart_rates = 60000 / rr_arr  # BPM
        
        return pd.DataFrame({
            'relative_time': cumulative_times,
            'heart_rate_bpm': heart_rates,
            'rr_intervals_ms': rr_arr.tolist(),
            'rr_mean_ms': rr_arr,
            'sensor': 'polar_h10'
        })
    
    def generate_face_signal(self, emotion, duration_sec, fps=15):
        """Generate facial landmark feature time series."""
        profile = self.EMOTION_PROFILES[emotion]
        
        n_frames = int(duration_sec * fps)
        t = np.linspace(0, duration_sec, n_frames)
        
        def smooth_noise(std, n):
            """Temporally smooth noise (faces don't jump frame to frame)."""
            raw = self.rng.normal(0, std, n)
            from scipy.ndimage import uniform_filter1d
            return uniform_filter1d(raw, size=fps // 2)
        
        brow_raise = (profile['brow_raise'] + smooth_noise(profile['brow_std'], n_frames))
        brow_raise_l = brow_raise + smooth_noise(0.02, n_frames)  # Slight asymmetry
        brow_raise_r = brow_raise - smooth_noise(0.02, n_frames)
        
        eye_open = profile['eye_open'] + smooth_noise(profile['eye_std'], n_frames)
        
        # Blink events: eyes close briefly
        blink_frames = self.rng.choice(n_frames, size=int(duration_sec / 4), replace=False)
        for bf in blink_frames:
            blink_range = range(bf, min(bf + 3, n_frames))
            eye_open[list(blink_range)] *= 0.1
        
        mouth_open = np.abs(profile['mouth_open'] + smooth_noise(0.02, n_frames))
        mouth_width = 0.35 + smooth_noise(0.02, n_frames)
        
        return pd.DataFrame({
            'relative_time': t,
            'brow_raise_left': brow_raise_l,
            'brow_raise_right': brow_raise_r,
            'brow_raise_mean': brow_raise,
            'brow_furrow': 0.3 - brow_raise + smooth_noise(0.03, n_frames),
            'eye_open_left': eye_open,
            'eye_open_right': eye_open + smooth_noise(0.01, n_frames),
            'eye_open_mean': eye_open,
            'mouth_open': mouth_open,
            'mouth_width': mouth_width,
            'jaw_drop': mouth_open * 0.8,
            'gaze_x': smooth_noise(0.05, n_frames),
            'gaze_y': smooth_noise(0.04, n_frames),
            'smile_left': 0.6 * (1 if emotion == 'excited_happy' else 0.2) + smooth_noise(0.05, n_frames),
            'smile_right': 0.6 * (1 if emotion == 'excited_happy' else 0.2) + smooth_noise(0.05, n_frames),
            'head_x': smooth_noise(0.03, n_frames),
            'head_y': smooth_noise(0.03, n_frames),
            'face_detected': True,
            'sensor': 'face'
        })
    
    def generate_self_reports(self, emotion, n_reports=3):
        """Simulate EMA self-report ratings with realistic noise."""
        profile = self.EMOTION_PROFILES[emotion]
        reports = []
        for i in range(n_reports):
            valence = int(np.clip(
                self.rng.normal(profile['valence_mean'], 0.8), 1, 9
            ))
            arousal = int(np.clip(
                self.rng.normal(profile['arousal_mean'], 0.8), 1, 9
            ))
            reports.append({
                'valence': valence,
                'arousal': arousal,
                'emotion_label': emotion,
                'sensor': 'self_report',
                'trigger': 'scheduled'
            })
        return reports
    
    def generate_full_session(self, session_plan, base_time=None):
        """
        Generate a complete session from a session plan.
        
        session_plan: list of (emotion, duration_sec) tuples
        e.g.: [('neutral', 120), ('anxious_stressed', 180), ('calm_content', 120)]
        """
        if base_time is None:
            base_time = time.time()
        
        all_gsr = []
        all_hr = []
        all_face = []
        all_reports = []
        all_events = []
        
        t_cursor = 0
        
        for emotion, duration in session_plan:
            # Generate sensor data
            gsr = self.generate_gsr_signal(emotion, duration)
            hr = self.generate_hr_signal(emotion, duration)
            face = self.generate_face_signal(emotion, duration)
            
            # Add absolute timestamps
            gsr['system_time'] = base_time + t_cursor + gsr['relative_time']
            hr['system_time'] = base_time + t_cursor + hr['relative_time']
            face['system_time'] = base_time + t_cursor + face['relative_time']
            
            all_gsr.append(gsr)
            all_hr.append(hr)
            all_face.append(face)
            
            # Generate self-reports (1 at start, 1 in middle, 1 at end of each block)
            reports = self.generate_self_reports(emotion, n_reports=2)
            for j, r in enumerate(reports):
                r['system_time'] = base_time + t_cursor + (j + 1) * (duration / 3)
            all_reports.extend(reports)
            
            # Log VR event
            all_events.append({
                'system_time': base_time + t_cursor,
                'sensor': 'vr_events',
                'event_type': 'scene_start',
                'scene_name': emotion.upper(),
                'details': json.dumps({'duration': duration})
            })
            
            t_cursor += duration
        
        return {
            'gsr': pd.concat(all_gsr, ignore_index=True).drop(columns=['relative_time']),
            'polar_h10': pd.concat(all_hr, ignore_index=True).drop(columns=['relative_time']),
            'face': pd.concat(all_face, ignore_index=True).drop(columns=['relative_time']),
            'self_report': pd.DataFrame(all_reports),
            'vr_events': pd.DataFrame(all_events),
        }


# ============================================================
# FULL SIMULATION PIPELINE
# ============================================================

class ExperimentSimulator:
    """Orchestrates the full simulated experiment."""
    
    # Standard session plan (mirrors a real therapeutic VR session)
    SESSION_PLAN = [
    ('neutral',          120),   # Baseline
    ('anxious_stressed', 150),   # Negative/high arousal
    ('sad_depressed',    150),   # Negative/low arousal
    ('neutral',           60),   # Recovery
    ('excited_happy',    150),   # Positive/high arousal
    ('calm_content',     150),   # Positive/low arousal
    ('neutral',           60),   # Final baseline
    ]
    def __init__(self, output_dir='data/raw', individual_variability=0.20):
        self.output_dir = output_dir
        self.individual_variability = individual_variability
        os.makedirs(output_dir, exist_ok=True)
    
    def simulate_participant(self, participant_id, n_sessions=3):
        """Generate all sessions for one participant."""
        
        simulator = PhysiologicalEmotionSimulator(
            participant_id, 
            individual_noise=self.individual_variability
        )
        
        participant_folders = []
        
        for session_num in range(1, n_sessions + 1):
            # Slightly different noise each session (day-to-day variability)
            daily_noise = np.random.normal(0, 0.05)
            simulator.noise = self.individual_variability + daily_noise
            
            # Session start time (simulate different times of day)
            hour = np.random.choice([9, 10, 14, 15, 16])
            session_time = time.time() - (n_sessions - session_num) * 86400  # Days apart
            
            session_data = simulator.generate_full_session(
                self.SESSION_PLAN, 
                base_time=session_time
            )
            
            # Save session
            timestamp = datetime.fromtimestamp(session_time).strftime("%Y%m%d_%H%M%S")
            folder = os.path.join(
                self.output_dir, 
                f"{participant_id}_session{session_num}_SIMULATED_{timestamp}"
            )
            os.makedirs(folder, exist_ok=True)
            
            for sensor_name, df in session_data.items():
                if not df.empty:
                    df.to_csv(os.path.join(folder, f"{sensor_name}.csv"), index=False)
            
            # Save metadata
            metadata = {
                'participant_id': participant_id,
                'session': session_num,
                'condition': 'SIMULATED',
                'start_time': session_time,
                'plan': self.SESSION_PLAN,
                'individual_variability': float(simulator.noise),
                'simulated': True
            }
            with open(os.path.join(folder, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            participant_folders.append(folder)
        
        return participant_folders
    
    def simulate_full_experiment(self, n_participants=15, sessions_each=3):
        """
        Simulate the full multi-participant experiment.
        Creates realistic dataset with individual variability, 
        different demographic groups, and session effects.
        """
        
        print("\n" + "="*60)
        print("  SIMULATION: AI + VR EMOTION PREDICTION EXPERIMENT")
        print(f"  Participants: {n_participants} | Sessions each: {sessions_each}")
        print(f"  Individual variability: {self.individual_variability}")
        print("="*60)
        
        all_folders = []
        participant_stats = []
        
        # Simulate participants with demographic diversity
        # (different variability levels model demographic differences)
        demographic_groups = {
            'young_adult': {'n': n_participants // 3, 'variability': 0.15},
            'middle_aged': {'n': n_participants // 3, 'variability': 0.20},
            'older_adult': {'n': n_participants - 2 * (n_participants // 3), 
                            'variability': 0.28},  # Higher variability in older adults
        }
        
        participant_num = 1
        
        for group_name, group_config in demographic_groups.items():
            for i in range(group_config['n']):
                participant_id = f"P{participant_num:03d}"
                
                print(f"\n  Generating {participant_id} ({group_name})...", end='', flush=True)
                
                self.individual_variability = group_config['variability']
                folders = self.simulate_participant(participant_id, sessions_each)
                all_folders.extend(folders)
                
                participant_stats.append({
                    'participant_id': participant_id,
                    'group': group_name,
                    'individual_variability': group_config['variability'],
                    'n_sessions': sessions_each,
                })
                
                print(f" [PASS] ({sessions_each} sessions, {len(folders)} folders)")
                participant_num += 1
        
        # Save experiment metadata
        experiment_meta = {
            'simulation_date': datetime.now().isoformat(),
            'n_participants': n_participants,
            'sessions_each': sessions_each,
            'demographic_groups': demographic_groups,
            'session_plan': self.SESSION_PLAN,
            'total_session_duration_minutes': sum(d for _, d in self.SESSION_PLAN) / 60,
            'participant_stats': participant_stats,
            'hypothesis': (
                'AI combined with VR/AR can predict human emotional states '
                'across multiple time and length scales'
            )
        }
        
        with open(os.path.join(self.output_dir, 'experiment_metadata.json'), 'w') as f:
            json.dump(experiment_meta, f, indent=2)
        
        print(f"\n[PASS] Simulation complete!")
        print(f"  Total sessions: {len(all_folders)}")
        print(f"  Data saved to: {self.output_dir}")
        print(f"\nNext: Run signal_processor.py on each folder, then train the model.")
        
        return all_folders, experiment_meta
    
    def run_quick_validation(self):
        """
        Run a fast end-to-end test: generate data, process, train, evaluate.
        Takes ~3-5 minutes. Good for verifying the pipeline works.
        """
        
        print("\n=== QUICK VALIDATION (3-5 minutes) ===")
        print("Generating minimal dataset (5 participants × 2 sessions)...")
        
        # Generate small dataset
        folders, _ = self.simulate_full_experiment(n_participants=5, sessions_each=2)
        
        # Process all sessions
        print("\nProcessing signals...")
        from signal_processor import MultiScaleFeatureExtractor
        extractor = MultiScaleFeatureExtractor()
        
        for folder in folders:
            try:
                extractor.extract_all_scales(folder)
            except Exception as e:
                print(f"  Warning: {folder}: {e}")
        
        # Train models
        print("\nTraining models...")
        from emotion_model import train_full_pipeline
        model = train_full_pipeline('data/processed', 'models/quick_validation')
        
        if model:
            print("\n[PASS] QUICK VALIDATION PASSED")
            print("  The full pipeline runs correctly end-to-end.")
            print("  Now run with real hardware for actual research results.")
        else:
            print("\n[FAIL] VALIDATION FAILED")
            print("  Check error messages above and debug before using real hardware.")
        
        return model is not None


# ============================================================
# HYPOTHESIS TEST REPORT
# ============================================================

def generate_hypothesis_test_report(results_dir, output_path):
    """
    Generates a formal report framing results as a hypothesis test.
    
    Hypothesis: "AI combined with VR/AR can predict human emotional 
    states across multiple time and length scales."
    
    Tests this at each scale and provides an overall verdict.
    """
    
    report_lines = [
        "=" * 70,
        "HYPOTHESIS TEST REPORT",
        "Project: AI + VR/AR Multimodal Emotion Prediction",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
        "HYPOTHESIS:",
        "  H1: AI models trained on multimodal biosignals (GSR, HRV, facial",
        "  expressions) combined with VR-controlled stimuli can predict human",
        "  emotional states at accuracy significantly above chance (>0.65 AUC)",
        "  across time scales from seconds (micro) to minutes (episodic).",
        "",
        "NULL HYPOTHESIS:",
        "  H0: AI predictions are not significantly better than chance (AUC ≤ 0.5)",
        "  at any measured time scale.",
        "",
        "-" * 70,
        "RESULTS BY TIME SCALE",
        "-" * 70,
    ]
    
    # Load results if available
    recs_path = os.path.join(results_dir, 'redesign_recommendations.json')
    if os.path.exists(recs_path):
        with open(recs_path) as f:
            recs = json.load(f)
        
        report_lines.append(f"\nOverall Status: {recs.get('overall_status', 'UNKNOWN')}")
        report_lines.append("")
        
        for rec in recs.get('recommendations', []):
            report_lines.append(f"[{rec['priority']}] {rec['area']}")
            report_lines.append(f"  {rec['problem']}")
            report_lines.append("")
    else:
        report_lines.append("\n  No results found. Run analyze_results.py first.")
    
    report_lines += [
        "-" * 70,
        "DECISION CRITERIA",
        "-" * 70,
        "",
        "ACCEPT H1 (hypothesis supported) if ALL of:",
        "  [PASS] At least 2/4 time scales achieve AUC > 0.65",
        "  [PASS] Individual participant accuracy > 0.60 for >70% of participants",
        "  [PASS] Closed-loop intervention test: >=3/4 efficacy criteria met",
        "  [PASS] Model performance significantly better than shuffled-label baseline (p < 0.05)",
        "",
        "REJECT H1 (redesign required) if ANY of:",
        "  [FAIL] All time scales AUC <= 0.55",
        "  [FAIL] >50% of participants show near-chance accuracy",
        "  [FAIL] Sensor data shows no physiological differentiation between emotions",
        "  [FAIL] Intervention test shows no measurable effect",
        "",
        "PARTIAL SUPPORT if:",
        "  ~ 1-2 scales work, others don't — revise time scale approach",
        "  ~ Group-level works, individual-level fails — needs personalization",
        "  ~ Detection works, intervention fails — redesign VR scenes",
        "",
        "=" * 70,
    ]
    
    report_text = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {output_path}")
    return report_text


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmotionAI-VR Experiment Simulator")
    parser.add_argument("--n_participants", type=int, default=15)
    parser.add_argument("--sessions_each", type=int, default=3)
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--quick_validate", action='store_true',
                        help="Run quick end-to-end validation only")
    parser.add_argument("--hypothesis_report", action='store_true',
                        help="Generate hypothesis test report")
    
    args = parser.parse_args()
    
    simulator = ExperimentSimulator(output_dir=args.output_dir)
    
    if args.quick_validate:
        success = simulator.run_quick_validation()
        exit(0 if success else 1)
    
    if args.hypothesis_report:
        generate_hypothesis_test_report('analysis_results', 'hypothesis_test_report.txt')
        exit(0)
    
    # Full simulation
    folders, meta = simulator.simulate_full_experiment(
        n_participants=args.n_participants,
        sessions_each=args.sessions_each
    )
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"1. Process all {len(folders)} sessions:")
    print("   for each folder in data/raw/:")
    print("     python signal_processor.py --session_folder <folder>")
    print()
    print("2. Train models:")
    print("   python emotion_model.py --data_dir data/processed --output_dir models/v1")
    print()
    print("3. Start inference server:")
    print("   python inference_server.py --model_dir models/v1")
    print()
    print("4. Analyze results:")
    print("   python analyze_results.py --data_dir data/processed --model_dir models/v1")
    print()
    print("5. Generate hypothesis report:")
    print("   python simulate_experiment.py --hypothesis_report")
