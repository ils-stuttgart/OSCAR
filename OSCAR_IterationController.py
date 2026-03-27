"""
OSCAR Iteration Controller 
=============================================================================

Usage:
  python OSCAR_IterationController.py --start 1 --end 1 --seed 42
  python OSCAR_IterationController.py --start 2 --end 2 --seed 42  # After manual image generation
  python OSCAR_IterationController.py --start 1 --end 1 --seeds 42 61 116

 fine-tuning:
  python OSCAR_IterationController.py --start 2 --end 2 --seed 42 --fine-tune
"""
import os
import sys
import json
import subprocess
import argparse
import time
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARNING] tqdm not installed. Install with: pip install tqdm")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENT_CONFIG = {
    'generation': {
        'base_percentage': 0.08,    # 8% of cumulative training size per iteration
        'min_images': 5,            # Minimum augmentation
        'max_images': 100,          # Maximum augmentation
        'use_dynamic': True,        # Enable adaptive scaling
        'num_variants': 3,          # Number of random variants for comparison
        'pool_size': 1000,          # Random dataset source pool size
    },
    # CNN TRAINING CONFIG
    'cnn': {
        'epochs': 30,               # 30 Training epochs per iteration
        'batch_size': 32,           # Batch size for training
        'learning_rate': 0.001,     # Learning rate for training
    },
    # SEED CONFIGURATION
    'seeds': {
        'data_seed': 42,
        'model_seeds': [42, 61, 116],
        'multi_seed_default': False,
    },
    # STOPPING CRITERIA
    'stopping': {
        'max_iterations': 1000,          # Safety limit
    },
    # MULTIPROCESSING CONFIG
    'multiprocessing': {
        'enabled': True,
        'max_workers': None,
        'chunk_size': 1,
    },
}

def check_stopping_criteria(iteration_num, seed_suffix=""):
    """Check ontology-driven stopping criteria from query result Q15 only."""
    query_result_path = os.path.join(SCRIPT_DIR, "query_result", f"querying_result{iteration_num}{seed_suffix}.json")
    if not os.path.exists(query_result_path):
        return (False, f"Query result not found: {query_result_path}", False)
    
    with open(query_result_path, 'r') as f:
        data = json.load(f)
    
    # Check Q15: Final Stopping Decision (Ontology-driven)
    results = data.get("results", {})
    q15 = results.get("Q15_Final_Stopping_Decision", [])
    
    if q15:
        decision = q15[0][1] if len(q15[0]) > 1 else ""
        reason = q15[0][2] if len(q15[0]) > 2 else ""
        if "STOP_EXCELLENCE" in decision:
            return (True, reason, False)
        elif "CONTINUE" in decision:
            return (False, reason, False)

    return (False, "No valid Q15 decision found; continue", False)

def run_script(script_name, args=[]):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script_name)] + args
    print(f"\n{'─'*80}\n▶ {script_name} {' '.join(args)}\n{'─'*80}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with code {result.returncode}")
    return result

def run_iteration(iteration_num, finetune_lr=0.0001, seed=42, epochs=None, rebuild_from_scratch=False):
    seed_suffix = f"_seed{seed}"
    print(f"\n{'#'*80}")
    print(f"# ITERATION {iteration_num} - SEED {seed}")
    print(f"{'#'*80}")
    iter_start = time.time()
    
    epochs = epochs if epochs is not None else EXPERIMENT_CONFIG['cnn']['epochs']
    print(f"\n[STEP 1] CNN Training")
    cnn_args = ['--start', str(iteration_num), '--seed', str(seed), 
                '--epochs', str(epochs),
                '--batch_size', str(EXPERIMENT_CONFIG['cnn']['batch_size'])]
    if (not rebuild_from_scratch) and iteration_num > 1:
        cnn_args.append('--fine-tune')
    if iteration_num > 1:
        cnn_args.extend(['--finetune_lr', str(finetune_lr)])
    run_script('OSCAR_CNN.py', cnn_args)
    
    print(f"\n[STEP 2] Ontology Population")
    run_script('OSCAR_Management.py', ['--iteration', str(iteration_num), '--seeds', str(seed)])
    
    print(f"\n[STEP 3] SWRL Rules")
    run_script('OSCAR_Rule.py', ['--iteration', str(iteration_num), '--seeds', str(seed)])
    
    print(f"\n[STEP 4] SPARQL Queries")
    run_script('OSCAR_Query.py', ['--iteration', str(iteration_num), '--seeds', str(seed)])
    
    print(f"\n[STEP 5] Ontology-Guided Dataset Generation")
    gen_args = ['--iteration', str(iteration_num), '--seeds', str(seed),
                '--min-images', str(EXPERIMENT_CONFIG['generation']['min_images']),
                '--max-images', str(EXPERIMENT_CONFIG['generation']['max_images'])]
    try:
        run_script('OSCAR_DatasetGenerator.py', gen_args)
    except Exception as e:
        print(f"⚠ Warning: {e}")
    
    # Step 6: Generate random JSON (for fair comparison in next iteration)
    print(f"\n{'─'*80}")
    print(f"STEP 6: Random JSON Generation (for comparison)")
    print(f"{'─'*80}")
    
    try:
        # Use Random_JSONGenerator to create JSON output
        run_script('OSCAR_Random_JSONGenerator.py', [
            '--iteration', str(iteration_num),
            '--seed', str(seed)
        ])
    except Exception as e:
        print(f"⚠ Random JSON generation warning: {e}")
        print("  Continuing to stopping criteria check...")
    
    # Step 7: Train random baseline (iteration 2+, using previous iteration's random dataset)
    if iteration_num >= 2:
        print(f"\n{'─'*80}")
        print(f"STEP 7: Random-CNN Training (using Random{iteration_num-1})")
        print(f"{'─'*80}")
        
        random_args = [
            '--start', str(iteration_num),
            '--seed', str(seed),
            '--epochs', str(epochs),
            '--use_random_dataset'
        ]
        if (not rebuild_from_scratch) and iteration_num > 1:
            random_args.append('--fine-tune')
        
        try:
            run_script('OSCAR_CNN.py', random_args)
        except Exception as e:
            print(f"⚠ Random CNN training warning: {e}")
            print("  Continuing to stopping criteria check...")
    
    iter_time = time.time() - iter_start
    
    print(f"\n{'#'*80}")
    print(f"# ITERATION {iteration_num} COMPLETE ({iter_time:.1f}s)")
    print(f"{'#'*80}\n")
    
    # Step 8: Check stopping criteria
    should_stop, reason, restore_best = check_stopping_criteria(iteration_num, seed_suffix)
    
    if should_stop:
        print(f"\n{'*'*80}")
        print(f"🛑 STOPPING: {reason}")
        if restore_best:
            print(f"Restoring best model (ScenAIro_best_model.h5)")
        print(f"{'*'*80}\n")
        return (True, restore_best)
    else:
        print(f"\n✓ Continue: {reason}")
        return (False, False)

def run_experiment(start_iter, end_iter, finetune_lr, seed, epochs, rebuild_from_scratch):
    seed_suffix = f"_seed{seed}"
    epochs = epochs or EXPERIMENT_CONFIG['cnn']['epochs']
    
    print(f"\n{'='*80}")
    print(f"OSCAR AUTONOMOUS CLOSED-LOOP EXPERIMENT")
    print(f"{'='*80}")
    print(f"  Seed: {seed}")
    print(f"  Iterations: {start_iter} to {end_iter}")
    print(f"  Epochs: {epochs}")
    print(f"  Fine-tune LR: {finetune_lr}")
    print(f"  Max images/iter: {EXPERIMENT_CONFIG['generation']['max_images']}")
    print(f"  Comparison: Ontology-driven vs Random augmentation")
    print(f"{'='*80}\n")
    
    i = start_iter
    while i <= end_iter:
        should_stop, restore_best = run_iteration(
            i,
            finetune_lr=finetune_lr,
            seed=seed,
            epochs=epochs,
            rebuild_from_scratch=rebuild_from_scratch
        )
        
        if should_stop:
            if restore_best:
                best_acc_file = os.path.join(SCRIPT_DIR, "CNN_Models", f"best_model_accuracy{seed_suffix}.txt")
                if os.path.exists(best_acc_file):
                    with open(best_acc_file, 'r') as f:
                        lines = f.readlines()
                        best_acc = float(lines[0].split(':')[1].strip()) if ':' in lines[0] else float(lines[0].strip())
                        best_iter = int(lines[1].split(':')[1].strip()) if len(lines) > 1 and ':' in lines[1] else 0
                    print(f"\n{'='*80}")
                    print(f"DEPLOYMENT: ScenAIro_best_model.h5")
                    print(f"   Best Accuracy: {best_acc:.2%} (Iteration {best_iter})")
                    print(f"   Seed: {seed}")
                    print(f"{'='*80}\n")
            return (i, "stopped")
        
        i += 1
    
    return (end_iter, "limit_reached")

def main():
    """Main entry point for OSCAR iteration controller."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OSCAR Semi-Automated Closed-Loop Controller (Steps 1-5 per iteration)',
    )
    
    parser.add_argument('--start', type=int, default=1, help='Starting iteration number')
    parser.add_argument('--end', type=int, default=10, help='Ending iteration number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--seeds', nargs='*', help='Seeds: space-separated (12 42 88) or comma-separated (12,42,88) - overrides --seed')
    parser.add_argument('--epochs', type=int, default=None, help=f'Training epochs (default: {EXPERIMENT_CONFIG["cnn"]["epochs"]} from config)')
    parser.add_argument('--finetune-lr', type=float, default=0.0001, help='Fine-tuning learning rate')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Fine-tune from previous iteration (default is rebuild from scratch)')
    
    args = parser.parse_args()
    
    if args.start < 1:
        print("Error: --start must be >= 1")
        sys.exit(1)
    if args.end < args.start:
        print("Error: --end must be >= --start")
        sys.exit(1)
    
    # Handle multi-seed support

    rebuild_from_scratch = not args.fine_tune

    if args.seeds:
        # Accept both "12 42 88" and "12,42,88" formats
        if len(args.seeds) == 1 and ',' in args.seeds[0]:
            seeds = [int(s.strip()) for s in args.seeds[0].split(',')]
        else:
            seeds = [int(s) for s in args.seeds]
        print(f"\n{'#'*80}")
        print(f"# MULTI-SEED EXPERIMENT: {len(seeds)} seeds")
        print(f"# Seeds: {', '.join(map(str, seeds))}")
        print(f"{'#'*80}")
        
        for seed in seeds:
            print(f"\n\n{'='*80}")
            print(f"STARTING EXPERIMENT WITH SEED {seed}")
            print(f"{'='*80}")
            final_iter, reason = run_experiment(
                start_iter=args.start,
                end_iter=args.end,
                finetune_lr=args.finetune_lr,
                seed=seed,
                epochs=args.epochs,
                rebuild_from_scratch=rebuild_from_scratch
            )
            print(f"\n{'='*80}")
            print(f"SEED {seed} COMPLETE - Final iteration: {final_iter}, Reason: {reason}")
            print(f"{'='*80}\n")
        
        print(f"\n{'#'*80}")
        print(f"# ALL SEEDS COMPLETE")
        print(f"# Compare results: python OSCAR_visualize_performance.py")
        print(f"{'#'*80}\n")
    else:
        # Single seed run
        final_iter, reason = run_experiment(
            start_iter=args.start,
            end_iter=args.end,
            finetune_lr=args.finetune_lr,
            seed=args.seed,
            epochs=args.epochs,
            rebuild_from_scratch=rebuild_from_scratch
        )
        
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT COMPLETE")
        print(f"{'#'*80}")
        print(f"  Final iteration: {final_iter}")
        print(f"  Reason: {reason}")
        print(f"  Seed: {args.seed}")
        print(f"\n  Compare results:")
        print(f"    python OSCAR_visualize_performance.py --seed {args.seed}")
        print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
