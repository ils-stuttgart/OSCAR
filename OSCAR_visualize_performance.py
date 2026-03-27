"""
OSCAR Performance Visualization - Ontology-guided vs Random CNN 
=====================================================================
Compares training/validation/testing accuracy across iterations for runway identification
Displays confusion matrix and misclassified images from best model

Supports multi-seed experiments:
  python OSCAR_visualize_performance.py --multi-seed --seeds 61 42 116
  python OSCAR_visualize_performance.py --single-seed --seeds 61

  
Maps to OSCAR domain: runway detection instead of shape classification
"""
import os
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for publication-quality plots with LaTeX-style fonts
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern math fonts
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts in PDF
matplotlib.rcParams['ps.fonttype'] = 42   # TrueType fonts in PS/EPS
matplotlib.rcParams['svg.fonttype'] = 'none'  # Preserve text as text in SVG, not paths
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import centralized config
try:
    from OSCAR_IterationController import EXPERIMENT_CONFIG, get_seed_config
    DEFAULT_SEEDS = EXPERIMENT_CONFIG['seeds']['model_seeds']
except ImportError:
    DEFAULT_SEEDS = [61, 116, 142]
    print("[WARNING] Using default seeds. Import from OSCAR_IterationController for centralized config.")

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARNING] tqdm not installed. Install with: pip install tqdm")

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(SCRIPT_DIR, "OSCAR_Experiments")
# Legacy input_image for initial training data
INPUT_IMAGE_DIR = os.path.join(SCRIPT_DIR, "Ontology_Input_Individuals", "input_image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_output")

def get_results_dir(seed, strategy="Ontology"):
    """Get results directory for a seed and strategy"""
    return os.path.join(EXPERIMENTS_DIR, f"seed_{seed}", strategy, "results")

def get_viz_dir(seed, strategy="Ontology"):
    """Get visualization directory for a seed and strategy"""
    return os.path.join(EXPERIMENTS_DIR, f"seed_{seed}", strategy, "visualizations")

# Plotting style
COLORS = {
    'train': '#4ECDC4',
    'val':  "#54CF54",
    'test': '#FF6B6B'
}

# DATA STRUCTURES
@dataclass
class IterationMetrics:
    iteration: int
    train_acc: float
    val_acc: float
    test_acc: float
    train_loss: float
    val_loss: float
    test_loss: float
    seed: Optional[int] = None  # Track which seed this run used

@dataclass
class AveragedIterationMetrics:
    """Metrics averaged across multiple seed runs"""
    iteration: int
    train_acc_mean: float
    val_acc_mean: float
    test_acc_mean: float
    train_loss_mean: float
    val_loss_mean: float
    test_loss_mean: float
    train_acc_std: float = 0.0
    val_acc_std: float = 0.0
    test_acc_std: float = 0.0
    train_loss_std: float = 0.0
    val_loss_std: float = 0.0
    test_loss_std: float = 0.0

def load_cnn_metrics(cnn_dir: str, prefix: str, seed: Optional[int] = None) -> List[IterationMetrics]:
    """Load metrics for a single seed run"""
    metrics = []
    
    # Determine suffix for seed-specific files
    seed_suffix = f"_seed{seed}" if seed is not None else ""

    base_path = os.path.join(cnn_dir, f"CNN_Base{seed_suffix}.json")
    
    # For Random approach: if base doesn't exist in random dir, load from ontology dir
    # Both ontology and random use the same initial model (iteration 0)
    if not os.path.exists(base_path) and prefix == "Random":
        # Get ontology results dir for the same seed
        ontology_dir = cnn_dir.replace("Random", "Ontology")
        base_path = os.path.join(ontology_dir, f"CNN_Base{seed_suffix}.json")
    
    if os.path.exists(base_path):
        with open(base_path, 'r') as f:
            data = json.load(f)
            pm = data['performance_metrics']  # Shorter variable name
            metrics.append(IterationMetrics(
                iteration=0,
                train_acc=pm['training']['accuracy'],
                val_acc=pm['validation']['accuracy'],
                test_acc=pm['testing']['accuracy'],
                train_loss=pm['training']['loss'],
                val_loss=pm['validation']['loss'],
                test_loss=pm['testing']['loss'],
                seed=seed
            ))
    
    # Load subsequent iterations
    iteration = 1
    while True:
        filepath = os.path.join(cnn_dir, f"CNN_with_{prefix}{iteration}{seed_suffix}.json")
        if not os.path.exists(filepath):
            break
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            pm = data['performance_metrics']
            metrics.append(IterationMetrics(
                iteration=iteration,
                train_acc=pm['training']['accuracy'],
                val_acc=pm['validation']['accuracy'],
                test_acc=pm['testing']['accuracy'],
                train_loss=pm['training']['loss'],
                val_loss=pm['validation']['loss'],
                test_loss=pm['testing']['loss'],
                seed=seed
            ))
        iteration += 1
    
    return metrics

def load_multi_seed_metrics(cnn_dir: str, prefix: str, seeds: List[int]) -> List[List[IterationMetrics]]:
    """Load metrics from multiple seed runs"""
    all_runs = []
    for seed in seeds:
        metrics = load_cnn_metrics(cnn_dir, prefix, seed=seed)
        if metrics:  # Only add if we found data for this seed
            all_runs.append(metrics)
    
    # If no seed-specific files found, try loading without seed suffix (backward compatibility)
    if not all_runs:
        metrics = load_cnn_metrics(cnn_dir, prefix, seed=None)
        if metrics:
            all_runs.append(metrics)
    
    return all_runs

def average_metrics_across_seeds(multi_run_metrics: List[List[IterationMetrics]]) -> List[AveragedIterationMetrics]:

    if not multi_run_metrics:
        return []
    
    # Find max iteration across all runs
    max_iter = max(max(m.iteration for m in run) for run in multi_run_metrics)
    num_seeds = len(multi_run_metrics)
    
    # Build lookup: for each run, map iteration -> metrics
    # Also track the last available metrics for carry-forward
    run_iter_map = []
    run_last_metrics = []
    
    for run in multi_run_metrics:
        iter_map = {m.iteration: m for m in run}
        run_iter_map.append(iter_map)
        # Find the last (highest) iteration for this run
        last_iter = max(m.iteration for m in run)
        run_last_metrics.append(iter_map[last_iter])
    
    averaged = []
    for iter_num in range(max_iter + 1):
        # Collect metrics for this iteration from all runs
        # Use carry-forward when a seed doesn't have data for this iteration
        iter_metrics = []
        carried_forward_count = 0
        
        for run_idx, iter_map in enumerate(run_iter_map):
            if iter_num in iter_map:
                # Normal case: seed has data for this iteration
                iter_metrics.append(iter_map[iter_num])
            else:
                # CARRY-FORWARD: seed stopped early, use its last metrics
                last_m = run_last_metrics[run_idx]
                if last_m.iteration < iter_num:
                    # Only carry forward if this seed stopped BEFORE this iteration
                    iter_metrics.append(last_m)
                    carried_forward_count += 1
        
        if not iter_metrics:
            continue
        
        # Log carry-forward usage (only on first occurrence per iter range)
        if carried_forward_count > 0 and iter_num == max_iter:
            print(f"  ℹ Carry-forward applied: {carried_forward_count}/{num_seeds} seeds stopped early")
        
        # Calculate mean and std
        averaged.append(AveragedIterationMetrics(
            iteration=iter_num,
            train_acc_mean=np.mean([m.train_acc for m in iter_metrics]),
            val_acc_mean=np.mean([m.val_acc for m in iter_metrics]),
            test_acc_mean=np.mean([m.test_acc for m in iter_metrics]),
            train_loss_mean=np.mean([m.train_loss for m in iter_metrics]),
            val_loss_mean=np.mean([m.val_loss for m in iter_metrics]),
            test_loss_mean=np.mean([m.test_loss for m in iter_metrics]),
            train_acc_std=np.std([m.train_acc for m in iter_metrics]),
            val_acc_std=np.std([m.val_acc for m in iter_metrics]),
            test_acc_std=np.std([m.test_acc for m in iter_metrics]),
            train_loss_std=np.std([m.train_loss for m in iter_metrics]),
            val_loss_std=np.std([m.val_loss for m in iter_metrics]),
            test_loss_std=np.std([m.test_loss for m in iter_metrics])
        ))
    
    return averaged

def extract_metric_lists(metrics) -> Dict[str, List]:
    """Extract metrics into lists, supports both regular and averaged metrics"""
    if not metrics:
        return {}
    
    # Check if we have averaged metrics with std dev
    has_std = hasattr(metrics[0], 'train_acc_std')
    
    if has_std:
        return {
            'iters': [m.iteration for m in metrics],
            'train': [m.train_acc_mean for m in metrics],
            'val': [m.val_acc_mean for m in metrics],
            'test': [m.test_acc_mean for m in metrics],
            'train_loss': [m.train_loss_mean for m in metrics],
            'val_loss': [m.val_loss_mean for m in metrics],
            'test_loss': [m.test_loss_mean for m in metrics],
            'train_std': [m.train_acc_std for m in metrics],
            'val_std': [m.val_acc_std for m in metrics],
            'test_std': [m.test_acc_std for m in metrics],
            'train_loss_std': [m.train_loss_std for m in metrics],
            'val_loss_std': [m.val_loss_std for m in metrics],
            'test_loss_std': [m.test_loss_std for m in metrics]
        }
    else:
        return {
            'iters': [m.iteration for m in metrics],
            'train': [m.train_acc for m in metrics],
            'val': [m.val_acc for m in metrics],
            'test': [m.test_acc for m in metrics],
            'train_loss': [m.train_loss for m in metrics],
            'val_loss': [m.val_loss for m in metrics],
            'test_loss': [m.test_loss for m in metrics],
            'train_std': [0] * len(metrics),
            'val_std': [0] * len(metrics),
            'test_std': [0] * len(metrics),
            'train_loss_std': [0] * len(metrics),
            'val_loss_std': [0] * len(metrics),
            'test_loss_std': [0] * len(metrics)
        }

def plot_single_seed_comparison(ontology_metrics,
                                 random_metrics,
                                 save_path: str,
                                 seed: int = None) -> None:
    """Plot comparison for single seed run (same format as multi-seed but without error bars)"""
    if not ontology_metrics:
        print("No ontology CNN data found!")
        return

    ont = extract_metric_lists(ontology_metrics)
    rand = extract_metric_lists(random_metrics) if random_metrics else None
    
    # Single seed - no standard deviations
    has_ont_std = False
    has_rand_std = False
    
    # Add seed info to title if provided
    seed_info = f" (seed {seed})" if seed else ""
    
    # Get base path without extension for separate file names
    base_path = os.path.splitext(save_path)[0]
    # Add seed number to filename if provided
    if seed is not None:
        base_path = f"{base_path}_seed{seed}"
    
    # ========== Plot 1: All Accuracies ==========
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    _plot_metrics_comparison(ax1, ont, rand, 
                             metrics=['train', 'val', 'test'],
                             title=f'Accuracy: Ontology-guided vs Random Augmentation{seed_info}',
                             ylabel='Accuracy', ylim=[0.5, 1.01])
    plt.tight_layout()
    save_path_1 = f"{base_path}_1_all_accuracies"
    plt.savefig(f"{save_path_1}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path_1}.svg", bbox_inches='tight')
    plt.savefig(f"{save_path_1}.pdf", bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ Plot 1 saved: {save_path_1}.(png/svg/pdf)")
    
    # ========== Plot 2: Test Accuracy Focus with Best Points ==========
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(ont['iters'], ont['test'], '-', label='Ontology Test',
             color='darkorange', linewidth=3) 
    
    # Mark best ontology point
    best_ont_idx = np.argmax(ont['test'])
    best_ont_pct = int(ont['test'][best_ont_idx] * 10000) / 100 
    label_text = f'Best Ontology: {best_ont_pct:.2f}%'  # (Iter {ont["iters"][best_ont_idx]})
    
    ax2.scatter(ont['iters'][best_ont_idx], ont['test'][best_ont_idx],
                s=300, c='gold', marker='*', edgecolors='black', linewidths=1,
                label=label_text, zorder=5)
    
    if rand:
        ax2.plot(rand['iters'], rand['test'], '--', label='Random Test',
                 color='#2E86AB', linewidth=3)
        
        best_rand_idx = np.argmax(rand['test'])
        best_rand_pct = int(rand['test'][best_rand_idx] * 10000) / 100
        label_text_rand = f'Best Random: {best_rand_pct:.2f}%'  # (Iter {rand["iters"][best_rand_idx]})
        
        ax2.scatter(rand['iters'][best_rand_idx], rand['test'][best_rand_idx],
                    s=300, c='#4B9DC0', marker='*', edgecolors='black', linewidths=1,
                    label=label_text_rand, zorder=5)
    
    _style_axis(ax2, f'Test Accuracy Comparison{seed_info}', 'Test Accuracy', 
                ylim=[0.5, 1.04], legend_loc='lower right')
    plt.tight_layout()
    save_path_2 = f"{base_path}_2_test_accuracy"
    plt.savefig(f"{save_path_2}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path_2}.svg", bbox_inches='tight')
    plt.savefig(f"{save_path_2}.pdf", bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Plot 2 saved: {save_path_2}.(png/svg/pdf)")
    
    # ========== Plot 3: All Losses ==========
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    _plot_metrics_comparison(ax3, ont, rand,
                             metrics=['train_loss', 'val_loss', 'test_loss'],
                             title=f'Loss: Ontology-guided vs Random Augmentation{seed_info}',
                             ylabel='Loss', legend_loc='upper right')
    plt.tight_layout()
    save_path_3 = f"{base_path}_3_all_losses"
    plt.savefig(f"{save_path_3}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path_3}.svg", bbox_inches='tight')
    plt.savefig(f"{save_path_3}.pdf", bbox_inches='tight')
    plt.close(fig3)
    print(f"✓ Plot 3 saved: {save_path_3}.(png/svg/pdf)")
    
    # ========== Plot 4: Test Loss Focus ==========
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(ont['iters'], ont['test_loss'], '-', label='Ontology Test',
             color='darkorange', linewidth=3)
    
    best_ont_loss_idx = np.argmin(ont['test_loss'])
    label_text_loss = f'Best Ontology: {ont["test_loss"][best_ont_loss_idx]:.4f}'  # (Iter {ont["iters"][best_ont_loss_idx]})
    
    ax4.scatter(ont['iters'][best_ont_loss_idx], ont['test_loss'][best_ont_loss_idx],
                s=300, c='gold', marker='*', edgecolors='black', linewidths=1,
                label=label_text_loss, zorder=5)
    
    if rand:
        ax4.plot(rand['iters'], rand['test_loss'], '--', label='Random Test',
                 color='#2E86AB', linewidth=3)
        1
        best_rand_loss_idx = np.argmin(rand['test_loss'])
        label_text_rand_loss = f'Best Random: {rand["test_loss"][best_rand_loss_idx]:.4f}'  # (Iter {rand["iters"][best_rand_loss_idx]})
        
        ax4.scatter(rand['iters'][best_rand_loss_idx], rand['test_loss'][best_rand_loss_idx],
                    s=300, c='#4B9DC0', marker='*', edgecolors='black', linewidths=1,
                    label=label_text_rand_loss, zorder=5)
    
    _style_axis(ax4, f'Test Loss Comparison{seed_info}', 'Test Loss', legend_loc='upper right')
    plt.tight_layout()
    save_path_4 = f"{base_path}_4_test_loss"
    plt.savefig(f"{save_path_4}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path_4}.svg", bbox_inches='tight')
    plt.savefig(f"{save_path_4}.pdf", bbox_inches='tight')
    plt.close(fig4)
    print(f"✓ Plot 4 saved: {save_path_4}.(png/svg/pdf)")
    
    print(f"\n{'='*80}")
    print(f"✓ All 4 single-seed plots saved to: {os.path.dirname(save_path)}")
    print(f"{'='*80}\n")

def plot_comparison(ontology_metrics,
                    random_metrics,
                    save_path: str,
                    seeds: List[int] = None) -> None:
    """Plot test accuracy and test loss with filled variance bands"""
    if not ontology_metrics:
        print("No ontology CNN data found!")
        return

    ont = extract_metric_lists(ontology_metrics)
    rand = extract_metric_lists(random_metrics) if random_metrics else None
    
    # Check if we have standard deviations (multi-seed runs)
    has_ont_std = ont and any(ont.get('test_std', []))
    has_rand_std = rand and any(rand.get('test_std', []))
    
    # Create seed info for titles
    if seeds:
        seed_info = f" (seeds {', '.join(map(str, seeds))})"
    else:
        seed_info = " (3-seed average)"
    
    # Get base path without extension for separate file names
    base_path = os.path.splitext(save_path)[0]
    
    # ========== Plot 1: Test Accuracy with Variance Bands ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Ontology line and variance band
    ont_iters = np.array(ont['iters'])
    ont_test = np.array(ont['test'])
    ont_test_std = np.array(ont['test_std'])
    
    ax1.plot(ont_iters, ont_test, '-', label='Ontology-guided', 
             color='darkorange', linewidth=3, zorder=3)
    
    if has_ont_std:
        ax1.fill_between(ont_iters, 
                         ont_test - ont_test_std, 
                         ont_test + ont_test_std,
                         color='darkorange', alpha=0.1, zorder=1,
                         label='Ontology variance')
    
    # Random line and variance band
    if rand:
        rand_iters = np.array(rand['iters'])
        rand_test = np.array(rand['test'])
        rand_test_std = np.array(rand['test_std'])
        
        ax1.plot(rand_iters, rand_test, '--', label='Random Augmentation', #TODO
                 color='#2E86AB', linewidth=3, zorder=3)
        
        if has_rand_std:
            ax1.fill_between(rand_iters,
                             rand_test - rand_test_std,
                             rand_test + rand_test_std,
                             color='#2E86AB', alpha=0.1, zorder=1,
                             label='Random variance')
    
    # Mark best points
    best_ont_idx = np.argmax(ont_test)
    best_ont_pct = ont_test[best_ont_idx] * 100
    best_ont_std_pct = ont_test_std[best_ont_idx] * 100 if has_ont_std else 0
    best_ont_iter = ont_iters[best_ont_idx]
    
    ont_label = f'Best Ontology: {best_ont_pct:.2f}%'
    if has_ont_std:
        ont_label += f' (±{best_ont_std_pct:.2f}%)'
    # ont_label += f' (iter {best_ont_iter})'
    
    ax1.scatter(best_ont_iter, ont_test[best_ont_idx],
                s=200, c='gold', marker='*', edgecolors='black', linewidths=1,
                label=ont_label, zorder=5)
    
    if rand:
        best_rand_idx = np.argmax(rand_test)
        best_rand_pct = rand_test[best_rand_idx] * 100
        best_rand_std_pct = rand_test_std[best_rand_idx] * 100 if has_rand_std else 0
        best_rand_iter = rand_iters[best_rand_idx]
        
        rand_label = f'Best Random: {best_rand_pct:.2f}%'
        if has_rand_std:
            rand_label += f' (±{best_rand_std_pct:.2f}%)'
        # rand_label += f' (iter {best_rand_iter})'
        
        ax1.scatter(best_rand_iter, rand_test[best_rand_idx],
                    s=200, c="#4B9DC0", marker='*', edgecolors='black', linewidths=1,
                    label=rand_label, zorder=5)
    
    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontweight='bold')
    # ax1.set_title(f'Test Accuracy Comparison{seed_info}', fontweight='bold', pad=15)  # Commented for LaTeX caption
    ax1.legend(loc='lower right', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0.5, 1.02])
    ax1.set_xlim(left=0)
    ax1.xaxis.set_major_locator(MultipleLocator(2))  # Even numbers only
    
    plt.tight_layout()
    save_path_1 = f"{base_path}_test_accuracy"
    plt.savefig(f"{save_path_1}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path_1}.svg", bbox_inches='tight')
    plt.savefig(f"{save_path_1}.pdf", bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ Test Accuracy plot saved: {save_path_1}.(png/svg/pdf)")
    
    # ========== Plot 2: Test Loss with Variance Bands ==========
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Ontology loss line and variance band
    ont_test_loss = np.array(ont['test_loss'])
    ont_test_loss_std = np.array(ont['test_loss_std'])
    
    ax2.plot(ont_iters, ont_test_loss, '-', label='Ontology-guided',
             color='darkorange', linewidth=3, zorder=3)
    
    if has_ont_std:
        ax2.fill_between(ont_iters,
                         ont_test_loss - ont_test_loss_std,
                         ont_test_loss + ont_test_loss_std,
                         color='darkorange', alpha=0.1, zorder=1,
                         label='Ontology variance')
    
    # Random loss line and variance band
    if rand:
        rand_test_loss = np.array(rand['test_loss'])
        rand_test_loss_std = np.array(rand['test_loss_std'])
        
        ax2.plot(rand_iters, rand_test_loss, '--', label='Random augmentation',
                 color='#2E86AB', linewidth=3, zorder=3)
        
        if has_rand_std:
            ax2.fill_between(rand_iters,
                             rand_test_loss - rand_test_loss_std,
                             rand_test_loss + rand_test_loss_std,
                             color='#2E86AB', alpha=0.1, zorder=1,
                             label='Random variance')
    
    # Mark best points (lowest loss)
    best_ont_loss_idx = np.argmin(ont_test_loss)
    best_ont_loss = ont_test_loss[best_ont_loss_idx]
    best_ont_loss_std = ont_test_loss_std[best_ont_loss_idx] if has_ont_std else 0
    best_ont_loss_iter = ont_iters[best_ont_loss_idx]
    
    ont_loss_label = f'Best Ontology: {best_ont_loss:.4f}'
    if has_ont_std:
        ont_loss_label += f' (±{best_ont_loss_std:.4f})'
    # ont_loss_label += f' (iter {best_ont_loss_iter})'
    
    ax2.scatter(best_ont_loss_iter, best_ont_loss,
                s=200, c='gold', marker='*', edgecolors='black', linewidths=1,
                label=ont_loss_label, zorder=5)
    
    if rand:
        best_rand_loss_idx = np.argmin(rand_test_loss)
        best_rand_loss = rand_test_loss[best_rand_loss_idx]
        best_rand_loss_std = rand_test_loss_std[best_rand_loss_idx] if has_rand_std else 0
        best_rand_loss_iter = rand_iters[best_rand_loss_idx]
        
        rand_loss_label = f'Best Random: {best_rand_loss:.4f}'
        if has_rand_std:
            rand_loss_label += f' (±{best_rand_loss_std:.4f})'
        # rand_loss_label += f' (iter {best_rand_loss_iter})'
        
        ax2.scatter(best_rand_loss_iter, best_rand_loss,
                    s=200, c='#4B9DC0', marker='*', edgecolors='black', linewidths=1,
                    label=rand_loss_label, zorder=5)
    
    ax2.set_xlabel('Iteration', fontweight='bold')
    ax2.set_ylabel('Test Loss', fontweight='bold')
    # ax2.set_title(f'Test Loss Comparison{seed_info}', fontweight='bold', pad=15) 
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)
    ax2.xaxis.set_major_locator(MultipleLocator(2))  # Even numbers only
    
    plt.tight_layout()
    save_path_2 = f"{base_path}_test_loss"
    plt.savefig(f"{save_path_2}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path_2}.svg", bbox_inches='tight')
    plt.savefig(f"{save_path_2}.pdf", bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Test Loss plot saved: {save_path_2}.(png/svg/pdf)")
    
    print(f"\n{'='*80}")
    print(f"✓ Both comparison plots saved to: {os.path.dirname(save_path)}")
    print(f"{'='*80}\n")
# all Training/Validation/Testing Accuracies and Losses from Ontology and Random
def _plot_metrics_comparison(ax, ont: Dict, rand: Optional[Dict],
                              metrics: List[str], title: str, ylabel: str,
                              ylim: Optional[List] = None,
                              legend_loc: str = 'lower right') -> None:

    # Map metric names to display labels and colors
    metric_info = {
        'train': ('Train', COLORS['train'], '-'),
        'val': ('Val', COLORS['val'], '-'),
        'test': ('Test', COLORS['test'], '-'),
        'train_loss': ('Train', COLORS['train'], '-'),
        'val_loss': ('Val', COLORS['val'], '-'),
        'test_loss': ('Test', COLORS['test'], '-')
    }
    
    for metric in metrics:
        label, color, style = metric_info[metric]
        ax.plot(ont['iters'], ont[metric], style, 
                label=f'Ontology {label}', color=color, linewidth=3)
        
        if rand:
            ax.plot(rand['iters'], rand[metric], style.replace('-', '--'),
                    label=f'Random {label}', color=color, alpha=0.5, 
                    linewidth=3)
    
    _style_axis(ax, title, ylabel, ylim, legend_loc)
def _style_axis(ax, title: str, ylabel: str, 
                ylim: Optional[List] = None,
                legend_loc: str = 'lower right') -> None:
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    # ax.set_title(title, fontweight='bold')  # Commented for LaTeX caption
    ax.legend(loc=legend_loc)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MultipleLocator(2))  # Even numbers only
    if ylim:
        ax.set_ylim(ylim)

# ============================================================================
# OSCAR ODD Dimension Definitions
# ============================================================================

# ODD1: Runway Classification
RUNWAY_CLASSES = {'runway': 'with_runway', 'no_runway': 'no_runway'}

# ODD2: Airports (ICAO codes)
AIRPORTS = ['EDDS', 'EDDV', 'EDNY', 'EDSB', 'ELLX', 'ENBR', 'KLAX']
AIRPORT_NAMES = {
    'EDDS': 'Stuttgart',
    'EDDV': 'Hannover',
    'EDNY': 'Friedrichshafen',
    'EDSB': 'Karlsruhe',
    'ELLX': 'Luxembourg',
    'ENBR': 'Bergen',
    'KLAX': 'Los Angeles'
}

# ODD3: Time of Day
TIME_CATEGORIES = ['Daytime', 'Nighttime']

# ============================================================================
# Helper Functions for Initial Dataset Analysis
# ============================================================================

def _load_single_json(args: Tuple[str, str, str]) -> Optional[Dict]:
    """Load single JSON file and extract OSCAR runway data."""
    filepath, split, runway_label = args
    try:
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        # OSCAR: Extract runway metadata from ScenAIro format
        images = metadata.get("images", [])
        if not images:
            return None
            
        image_info = images[0]
        
        # Ground truth from categories
        categories = metadata.get("categories", [{}])
        category_name = categories[0].get("name", "no_runway") if categories else "no_runway"
        
        # Airport and time data
        runway_data = metadata.get("runway_data", {})
        icao_code = runway_data.get("icao_code", "UNKNOWN")
        
        daytime_data = metadata.get("daytime", {})
        hour = int(daytime_data.get("hours", 12))
        
        # Classify time of day (STRICT ODD boundaries from TIME_OF_DAY_THRESHOLDS)
        # Daytime: 10-15 inclusive (10 <= hour < 16), Nighttime: 0-5 inclusive (0 <= hour < 6)
        if 10 <= hour < 16:
            time_category = 'Daytime'
        elif 0 <= hour < 6:
            time_category = 'Nighttime'
        else:
            time_category = 'Other'  # Outside defined ODD ranges
        
        split_name = {"train": "Train", "test": "Test", "val": "Val"}.get(split, split)
        
        return {
            'runway_class': category_name,
            'airport': icao_code,
            'airport_name': AIRPORT_NAMES.get(icao_code, icao_code),
            'time_category': time_category,
            'hour': hour,
            'split': split_name
        }
    except Exception as e:
        print(f"⚠ Error loading {filepath}: {e}")
        return None


def _load_initial_datasets_metadata(base_dir: str = None) -> List[Dict]:
    """Load OSCAR initial datasets using parallel I/O."""
    if base_dir is None:
        base_dir = INPUT_IMAGE_DIR
    
    # Collect all JSON file paths - ScenAIro structure: train/test/val/{norunway,runway} with JSON files
    json_files = []
    for split in ["train", "test", "val"]:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        # ScenAIro: JSON files are in norunway/runway subfolders
        for runway_label in ["norunway", "runway"]:
            label_dir = os.path.join(split_dir, runway_label)
            if not os.path.exists(label_dir):
                continue
                
            for filename in os.listdir(label_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(label_dir, filename)
                    json_files.append((filepath, split, runway_label))
    
    if not json_files:
        return []
    
    # Parallel loading (I/O bound)
    with ThreadPoolExecutor(max_workers=min(8, len(json_files))) as executor:
        results = executor.map(_load_single_json, json_files)
        return [r for r in results if r is not None]

def plot_initial_datasets(save_path: str = None, base_dir: str = None) -> None:
    """
    Visualize OSCAR initial datasets with OntoLoop-mapped ODD dimensions.
    
    OSCAR ODD Mapping to OntoLoop:
    - Plot 2: Airport (7 ICAO codes) - General distribution 
    - Plot 4: Airport Balance (7 values) → Maps to Color position (OntoLoop: 3 colors)
    - Plot 5: Time of Day Balance (2 values) → Maps to Area position (OntoLoop: 3 sizes)
    - Plot 6: Runway Class Balance (2 values) → Maps to Shape position (OntoLoop: 2 targets)
    """
    # 1. Load Data
    raw_data = _load_initial_datasets_metadata(base_dir)
    if not raw_data:
        print("⚠ No data found! Make sure initial datasets exist.")
        return
    df = pd.DataFrame(raw_data)
    total_images = len(df)
    print(f"✓ Loaded {total_images} images for analysis")

    # 2. Setup Figure - 2 rows × 3 columns
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.25)

    # --- PLOT 1: Dataset Split Distribution (Pie Chart) ---
    ax1 = fig.add_subplot(gs[0, 0])
    split_order = ['Train', 'Val', 'Test']
    split_counts = df['split'].value_counts()
    split_counts = split_counts.reindex(split_order, fill_value=0)
    
    split_colors_pie = {'Train': '#4ECDC4', 'Test': '#FF6B6B', 'Val': '#54CF54'}
    colors = [split_colors_pie.get(s, '#CCCCCC') for s in split_counts.index]
    
    # Bigger donut with percentages in the middle of the ring
    wedges, texts, autotexts = ax1.pie(split_counts, labels=split_counts.index, autopct='%1.1f%%', 
                                        colors=colors,
                                        startangle=140, pctdistance=0.75, 
                                        explode=[0.05]*len(split_counts),  
                                        textprops={'fontsize': 18, 'weight': 'bold'})  
    plt.setp(autotexts, size=16, weight="bold", color="white")  
    plt.setp(texts, size=18, weight="bold")
    
    centre_circle = plt.Circle((0,0),0.50,fc='white')
    ax1.add_artist(centre_circle)
    ax1.text(0, 0, f'Total\n{total_images}', ha='center', va='center', fontsize=20, fontweight='bold', color='black') 
    ax1.set_title('Total Data Split', fontsize=18, fontweight='bold', pad=15)

    # --- PLOT 2: Airport Distribution (Bar Chart) - Spans 2 columns ---
    ax2 = fig.add_subplot(gs[0, 1:3])
    airport_counts = df.groupby(['split', 'airport']).size().unstack(fill_value=0)
    
    # Color palette for splits
    split_colors = {'Train': '#4ECDC4', 'Test': '#FF6B6B', 'Val': '#54CF54'}
    
    airport_counts.T.plot(kind='bar', ax=ax2, 
                          color=[split_colors.get(s, 'grey') for s in airport_counts.index])
    
    ax2.set_title(f'Airport Distribution Across Splits (n={total_images})', 
                  fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel('Airport (ICAO)', fontsize=16, fontweight='bold')  
    ax2.set_ylabel('Image Count', fontsize=16, fontweight='bold')  
    ax2.legend(title='Split', loc='upper right', fontsize=18, title_fontsize=18)  
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='both', labelsize=14)  
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')



    # --- PLOT 4: Airport Balance per Split (Stacked Bar) - Maps to OntoLoop Color ---
    ax4 = fig.add_subplot(gs[1, 0])
    airport_counts_df = df.groupby(['split', 'airport']).size().unstack(fill_value=0)
    airport_counts_df = airport_counts_df.reindex(['Train', 'Val', 'Test'], fill_value=0)
    
    # Convert to percentages
    airport_pcts = airport_counts_df.div(airport_counts_df.sum(axis=1), axis=0) * 100

    # Color palette for airports (7 distinct colors)
    airport_colors = {
        'EDDS': '#FF6B6B', 'EDDV': '#4ECDC4', 'EDNY': '#FFD93D',
        'EDSB': '#6BCB77', 'ELLX': '#9B59B6', 'ENBR': '#E67E22',
        'KLAX': '#3498DB'
    }
    colors_mapped = [airport_colors.get(col, '#888888') for col in airport_pcts.columns]

    airport_pcts.plot(kind='bar', stacked=True, color=colors_mapped, ax=ax4)
    
    ax4.set_title('Airport Balance per Split', 
                  fontsize=18, fontweight='bold', pad=15)
    ax4.set_ylabel('Percentage (%)', fontsize=16, fontweight='bold')  
    ax4.set_xlabel('', fontsize=16)
    ax4.legend(title='Airport', loc='upper right', bbox_to_anchor=(1.10, 1), fontsize=12, title_fontsize=13)  
    ax4.tick_params(axis='both', labelsize=14)  
    
    # Add percentage labels on bars
    for c in ax4.containers:
        labels = [f'{v.get_height():.0f}%' if v.get_height() > 5 else '' for v in c]
        ax4.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=14)  

    # --- PLOT 5: Time of Day Balance per Split (Stacked Bar) - Maps to OntoLoop Area ---
    ax5 = fig.add_subplot(gs[1, 1])
    time_counts = df.groupby(['split', 'time_category']).size().unstack(fill_value=0)
    time_counts = time_counts.reindex(['Train', 'Val', 'Test'], fill_value=0)
    
    # Ensure all categories exist (Daytime/Nighttime are ODD, Other is outside ODD)
    for cat in ['Daytime', 'Nighttime', 'Other']:
        if cat not in time_counts.columns:
            time_counts[cat] = 0
    
    # Convert to percentages
    time_pcts = time_counts.div(time_counts.sum(axis=1), axis=0) * 100
    
    # Colors for Time Categories (Daytime/Nighttime=ODD, Other=out-of-ODD)
    time_colors = {'Daytime': '#2ECC71', 'Nighttime':'#3498DB', 'Other': '#FFB84D'}
    colors_mapped = [time_colors.get(col, '#888888') for col in time_pcts.columns]
    
    time_pcts.plot(kind='bar', stacked=True, color=colors_mapped, ax=ax5)
    
    ax5.set_title('Time of Day Balance per Split', 
                  fontsize=18, fontweight='bold', pad=15)
    ax5.set_ylabel('Percentage (%)', fontsize=16, fontweight='bold') 
    ax5.set_xlabel('', fontsize=16)
    ax5.legend(title='Time Category', loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=12, title_fontsize=13)  
    ax5.tick_params(axis='both', labelsize=14)  
    
    # Add percentage labels on bars
    for c in ax5.containers:
        labels = [f'{v.get_height():.0f}%' if v.get_height() > 5 else '' for v in c]
        ax5.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=14) 

    # --- PLOT 6: Runway Class Balance per Split (Stacked Bar) - Maps to OntoLoop Shape ---
    ax6 = fig.add_subplot(gs[1, 2])
    runway_counts = df.groupby(['split', 'runway_class']).size().unstack(fill_value=0)
    # Reorder splits as Train, Val, Test
    runway_counts = runway_counts.reindex(['Train', 'Val', 'Test'], fill_value=0)
    
    # Ensure both classes exist
    target_cols = ['runway', 'no_runway']
    for col in target_cols:
        if col not in runway_counts.columns:
            runway_counts[col] = 0
    runway_counts = runway_counts[target_cols]  # Ensure order
    
    # Calculate Percentages
    runway_pcts = runway_counts.div(runway_counts.sum(axis=1), axis=0) * 100

    # Colors for Runway Class (analogous to Shape: Circle/Square → runway/no_runway)
    runway_colors_map = {'runway': '#FF9999', 'no_runway': '#66B2FF'}
    colors_mapped_runway = [runway_colors_map.get(col, '#888888') for col in runway_pcts.columns]
    
    # Plot
    runway_pcts.plot(kind='bar', stacked=True, color=colors_mapped_runway, ax=ax6)
    
    ax6.set_title('Runway Class Balance per Split', 
                  fontsize=18, fontweight='bold', pad=15)
    ax6.set_ylabel('Percentage (%)', fontsize=16, fontweight='bold')  
    ax6.set_xlabel('', fontsize=16)
    ax6.legend(title='Runway Class', loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=12, title_fontsize=13)  
    ax6.tick_params(axis='both', labelsize=14)  
    
    # Add percentage labels on bars
    for c in ax6.containers:
        labels = [f'{v.get_height():.0f}%' if v.get_height() > 5 else '' for v in c]
        ax6.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=14) 

    # Save and display
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "OSCAR_InitialDatasets_Analysis")
    else:
        save_path = os.path.splitext(save_path)[0]  # Remove extension if provided
    
    plt.savefig(f"{save_path}.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{save_path}.svg", bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"✓ OSCAR initial datasets analysis saved: {save_path}.(png/svg/pdf)")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def visualize_all(use_multi_seed: bool = True, seeds: List[int] = [42, 123, 456]):
    """
    Main visualization function for OSCAR runway identification.
    
    Args:
        use_multi_seed: If True, load and average multiple seed runs
        seeds: List of seeds to load data for
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Determine directories based on seed(s)
    primary_seed = seeds[0] if seeds else 61
    INPUT_CNN_DIR = get_results_dir(primary_seed, "Ontology")
    RANDOM_CNN_DIR = get_results_dir(primary_seed, "Random")
    
    # Visualize initial datasets
    plot_initial_datasets(
        save_path=os.path.join(OUTPUT_DIR, "OSCAR_InitialDatasets_Analysis")
    )
    
    # Load metrics
    if use_multi_seed:
        print(f"\nLoading multi-seed runs with seeds: {seeds}")
        print("=" * 80)
        
        # Load ontology metrics from multiple seeds - need to check all seed directories
        ont_multi_runs = []
        for seed in seeds:
            seed_dir = get_results_dir(seed, "Ontology")
            metrics = load_cnn_metrics(seed_dir, "Rec", seed=seed)
            if metrics:
                ont_multi_runs.append(metrics)
        
        print(f"✓ Loaded {len(ont_multi_runs)} ontology seed runs")
        
        # Load random metrics from multiple seeds
        rand_multi_runs = []
        for seed in seeds:
            seed_dir = get_results_dir(seed, "Random")
            metrics = load_cnn_metrics(seed_dir, "Random", seed=seed)
            if metrics:
                rand_multi_runs.append(metrics)
        
        print(f"✓ Loaded {len(rand_multi_runs)} random seed runs")
        
        # Average across seeds
        ontology_metrics = average_metrics_across_seeds(ont_multi_runs)
        random_metrics = average_metrics_across_seeds(rand_multi_runs) if rand_multi_runs else None
        
        print(f"\n✓ Averaged metrics across {len(ont_multi_runs)} ontology runs")
        if rand_multi_runs:
            print(f"✓ Averaged metrics across {len(rand_multi_runs)} random runs")
        print("=" * 80 + "\n")
    else:
        # Single seed mode: Load data for first seed or without seed suffix
        print("\nLoading single-seed runs")
        print("=" * 80)
        
        # Try loading with seed suffix first
        single_seed = seeds[0] if seeds else None
        seed_dir_ont = get_results_dir(single_seed, "Ontology")
        seed_dir_rand = get_results_dir(single_seed, "Random")
        
        ontology_metrics = load_cnn_metrics(seed_dir_ont, "Rec", seed=single_seed)
        random_metrics = load_cnn_metrics(seed_dir_rand, "Random", seed=single_seed)
        
        # If no data found with seed, try legacy mode (no seed suffix)
        if not ontology_metrics:
            print(f"No data found for seed {single_seed}, trying legacy mode...")
            ontology_metrics = load_cnn_metrics(seed_dir_ont, "Rec", seed=None)
            random_metrics = load_cnn_metrics(seed_dir_rand, "Random", seed=None)
            single_seed = None
        
        print("=" * 80 + "\n")
        
        # Use single-seed plotting function
        plot_single_seed_comparison(
            ontology_metrics, 
            random_metrics,
            save_path=os.path.join(OUTPUT_DIR, "performance_comparison_single_seed.png"),
            seed=single_seed
        )
        
    # Also generate multi-seed plot if in multi-seed mode
    if use_multi_seed:
        plot_comparison(
            ontology_metrics, 
            random_metrics,
            save_path=os.path.join(OUTPUT_DIR, "performance_comparison.png"),
            seeds=seeds
        )

    print(f"\n{'='*80}")
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize CNN performance with multi-seed support')
    parser.add_argument('--multi-seed', action='store_true', default=True,
                        help='Use multi-seed averaging (default: True)')
    parser.add_argument('--single-seed', action='store_true',
                        help='Use single seed mode (legacy)')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS,
                        help=f'Seeds to use for multi-seed runs (default: {DEFAULT_SEEDS})')
    
    args = parser.parse_args()
    
    # Single-seed takes precedence if specified
    use_multi = not args.single_seed
    
    visualize_all(use_multi_seed=use_multi, seeds=args.seeds)