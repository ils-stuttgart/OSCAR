"""
OSCAR Image Renamer - Remove '_from_json' Suffix
=================================================
After manual image generation in ScenAIro/MSFS, external tools may add '_from_json.png' 
suffix to generated images. This script renames them to match JSON file names exactly.

Example:
  2026-02-05_132905_EDDS_norunway_night_rec1_001_from_json.png
  → 2026-02-05_132905_EDDS_norunway_night_rec1_001.png

Usage:
  # Rename all images in specific iteration
  python OSCAR_Rename_Images.py --iteration 1 --seed 42
  
  # Rename all images across all iterations and seeds
  python OSCAR_Rename_Images.py --all
  
  # Dry run (preview changes without renaming)
  python OSCAR_Rename_Images.py --iteration 1 --seed 42 --dry-run
"""
import os
import argparse
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(SCRIPT_DIR, "OSCAR_Experiments")
ONTOLOGY_INPUT_DIR = os.path.join(SCRIPT_DIR, "Ontology_Input_Individuals", "input_image")

def rename_images_in_directory(directory, dry_run=False, verbose=True):
    """
    Rename all PNG files with '_from_json' suffix in a directory.
    
    Args:
        directory: Path to directory to process
        dry_run: If True, only show what would be renamed without actually renaming
        verbose: If True, print each rename operation
    
    Returns:
        Tuple of (renamed_count, skipped_count)
    """
    if not os.path.exists(directory):
        return (0, 0)
    
    renamed_count = 0
    skipped_count = 0
    
    # Find all PNG files with '_from_json' in the name
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.png') and '_from_json' in filename:
                old_path = os.path.join(root, filename)
                
                # Generate new filename by removing '_from_json'
                new_filename = filename.replace('_from_json', '')
                new_path = os.path.join(root, new_filename)
                
                # Check if target already exists
                if os.path.exists(new_path):
                    if verbose:
                        print(f"  ⚠ SKIP (target exists): {filename}")
                    skipped_count += 1
                    continue
                
                # Rename the file
                if dry_run:
                    print(f"  [DRY-RUN] {filename} → {new_filename}")
                    renamed_count += 1
                else:
                    os.rename(old_path, new_path)
                    if verbose:
                        print(f"  ✓ RENAMED: {filename} → {new_filename}")
                    renamed_count += 1
    
    return (renamed_count, skipped_count)

def process_iteration(iteration, seed, dry_run=False):
    """
    Rename images for a specific iteration and seed.
    
    Args:
        iteration: Iteration number (e.g., 1, 2, 3)
        seed: Seed number (e.g., 42, 61, 116)
        dry_run: If True, preview changes without renaming
    """
    seed_dir = os.path.join(EXPERIMENTS_DIR, f"seed_{seed}")
    
    # Process Ontology datasets
    ontology_dataset_dir = os.path.join(
        seed_dir, "Ontology", "datasets", f"Rec{iteration}-scenairo-seed{seed}"
    )
    
    # Process Random datasets
    random_dataset_dir = os.path.join(
        seed_dir, "Random", "datasets", f"Random{iteration}_seed{seed}"
    )
    
    total_renamed = 0
    total_skipped = 0
    
    print(f"\n{'='*80}")
    print(f"Processing Iteration {iteration}, Seed {seed}")
    print(f"{'='*80}")
    
    # Rename in Ontology datasets
    if os.path.exists(ontology_dataset_dir):
        print(f"\nOntology datasets: {ontology_dataset_dir}")
        renamed, skipped = rename_images_in_directory(ontology_dataset_dir, dry_run)
        total_renamed += renamed
        total_skipped += skipped
        print(f"  → Renamed: {renamed}, Skipped: {skipped}")
    else:
        print(f"\n⚠ Ontology dataset not found: {ontology_dataset_dir}")
    
    # Rename in Random datasets
    if os.path.exists(random_dataset_dir):
        print(f"\nRandom datasets: {random_dataset_dir}")
        renamed, skipped = rename_images_in_directory(random_dataset_dir, dry_run)
        total_renamed += renamed
        total_skipped += skipped
        print(f"  → Renamed: {renamed}, Skipped: {skipped}")
    else:
        print(f"\n⚠ Random dataset not found: {random_dataset_dir}")
    
    return (total_renamed, total_skipped)

def process_all(dry_run=False):
    """
    Rename images in all iterations and seeds.
    
    Also processes the input_image directory used for initial training.
    """
    print(f"\n{'#'*80}")
    print(f"# RENAMING ALL IMAGES (All Iterations, All Seeds)")
    print(f"{'#'*80}")
    
    total_renamed = 0
    total_skipped = 0
    
    # Process initial training images
    if os.path.exists(ONTOLOGY_INPUT_DIR):
        print(f"\n{'='*80}")
        print(f"Processing Initial Training Images")
        print(f"{'='*80}")
        print(f"\nDirectory: {ONTOLOGY_INPUT_DIR}")
        renamed, skipped = rename_images_in_directory(ONTOLOGY_INPUT_DIR, dry_run)
        total_renamed += renamed
        total_skipped += skipped
        print(f"  → Renamed: {renamed}, Skipped: {skipped}")
    
    # Process all experiment directories
    if os.path.exists(EXPERIMENTS_DIR):
        for seed_folder in os.listdir(EXPERIMENTS_DIR):
            if not seed_folder.startswith("seed_"):
                continue
            
            seed = seed_folder.split("_")[1]
            seed_path = os.path.join(EXPERIMENTS_DIR, seed_folder)
            
            # Process both Ontology and Random
            for strategy in ["Ontology", "Random"]:
                datasets_dir = os.path.join(seed_path, strategy, "datasets")
                if not os.path.exists(datasets_dir):
                    continue
                
                # Process each dataset folder
                for dataset_folder in os.listdir(datasets_dir):
                    dataset_path = os.path.join(datasets_dir, dataset_folder)
                    if not os.path.isdir(dataset_path):
                        continue
                    
                    print(f"\n{'='*80}")
                    print(f"Processing: Seed {seed}, {strategy}, {dataset_folder}")
                    print(f"{'='*80}")
                    print(f"\nDirectory: {dataset_path}")
                    
                    renamed, skipped = rename_images_in_directory(dataset_path, dry_run)
                    total_renamed += renamed
                    total_skipped += skipped
                    print(f"  → Renamed: {renamed}, Skipped: {skipped}")
    
    return (total_renamed, total_skipped)

def main():
    parser = argparse.ArgumentParser(
        description='Rename PNG files by removing "_from_json" suffix to match JSON names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rename images for iteration 1, seed 42
  python OSCAR_Rename_Images.py --iteration 1 --seed 42
  
  # Preview changes without renaming
  python OSCAR_Rename_Images.py --iteration 1 --seed 42 --dry-run
  
  # Rename all images across all iterations and seeds
  python OSCAR_Rename_Images.py --all
  
  # Rename images for multiple seeds in iteration 1
  python OSCAR_Rename_Images.py --iteration 1 --seeds 42 61 116
        """
    )
    
    parser.add_argument('--iteration', type=int, help='Iteration number to process')
    parser.add_argument('--seed', type=int, help='Seed number to process')
    parser.add_argument('--seeds', nargs='*', help='Multiple seeds (space-separated)')
    parser.add_argument('--all', action='store_true', help='Process all iterations and seeds')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without renaming')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("\n" + "="*80)
        print("DRY RUN MODE - No files will be renamed")
        print("="*80)
    
    total_renamed = 0
    total_skipped = 0
    
    if args.all:
        # Process all iterations and seeds
        renamed, skipped = process_all(dry_run=args.dry_run)
        total_renamed += renamed
        total_skipped += skipped
    elif args.iteration is not None:
        # Process specific iteration
        if args.seeds:
            # Multiple seeds
            seeds = [int(s.strip()) for s in args.seeds[0].split(',')] if len(args.seeds) == 1 and ',' in args.seeds[0] else [int(s) for s in args.seeds]
            for seed in seeds:
                renamed, skipped = process_iteration(args.iteration, seed, dry_run=args.dry_run)
                total_renamed += renamed
                total_skipped += skipped
        elif args.seed is not None:
            # Single seed
            renamed, skipped = process_iteration(args.iteration, args.seed, dry_run=args.dry_run)
            total_renamed += renamed
            total_skipped += skipped
        else:
            print("Error: --seed or --seeds required when using --iteration")
            return
    else:
        print("Error: Must specify either --all or --iteration with --seed/--seeds")
        parser.print_help()
        return
    
    # Summary
    print(f"\n{'#'*80}")
    print(f"# SUMMARY")
    print(f"{'#'*80}")
    print(f"  Total renamed: {total_renamed}")
    print(f"  Total skipped: {total_skipped}")
    if args.dry_run:
        print(f"\n  ℹ This was a DRY RUN - no files were actually renamed")
        print(f"  ℹ Run without --dry-run to apply changes")
    print(f"{'#'*80}\n")

if __name__ == "__main__":
    main()
