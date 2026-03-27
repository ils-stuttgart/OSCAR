"""
OSCAR CNN - Runway Detection Model
======================================================
Usage:
    # Iteration 1 - Both strategies use same initial dataset
    python OSCAR_CNN.py --start 1 --epochs 10 --seed 61
    python OSCAR_CNN.py --start 4 --epochs 10 --seed 116 --use_random_dataset
    
    # Iteration 2+ - Ontology-guided uses Rec datasets
    python OSCAR_CNN.py --start 2 --epochs 10 --seed 42
    
    # Iteration 2+ - Random uses Random datasets (CUMULATIVE)
    python OSCAR_CNN.py --start 2 --epochs 10 --seed 42 --use_random_dataset
    python OSCAR_CNN.py --start 6 --epochs 30 --seed 116 --use_random_dataset
    
    # Batch training
    python OSCAR_CNN.py --start 2 --end 13 --epochs 10 --seed 61 --use_random_dataset
"""

import os
import json
import argparse
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import tempfile

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    from tqdm.keras import TqdmCallback
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[INFO] tqdm not available. Install with: pip install tqdm")

# ============================================================================
# SEED CONFIGURATION
# ============================================================================
DEFAULT_SEED = 42

def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seeds(DEFAULT_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# MODEL ARCHITECTURE - ScenAIro CNN
# ============================================================================
def build_scenairo_cnn(input_size=(100, 100, 3)):
    model = Sequential()

    model.add(Input(shape=input_size))

    model.add(Conv2D(3, (3, 3), padding='same', strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    
    model.add(Conv2D(3, (5, 5), padding='valid', strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))  

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model
# ============================================================================
# UNIFIED DATA LOADING 
# ============================================================================

def load_images_from_directory(directory, target_size=(100, 100)):

    images = []
    labels = []
    filenames = []
    
    class_mapping = {'norunway': 0, 'runway': 1}
    
    for class_name, class_idx in class_mapping.items():
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, filename)
                try:
                    img = keras_image.load_img(img_path, target_size=target_size)
                    img_array = keras_image.img_to_array(img) / 255.0
                    images.append(img_array)
                    labels.append(class_idx)
                    filenames.append(img_path)
                except Exception as e:
                    print(f"[WARNING] Could not load {img_path}: {e}")
    
    return np.array(images), np.array(labels), filenames, len(images)
class UnifiedDataGenerator(Sequence):

    
    def __init__(self, images, labels, filenames=None, batch_size=32, shuffle=True, seed=None, augment=False):

        self.images = images
        self.labels = labels
        self.filenames_list = filenames if filenames is not None else [f"image_{i}.png" for i in range(len(images))]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed
        self.indices = np.arange(len(self.images))
        
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        if self.augment:
            batch_images = self._apply_augmentation(batch_images)
        
        return batch_images, batch_labels
    
    def _apply_augmentation(self, images):

        augmented = []
        for img in images:

            if np.random.random() > 0.5:
                img = np.fliplr(img)
            augmented.append(img)
        return np.array(augmented)
    
    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    @property
    def samples(self):
        return len(self.images)
    
    @property
    def classes(self):
        return self.labels[self.indices]
    
    @property
    def class_indices(self):
        return {'norunway': 0, 'runway': 1}
    
    @property
    def filepaths(self):
        return [self.filenames_list[i] for i in self.indices]


def load_training_data_unified(base_dir, augmentation_folders, target_size=(100, 100), 
                                strategy_name="", seed=DEFAULT_SEED):
    print(f"\n{'='*60}")
    print(f"[{strategy_name}] Loading Training Data (Unified Method)")
    print(f"{'='*60}")
    
    # Load original training images
    train_dir = os.path.join(base_dir, 'train')
    original_images, original_labels, original_filenames, original_count = load_images_from_directory(
        train_dir, target_size
    )
    print(f"  Original training images: {original_count}")
    
    # Load augmentation images (cumulative)
    aug_images_list = []
    aug_labels_list = []
    aug_filenames_list = []
    augmentation_count = 0
    
    for folder_path in augmentation_folders:
        if os.path.exists(folder_path):
            imgs, lbls, fnames, cnt = load_images_from_directory(folder_path, target_size)
            if cnt > 0:
                aug_images_list.append(imgs)
                aug_labels_list.append(lbls)
                aug_filenames_list.append(fnames)
                augmentation_count += cnt
                print(f"  + {os.path.basename(folder_path)}: {cnt} images")
        else:
            print(f"  [WARNING] Folder not found: {folder_path}")
    
    # Combine all training data
    if aug_images_list:
        all_images = np.concatenate([original_images] + aug_images_list, axis=0)
        all_labels = np.concatenate([original_labels] + aug_labels_list, axis=0)
        all_filenames = original_filenames + [f for sublist in aug_filenames_list for f in sublist]
    else:
        all_images = original_images
        all_labels = original_labels
        all_filenames = original_filenames
    
    total_train = len(all_images)
    print(f"  TOTAL training: {total_train} (Original: {original_count}, Augmentation: {augmentation_count})")
    
    train_gen = UnifiedDataGenerator(
        all_images, all_labels, all_filenames,
        batch_size=32, 
        shuffle=True, 
        seed=seed,
        augment=False  
    )

    val_images, val_labels, val_filenames, val_count = load_images_from_directory(
        os.path.join(base_dir, 'val'), target_size
    )
    test_images, test_labels, test_filenames, test_count = load_images_from_directory(
        os.path.join(base_dir, 'test'), target_size
    )
    
    val_gen = UnifiedDataGenerator(val_images, val_labels, val_filenames, batch_size=32, shuffle=False)
    test_gen = UnifiedDataGenerator(test_images, test_labels, test_filenames, batch_size=32, shuffle=False)
    
    print(f"  Validation: {val_count}, Test: {test_count}")
    print(f"{'='*60}\n")
    
    return train_gen, val_gen, test_gen, original_count, augmentation_count
# ============================================================================
# TRAINING
# ============================================================================
def train_model(model, train_generator, validation_generator, epochs=10, batch_size=64):
    """Train the model with progress bar"""
    print(f"\n{'='*60}")
    print(f"TRAINING - {epochs} epochs")
    print(f"{'='*60}\n")
    
    callbacks = []
    if TQDM_AVAILABLE:
        callbacks.append(TqdmCallback(verbose=1))
        verbose_setting = 0
    else:
        verbose_setting = 1
    
    history = model.fit(
        train_generator,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        verbose=verbose_setting,
        callbacks=callbacks if callbacks else None
    )
    
    return history
# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(model, train_generator, val_generator, test_generator):
    """Evaluate model on train/val/test sets"""
    print(f"\n{'='*60}")
    print(f"EVALUATION")
    print(f"{'='*60}\n")
    
    train_loss, train_accuracy = model.evaluate(train_generator)
    val_loss, val_accuracy = model.evaluate(val_generator)
    test_loss, test_accuracy = model.evaluate(test_generator)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Accuracy:   {val_accuracy:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    
    return {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }

def get_misclassified_images(model, generator, split_name='test'):
    generator.reset()
    
    print(f"\nPredicting on {split_name} set...")
    pred_probs = model.predict(generator, verbose=0)
    pred_classes = (pred_probs > 0.5).astype(int).ravel()
    
    true_classes = generator.classes
    file_paths = generator.filepaths
    class_labels = list(generator.class_indices.keys())
    
    misclassified_idx = np.where(pred_classes != true_classes)[0]
    
    print(f"\n[{split_name.upper()}] Total: {len(true_classes)}, Misclassified: {len(misclassified_idx)}")
    
    misclassified = []
    for idx in misclassified_idx:
        true_label = class_labels[true_classes[idx]]
        pred_label = class_labels[pred_classes[idx]]
        confidence = pred_probs[idx][0]
        
        filename = os.path.basename(file_paths[idx])
        
        misclassified.append({
            'filename': filename,
            'ground_truth': true_label,
            'predicted': pred_label,
            'confidence': float(confidence),
            'split': split_name,
            'index': int(idx)  # Add index for visualization
        })
        
        if len(misclassified) <= 10:  # Only print first 10
            print(f"  {filename} → True: {true_label}, Predicted: {pred_label} (conf: {confidence:.4f})")
    
    if len(misclassified) > 10:
        print(f"  ... and {len(misclassified) - 10} more")
    
    return misclassified

def visualize_misclassified_images(generator, misclassified_list, output_dir, output_filename=None, max_images=20, seed=None):
    if not misclassified_list:
        print("No misclassifications to visualize")
        return
    
    misclassified_list = misclassified_list[:max_images]
    num_images = len(misclassified_list)

    cols = 8
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows))  # Reduced height per row
    
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    generator.reset()
    
    for i, misclass in enumerate(misclassified_list):
        idx = misclass['index']
        
        # Load original image directly from file for maximum resolution
        img = None
        try:
            img_path = generator.filepaths[idx]
            # Load original resolution image without resizing
            img = keras_image.load_img(img_path)
            img_array = keras_image.img_to_array(img)
            # Normalize to 0-1 range for display
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            img = img_array
        except Exception as e:
            # Fallback to generator image if file loading fails
            img = generator.images[idx]
        
        # Plot image
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        
        # Add text above image on separate lines
        true_label = misclass['ground_truth']
        pred_label = misclass['predicted']
        conf = misclass['confidence']
        #filename = os.path.basename(misclass['filename'])
        
        # Color for wrong predictions
        color = 'blue'

        title = f"GT: {true_label}\nPred: {pred_label}\nConf: {conf:.1%}"
        ax.set_title(title, fontsize=12, color=color, fontweight='bold', pad=8)
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    split_name = misclassified_list[0]['split'] if misclassified_list else 'unknown'
    seed_text = f" (Seed {seed})" if seed else ""
    plt.suptitle(f"Misclassified Images - {split_name.upper()} Set{seed_text}", fontsize=28, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.85)  # Reduced hspace from 0.4 to 0.15
    
    # Save with very high DPI for maximum quality
    os.makedirs(output_dir, exist_ok=True)
    if output_filename:
        output_path = os.path.join(output_dir, f"{output_filename}_Misclassified.png")
    else:
        output_path = os.path.join(output_dir, f"Misclassified_{split_name}.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
    print(f"\nMisclassified visualization saved: {output_path}")
    plt.close()

def print_confusion_matrix(model, test_generator, output_filename, output_dir=None):
    """Print confusion matrix and classification report"""
    test_generator.reset()
    
    pred_probs = model.predict(test_generator, verbose=0)
    pred_classes = (pred_probs > 0.5).astype(int).ravel()
    true_classes = test_generator.classes
    
    class_labels = list(test_generator.class_indices.keys())
    
    cm = confusion_matrix(true_classes, pred_classes)
    
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}\n")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Runway Detection')
    
    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "visualization_output")
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, f"{output_filename}_ConfusionMatrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    plt.close()
    
    report = classification_report(true_classes, pred_classes, target_names=class_labels)
    print("\nCLASSIFICATION REPORT:")
    print(report)
    
    return cm, report
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_model_path(strategy, iteration, seed_suffix, script_dir, seed_num):
    # New structure: OSCAR_Experiments/seed_XX/Strategy/models/
    model_dir = os.path.join(script_dir, "OSCAR_Experiments", f"seed_{seed_num}", strategy, "models")
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"iteration_{iteration}.h5")

def verify_training_sizes(onto_count, random_count, iteration):
    if onto_count != random_count:
        print(f"\n WARNING: Training size mismatch at iteration {iteration}!")
        print(f"   This may affect fairness of comparison.")
        print(f"   Ontology: {onto_count}, Random: {random_count}")
        return False
    return True

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='OSCAR CNN for Runway Detection (Fair Comparison)')
    
    # Data arguments
    parser.add_argument('--json_dir', type=str, 
                        default=os.path.join(SCRIPT_DIR, "Ontology_Input_Individuals", "input_image"),
                        help='Base directory for train/val/test data')
    
    # Training arguments
    parser.add_argument('--start', type=int, default=1, help='Start iteration number')
    parser.add_argument('--end', type=int, default=None, help='End iteration number')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--target_size', type=int, default=100, help='Image size')
    parser.add_argument('--use_random_dataset', action='store_true', 
                        help='Use random augmented dataset instead of ontology-guided')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Fine-tune from previous iteration (default is rebuild from scratch)')
    
    # Fine-tuning arguments
    parser.add_argument('--finetune_lr', type=float, default=0.0001, 
                        help='Learning rate for fine-tuning')
    
    # Seed arguments
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    parser.add_argument('--seed_suffix', type=str, default='', help='Seed suffix for file naming')
    
    args = parser.parse_args()
    
    # Set seed
    set_seeds(args.seed)
    
    # Auto-generate seed_suffix
    if not args.seed_suffix:
        args.seed_suffix = f"_seed{args.seed}"
    
    # Default to rebuild-from-scratch unless fine-tune is explicitly enabled
    rebuild_from_scratch = not args.fine_tune

    # Determine strategy
    strategy = "Random" if args.use_random_dataset else "Ontology"
    
    # Determine iteration range
    start_iter = args.start
    end_iter = args.end if args.end else args.start
    
    print(f"\n{'='*70}")
    print(f"OSCAR CNN - RUNWAY DETECTION ({strategy.upper()} AUGMENTATION)")
    print(f"{'='*70}")
    if start_iter == end_iter:
        print(f"Iteration: {start_iter}, Epochs: {args.epochs}, Seed: {args.seed}")
    else:
        print(f"Batch Training: Iterations {start_iter} to {end_iter}")
    print(f"Strategy: {strategy}")
    print(f"Output suffix: {args.seed_suffix}")
    print(f"{'='*70}\n")
    
    # Loop through iterations
    for current_iteration in range(start_iter, end_iter + 1):
        print(f"\n{'#'*70}")
        print(f"# TRAINING ITERATION {current_iteration} ({strategy})")
        print(f"{'#'*70}\n")
        
        target_size = (args.target_size, args.target_size)
        
        # =====================================================================
        # BUILD AUGMENTATION FOLDER LIST (CUMULATIVE FOR BOTH STRATEGIES)
        # =====================================================================
        if current_iteration == 1:
            augmentation_folders = []  # No augmentation for iteration 1
        else:
            if args.use_random_dataset:
                # CUMULATIVE random datasets: Random1, Random2, ..., Random(N-1)
                datasets_base = os.path.join(SCRIPT_DIR, "OSCAR_Experiments", f"seed_{args.seed}", "Random", "datasets")
                augmentation_folders = [
                    os.path.join(datasets_base, f"Random{i}_seed{args.seed}")
                    for i in range(1, current_iteration)
                ]
            else:
                # CUMULATIVE ontology datasets: Rec1, Rec2, ..., Rec(N-1)
                datasets_base = os.path.join(SCRIPT_DIR, "OSCAR_Experiments", f"seed_{args.seed}", "Ontology", "datasets")
                augmentation_folders = [
                    os.path.join(datasets_base, f"Rec{i}-scenairo-seed{args.seed}")
                    for i in range(1, current_iteration)
                ]
        
        # =====================================================================
        # DATA LOADING (UNIFIED FOR FAIR COMPARISON)
        # =====================================================================
        train_gen, val_gen, test_gen, original_count, aug_count = load_training_data_unified(
            args.json_dir,
            augmentation_folders,
            target_size=target_size,
            strategy_name=strategy,
            seed=args.seed
        )
        
        total_train = train_gen.samples
        
        # =====================================================================
        # MODEL CREATION/LOADING
        # =====================================================================
        if rebuild_from_scratch:
            print("[REBUILD] Building model from scratch (fine-tuning disabled)")
            model = build_scenairo_cnn(input_size=(args.target_size, args.target_size, 3))
        elif current_iteration > 1:
            # Load previous model from SAME strategy chain
            prev_model_path = get_model_path(strategy, current_iteration - 1, args.seed_suffix, SCRIPT_DIR, args.seed)
            
            if os.path.exists(prev_model_path):
                print(f"[FINETUNE] Loading previous model: {prev_model_path}")
                try:
                    model = load_model(prev_model_path)
                except (TypeError, ValueError) as e:
                    if 'quantization_config' in str(e):
                        print(f"[WARNING] Keras version mismatch. Loading weights only...")
                        model = build_scenairo_cnn(input_size=(args.target_size, args.target_size, 3))
                        model.load_weights(prev_model_path)
                    else:
                        raise
                
                # Recompile with lower learning rate
                model.compile(
                    loss='binary_crossentropy',
                    optimizer=Adam(learning_rate=args.finetune_lr),
                    metrics=['accuracy']
                )
                print(f"[FINETUNE] Recompiled with learning rate: {args.finetune_lr}")
            else:
                print(f"[WARNING] Previous model not found: {prev_model_path}")
                print(f"[WARNING] Building new model from scratch")
                model = build_scenairo_cnn(input_size=(args.target_size, args.target_size, 3))
        else:
            # Build new model from scratch (iteration 1)
            print(f"[ITERATION 1] Building new ScenAIro CNN model")
            model = build_scenairo_cnn(input_size=(args.target_size, args.target_size, 3))
        
        print("\nModel Architecture:")
        model.summary()
        
        # =====================================================================
        # TRAINING
        # =====================================================================
        history = train_model(model, train_gen, val_gen, epochs=args.epochs, batch_size=args.batch_size)
        
        # =====================================================================
        # EVALUATION
        # =====================================================================
        metrics = evaluate_model(model, train_gen, val_gen, test_gen)
        
        # Get misclassifications
        test_failures = get_misclassified_images(model, test_gen, 'test')
        train_failures = get_misclassified_images(model, train_gen, 'train')
        val_failures = get_misclassified_images(model, val_gen, 'val')
        
        # =====================================================================
        # DETERMINE OUTPUT PATHS (SEPARATE FOR EACH STRATEGY)
        # =====================================================================
        # New structure: OSCAR_Experiments/seed_XX/Strategy/results/ and visualizations/
        strategy_base = os.path.join(SCRIPT_DIR, "OSCAR_Experiments", f"seed_{args.seed}", strategy)
        result_output_dir = os.path.join(strategy_base, "results")
        viz_output_dir = os.path.join(strategy_base, "visualizations")
        
        os.makedirs(viz_output_dir, exist_ok=True)
        os.makedirs(result_output_dir, exist_ok=True)
        
        # Determine output filename
        if current_iteration == 1:
            output_filename = f"CNN_Base{args.seed_suffix}"
            version_name = "CNN_Base"
        else:
            if args.use_random_dataset:
                output_filename = f"CNN_with_Random{current_iteration-1}{args.seed_suffix}"
                version_name = f"CNN_with_Random{current_iteration-1}"
            else:
                output_filename = f"CNN_with_Rec{current_iteration-1}{args.seed_suffix}"
                version_name = f"CNN_with_Rec{current_iteration-1}"
        
        # =====================================================================
        # VISUALIZE MISCLASSIFICATIONS (TEST SET ONLY)
        # =====================================================================
        print(f"\n{'='*60}")
        print("CREATING MISCLASSIFICATION VISUALIZATIONS")
        print(f"{'='*60}")
        
        if test_failures:
            visualize_misclassified_images(test_gen, test_failures, viz_output_dir, 
                                          output_filename=output_filename, max_images=20, seed=args.seed)
        
        # Print confusion matrix
        cm, report = print_confusion_matrix(model, test_gen, output_filename, output_dir=viz_output_dir)
        
        # =====================================================================
        # SAVE RESULTS
        # =====================================================================
        result_data = {
            "cnn_version": version_name,
            "model_name": f"CNN_RunwayDetector_{strategy}_{version_name}",
            "strategy": strategy,
            "description": f"CNN iteration {current_iteration} - {strategy} augmentation",
            "rec_version": None if current_iteration == 1 else f"Rec{current_iteration-1}" if not args.use_random_dataset else f"Random{current_iteration-1}",
            "dataset_split": {
                "training": {
                    "original": original_count,
                    "augmentation": aug_count,
                    "total": total_train
                },
                "validation": val_gen.samples,
                "testing": test_gen.samples,
                "total": total_train + val_gen.samples + test_gen.samples
            },
            "augmentation_folders": [os.path.basename(f) for f in augmentation_folders],
            "performance_metrics": {
                "training": {
                    "accuracy": round(metrics['train_accuracy'], 4),
                    "loss": round(metrics['train_loss'], 4),
                    "total": total_train
                },
                "validation": {
                    "accuracy": round(metrics['val_accuracy'], 4),
                    "loss": round(metrics['val_loss'], 4),
                    "total": val_gen.samples
                },
                "testing": {
                    "accuracy": round(metrics['test_accuracy'], 4),
                    "loss": round(metrics['test_loss'], 4),
                    "total": test_gen.samples
                }
            },
            "misclassifications": {
                "training": {
                    "original_dataset": train_failures
                },
                "validation": val_failures,
                "testing": test_failures
            }
        }
        
        # Save result JSON
        result_filename = f"{output_filename}.json"
        result_path = os.path.join(result_output_dir, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {result_path}")
        
        # Save model (with strategy in name)
        model_path = get_model_path(strategy, current_iteration, args.seed_suffix, SCRIPT_DIR, args.seed)
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        print(f"\n{'='*70}")
        print(f"ITERATION {current_iteration} ({strategy}) COMPLETE")
        print(f"  Training Size: {total_train} (Original: {original_count}, Aug: {aug_count})")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"{'='*70}\n")
    
    # All iterations complete
    print(f"\n{'#'*70}")
    print(f"# ALL ITERATIONS COMPLETE ({strategy} STRATEGY)")
    if start_iter == end_iter:
        print(f"# Trained iteration: {start_iter}")
    else:
        print(f"# Trained iterations: {start_iter} to {end_iter}")
    print(f"{'#'*70}\n")
if __name__ == "__main__":
    main()
