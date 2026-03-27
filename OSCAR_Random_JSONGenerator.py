"""
OSCAR Random JSON Generator (Optimized)
========================================
Directly generates random ScenAIro JSON files for manual image generation.

Usage:
    python OSCAR_Random_JSONGenerator.py --iteration 4 --seed 116
    python OSCAR_Random_JSONGenerator.py --iteration 1 --end 11 --seed 61
    
Workflow:
    1. Generate random JSON files (this script)
    2. Manually generate images using ScenAIro+MSFS
    3. Train CNN with generated datasets
"""

import os
import json
import random
import argparse
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE_DIR = os.path.join(SCRIPT_DIR, "Ontology_Input_Individuals", "input_image")
# New structure: OSCAR_Experiments/seed_XX/Random/datasets/
OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "OSCAR_Experiments")

# Airport configurations (consistent with OSCAR_DatasetGenerator.py)
AIRPORTS = {
    'EDDS': {'name': 'Stuttgart', 'runway_name': '25', 'runway_width': 45.0, 'runway_length': 3045.0,
             'runway_heading': 74.025597, 'center': {'latitude': 48.690248, 'longitude': 9.223924, 'altitude': 384.21701},
             'start_height': 388.9248, 'end_height': 388.949184, 'aircraft_pos': [48.69519465345375, 9.250042464967704, 484.21701]},
    'EDDV': {'name': 'Hannover', 'runway_name': '9L', 'runway_width': 45.0, 'runway_length': 3198.0,
             'runway_heading': 92.56987, 'center': {'latitude': 52.467599, 'longitude': 9.676213, 'altitude': 52.781002},
             'start_height': 52.781002, 'end_height': 51.781002, 'aircraft_pos': [52.466789466703254, 9.705611876930334, 152.781002]},
    'EDNY': {'name': 'Friedrichshafen', 'runway_name': '24', 'runway_width': 45.0, 'runway_length': 2352.0,
             'runway_heading': 60.119431, 'center': {'latitude': 47.671325, 'longitude': 9.511504, 'altitude': 411.937012},
             'start_height': 411.937012, 'end_height': 411.937012, 'aircraft_pos': [47.68028428294408, 9.53459949691774, 511.937012]},
    'EDSB': {'name': 'Karlsruhe / Baden-Baden', 'runway_name': '21', 'runway_width': 45.0, 'runway_length': 2998.0,
             'runway_heading': 211.724258, 'center': {'latitude': 48.779356, 'longitude': 8.080506, 'altitude': 123.596008},
             'start_height': 123.596008, 'end_height': 123.596008, 'aircraft_pos': [48.764057487566646, 8.066200919467537, 223.59600799999998]},
    'ELLX': {'name': 'Luxembourg', 'runway_name': '24', 'runway_width': 45.0, 'runway_length': 3998.0,
             'runway_heading': 60.199310, 'center': {'latitude': 49.626451, 'longitude': 6.211552, 'altitude': 361.492004},
             'start_height': 367.8936, 'end_height': 354.1776, 'aircraft_pos': [49.6353853125247, 6.2355775077074, 461.492004]},
    'ENBR': {'name': 'Flesland, Bergen', 'runway_name': '35', 'runway_width': 45.0, 'runway_length': 2981.0,
             'runway_heading': 350.534576, 'center': {'latitude': 60.293437, 'longitude': 5.218154, 'altitude': 47.701004},
             'start_height': 47.701004, 'end_height': 52.701004, 'aircraft_pos': [60.31114298706362, 5.2122036872189526, 147.701004]},
    'KLAX': {'name': 'Los Angeles International Airport', 'runway_name': 'R25', 'runway_width': 46.0, 'runway_length': 3939.0,
             'runway_heading': 82.960899, 'center': {'latitude': 33.937723, 'longitude': -118.400929, 'altitude': 34.022},
             'start_height': 34.022, 'end_height': 34.022, 'aircraft_pos': [33.93993074319327, -118.3794586855742, 134.022]}
}

AIRPORT_ANNOTATIONS = {
    'EDDS': {'bbox': [1200, 380, 160, 294], 'segmentation': [[[1269, 674, 1291, 674, 1360, 380, 1200, 380]]], 'area': 47040},
    'EDDV': {'bbox': [1184, 295, 192, 377], 'segmentation': [[[1269, 672, 1291, 672, 1376, 295, 1184, 295]]], 'area': 72384},
    'EDNY': {'bbox': [1234, 513, 92, 153], 'segmentation': [[[1268, 666, 1292, 666, 1326, 513, 1234, 513]]], 'area': 14076},
    'EDSB': {'bbox': [1204, 380, 152, 291], 'segmentation': [[[1269, 671, 1291, 671, 1356, 380, 1204, 380]]], 'area': 44232},
    'ELLX': {'bbox': [-36972, -159093, 76674, 159767], 'segmentation': [[[1270, 674, 1290, 674, 39702, -159093, -36972, -158432]]], 'area': 12249974958},
    'ENBR': {'bbox': [1205, 386, 150, 288], 'segmentation': [[[1269, 674, 1291, 674, 1355, 386, 1205, 386]]], 'area': 43200},
    'KLAX': {'bbox': [-4, -4864, 2568, 5541], 'segmentation': [[[1270, 677, 1290, 677, 2564, -4864, -4, -4863]]], 'area': 14229288}
}
# Adaptive limit configuration (mirrors ontology approach)
CONFIG = {
    'base_percentage': 0.08,  # 8% of cumulative training size
    'min_images': 5,
    'max_images': 30
}
# ============================================================================
# Helper Functions
# ============================================================================
def get_initial_training_size():
    """Auto-detect initial training size from input_image/train folder."""
    train_dir = os.path.join(INPUT_IMAGE_DIR, "train")
    if not os.path.exists(train_dir):
        return 120  # Default
    
    count = sum(len([f for f in os.listdir(os.path.join(train_dir, d)) if f.endswith('.json')])
                for d in ['norunway', 'runway'] if os.path.exists(os.path.join(train_dir, d)))
    return count if count > 0 else 120

def calc_image_limit(iteration):
    """Calculate adaptive image limit (same formula as ontology)."""
    cumulative = get_initial_training_size()
    for _ in range(1, iteration):
        cumulative += max(CONFIG['min_images'], min(CONFIG['max_images'], round(cumulative * CONFIG['base_percentage'])))
    
    limit = max(CONFIG['min_images'], min(CONFIG['max_images'], round(cumulative * CONFIG['base_percentage'])))
    print(f"[ADAPTIVE] Iter {iteration}: cumulative={cumulative} × 8% = {limit} images")
    return limit

def generate_scenairo_json(airport_code, runway_type, time_category, index, seed):
    """Generate single ScenAIro JSON (consistent with OSCAR_DatasetGenerator.py)."""
    airport = AIRPORTS[airport_code]
    
    # Time of day
    hours = random.randint(10, 15) if time_category == 'daytime' else random.randint(0, 5)
    minutes = random.randint(0, 59)
    
    # Filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_label = 'day' if time_category == 'daytime' else 'night'
    filename = f"{timestamp}_Random_seed{seed}_{airport_code}_{runway_type}_{time_label}_{index:05d}.png"
    
    # Category
    category_name = "runway" if runway_type == 'runway' else "no_runway"
    
    # Build JSON
    data = {
        "images": [{"file_name": filename, "id": filename, "width": 2560, "height": 1440}],
        "annotations": [],
        "categories": [{"id": 1, "name": category_name, "supercategory": "infrastructure"}],
        "runway_data": {
            "name": airport['name'], "icao_code": airport_code, "runway_name": airport['runway_name'],
            "runway_width": airport['runway_width'], "runway_length": airport['runway_length'],
            "runway_heading": airport['runway_heading'],
            "runway_center": airport['center'],
            "start_height": airport['start_height'], "end_height": airport['end_height']
        },
        "landing_approach_cone": {
            "apex": [1372.5, 0.0, 0.0], "lateral_angle_left": -4.0, "lateral_angle_right": 4.0,
            "vertical_min_angle": 2.0, "vertical_max_angle": 3.8, "max_distance": 3000.0, "number_of_points": 10
        },
        "position_of_aircraft": airport['aircraft_pos'],
        "distance_aircraft_2_runway": {"ground_distance_in_meters": 1372.5, "altitude_difference_in_meters": 0.0},
        "aircraft_orientation": {"pitch": 0, "yaw": 0.0 if runway_type == 'runway' else 90.0, "roll": 0},
        "daytime": {"hours": hours, "minutes": minutes}
    }
    
    # Add annotation for runway
    if runway_type == 'runway':
        anno = AIRPORT_ANNOTATIONS[airport_code]
        data["annotations"].append({
            "id": 0, "image_id": filename, "category_id": 1,
            "bbox": anno['bbox'], "segmentation": anno['segmentation'], "area": anno['area'], "iscrowd": 0
        })
    
    return data, filename.replace('.png', '.json')

# ============================================================================
# Main Generation
# ============================================================================
def generate_random_jsons(iteration, seed):
    """Generate random JSON files with truly random (potentially poor) distribution."""
    random.seed(seed)
    
    num_images = calc_image_limit(iteration)
    # New structure: OSCAR_Experiments/seed_XX/Random/datasets/RandomN_seedXX/
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"seed_{seed}", "Random", "datasets", f"Random{iteration}_seed{seed}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create runway/norunway subfolders to match CNN training structure
    for subfolder in ['runway', 'norunway']:
        os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"RANDOM JSON GENERATOR - Iteration {iteration}, Seed {seed}")
    print(f"Generating {num_images} JSON files (truly random, no balancing)")
    print(f"{'='*70}\n")
    
    # Lists for random selection
    airports = list(AIRPORTS.keys())
    runway_types = ['runway', 'norunway']
    time_categories = ['daytime', 'nighttime']
    
    # TRUE RANDOM: Use random.choices with replacement (allows poor distribution)
    # Each image gets completely random selection - no forced balancing
    random_airports = random.choices(airports, k=num_images)
    random_runways = random.choices(runway_types, k=num_images)
    random_times = random.choices(time_categories, k=num_images)
    
    generated = []
    for i in range(num_images):
        airport = random_airports[i]
        runway = random_runways[i]
        time_cat = random_times[i]
        
        data, json_filename = generate_scenairo_json(airport, runway, time_cat, i + 1, seed)
        
        # Save JSON in runway/norunway subfolder
        json_path = os.path.join(output_dir, runway, json_filename)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        generated.append({'file': f"{runway}/{json_filename}", 'airport': airport, 'runway': runway, 'time': time_cat})
    
    # Summary
    summary = {
        "iteration": iteration, "seed": seed, "total_images": num_images,
        "generation_timestamp": datetime.now().isoformat(),
        "generation_method": "truly_random_with_replacement",
        "note": "No balancing applied - expect potentially poor distribution (realistic random baseline)",
        "distribution": {
            "airports": {k: sum(1 for g in generated if g['airport'] == k) for k in airports},
            "runway_types": {k: sum(1 for g in generated if g['runway'] == k) for k in runway_types},
            "time_categories": {k: sum(1 for g in generated if g['time'] == k) for k in time_categories}
        }
    }
    
    with open(os.path.join(output_dir, "generation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Generated {num_images} JSON files (truly random)")
    print(f"✓ Output: {output_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random ScenAIro JSONs')
    parser.add_argument('--iteration', type=int, help='Single iteration number')
    parser.add_argument('--start', type=int, help='Start iteration (inclusive)')
    parser.add_argument('--end', type=int, help='End iteration (inclusive)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    
    args = parser.parse_args()
    
    # Determine iteration range (support flexible combinations)
    if args.start is not None and args.end is not None:
        # --start X --end Y: range mode
        iterations = range(args.start, args.end + 1)
    elif args.iteration is not None and args.end is not None:
        # --iteration X --end Y: treat iteration as start
        iterations = range(args.iteration, args.end + 1)
        print(f"[INFO] Treating --iteration {args.iteration} as start, generating {args.iteration} to {args.end}")
    elif args.iteration is not None:
        # --iteration X: single iteration
        iterations = [args.iteration]
    else:
        parser.error('Specify: --iteration N (single) OR --start N --end M (range) OR --iteration N --end M (range)')
    
    # Generate for all iterations
    for iteration in iterations:
        generate_random_jsons(iteration, args.seed)
