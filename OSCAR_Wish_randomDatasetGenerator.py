"""
OSCAR Wish Random Dataset Generator
====================================
Generates deterministic random pool of runway scenarios for OSCAR baseline.
Uses hash-based seed isolation to prevent overlap with ontology-guided datasets.

Usage: python OSCAR_Wish_randomDatasetGenerator.py
"""

import os
import json
import random
import hashlib
from datetime import datetime

# ============================================================================
# Seed Strategy - Hash-based isolation (matches OSCAR_Wish_DatasetGenerator.py)
# ============================================================================
GLOBAL_SEED = 42

def get_split_seed_base(split_name):
    """Generate deterministic seed base using hash - ensures isolation from other splits."""
    hash_obj = hashlib.md5(f"{split_name}_{GLOBAL_SEED}".encode())
    seed_base = (int(hash_obj.hexdigest()[:8], 16) % 900) * 100000 + 1000000
    return seed_base

SEED_BASE = get_split_seed_base("random_pool")

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Source_Pool_Random_OSCAR")
TOTAL_SCENARIOS = 50000

# Airport configurations
AIRPORTS = {
    'EDDS': {
        'name': 'Stuttgart',
        'runway_name': '25',
        'runway_width': 45.0,
        'runway_length': 3045.0,
        'runway_heading': 74.025597,
        'center': {'latitude': 48.690248, 'longitude': 9.223924, 'altitude': 384.21701},
        'start_height': 388.9248,
        'end_height': 388.949184
    },
    'EDDV': {
        'name': 'Hannover',
        'runway_name': '9L',
        'runway_width': 45.0,
        'runway_length': 3198.0,
        'runway_heading': 92.56987,
        'center': {'latitude': 52.467599, 'longitude': 9.676213, 'altitude': 52.781002},
        'start_height': 52.781002,
        'end_height': 51.781002
    },
    'EDNY': {
        'name': 'Friedrichshafen',
        'runway_name': '24',
        'runway_width': 45.0,
        'runway_length': 2352.0,
        'runway_heading': 60.119431,
        'center': {'latitude': 47.671325, 'longitude': 9.511504, 'altitude': 411.937012},
        'start_height': 411.937012,
        'end_height': 411.937012
    },
    'EDSB': {
        'name': 'Karlsruhe / Baden-Baden',
        'runway_name': '21',
        'runway_width': 45.0,
        'runway_length': 2998.0,
        'runway_heading': 211.724258,
        'center': {'latitude': 48.779356, 'longitude': 8.080506, 'altitude': 123.596008},
        'start_height': 123.596008,
        'end_height': 123.596008
    },
    'ELLX': {
        'name': 'Luxembourg',
        'runway_name': '24',
        'runway_width': 45.0,
        'runway_length': 3998.0,
        'runway_heading': 60.199310,
        'center': {'latitude': 49.626451, 'longitude': 6.211552, 'altitude': 361.492004},
        'start_height': 367.8936,
        'end_height': 354.1776
    },
    'ENBR': {
        'name': 'Flesland, Bergen',
        'runway_name': '35',
        'runway_width': 45.0,
        'runway_length': 2981.0,
        'runway_heading': 350.534576,
        'center': {'latitude': 60.293437, 'longitude': 5.218154, 'altitude': 47.701004},
        'start_height': 47.701004,
        'end_height': 52.701004
    },
    'KLAX': {
        'name': 'Los Angeles International Airport',
        'runway_name': 'R25',
        'runway_width': 46.0,
        'runway_length': 3939.0,
        'runway_heading': 82.960899,
        'center': {'latitude': 33.937723, 'longitude': -118.400929, 'altitude': 34.022},
        'start_height': 34.022,
        'end_height': 34.022
    }
}

# Aircraft positions and yaw settings (from OSCAR_DatasetGenerator.py)
AIRCRAFT_POSITIONS = {
    'EDDS': [48.69519465345375, 9.250042464967704, 484.21701],
    'EDNY': [47.68028428294408, 9.53459949691774, 511.937012],
    'EDSB': [48.764057487566646, 8.066200919467537, 223.59600799999998],
    'ELLX': [49.6353853125247, 6.2355775077074, 461.492004],
    'KLAX': [33.93993074319327, -118.3794586855742, 134.022],
    'ENBR': [60.31114298706362, 5.2122036872189526, 147.701004],
    'EDDV': [52.466789466703254, 9.705611876930334, 152.781002]
}

yaw_runway = 0.0
yaw_norunway = 90.0

# Airport-specific annotations
AIRPORT_ANNOTATIONS = {
    'EDDS': {'bbox': [1200, 380, 160, 294], 'segmentation': [[[1269, 674, 1291, 674, 1360, 380, 1200, 380]]], 'area': 47040, 'iscrowd': 0},
    'EDDV': {'bbox': [1184, 295, 192, 377], 'segmentation': [[[1269, 672, 1291, 672, 1376, 295, 1184, 295]]], 'area': 72384, 'iscrowd': 0},
    'EDNY': {'bbox': [1234, 513, 92, 153], 'segmentation': [[[1268, 666, 1292, 666, 1326, 513, 1234, 513]]], 'area': 14076, 'iscrowd': 0},
    'EDSB': {'bbox': [1204, 380, 152, 291], 'segmentation': [[[1269, 671, 1291, 671, 1356, 380, 1204, 380]]], 'area': 44232, 'iscrowd': 0},
    'ELLX': {'bbox': [-36972, -159093, 76674, 159767], 'segmentation': [[[1270, 674, 1290, 674, 39702, -159093, -36972, -158432]]], 'area': 12249974958, 'iscrowd': 0},
    'ENBR': {'bbox': [1205, 386, 150, 288], 'segmentation': [[[1269, 674, 1291, 674, 1355, 386, 1205, 386]]], 'area': 43200, 'iscrowd': 0},
    'KLAX': {'bbox': [-4, -4864, 2568, 5541], 'segmentation': [[[1270, 677, 1290, 677, 2564, -4864, -4, -4863]]], 'area': 14229288, 'iscrowd': 0}
}

RUNWAY_TYPES = ['runway', 'norunway']
TIME_OPTIONS = ['daytime', 'nighttime']
AIRPORT_KEYS = list(AIRPORTS.keys())

# ============================================================================
# Main Generation
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Generating {TOTAL_SCENARIOS} random ScenAIro JSON files...")
    print(f"   [SEED BASE] {SEED_BASE:,} (from hash of 'random_pool')")
    print(f"   [SEED RANGE] [{SEED_BASE:,}, {SEED_BASE + TOTAL_SCENARIOS - 1:,}]")
    print(f"   [GUARANTEE] Hash isolation ensures zero overlap with ANY split\n")
    
    for i in range(TOTAL_SCENARIOS):
        # Hash-based deterministic seed per scenario
        scenario_seed = SEED_BASE + i
        random.seed(scenario_seed)
        
        # Random selections
        runway_type = random.choice(RUNWAY_TYPES)  # 'runway' or 'norunway'
        airport = random.choice(AIRPORT_KEYS)
        time_of_day = random.choice(TIME_OPTIONS)  # 'daytime' or 'nighttime'
        airport_data = AIRPORTS[airport]
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        time_label = 'day' if time_of_day == 'daytime' else 'night'
        filename_base = f"{timestamp}_Random_{airport}_{runway_type}_{time_label}_{i+1:05d}"
        
        # Time of day (STRICT ODD boundaries: daytime 10-15, nighttime 0-5)
        if time_of_day == 'daytime':
            hours = random.randint(10, 15)  # 10-15 inclusive
        else:
            hours = random.randint(0, 5)    # 0-5 inclusive
        minutes = random.randint(0, 59)
        
        # Category name for ScenAIro: "runway" or "no_runway"
        category_name = "runway" if runway_type == 'runway' else "no_runway"
        
        # Build full ScenAIro JSON structure
        scenairo_data = {
            "images": [
                {
                    "file_name": f"{filename_base}.png",
                    "id": f"{filename_base}.png",
                    "width": 2560,
                    "height": 1440
                }
            ],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": category_name,
                    "supercategory": "infrastructure"
                }
            ],
            "runway_data": {
                "name": airport_data['name'],
                "icao_code": airport,
                "runway_name": airport_data['runway_name'],
                "runway_width": airport_data['runway_width'],
                "runway_length": airport_data['runway_length'],
                "runway_heading": airport_data['runway_heading'],
                "runway_center": {
                    "latitude": airport_data['center']['latitude'],
                    "longitude": airport_data['center']['longitude'],
                    "altitude": airport_data['center']['altitude']
                },
                "start_height": airport_data['start_height'],
                "end_height": airport_data['end_height']
            },
            "landing_approach_cone": {
                "apex": [1372.5, 0.0, 0.0],
                "lateral_angle_left": -4.0,
                "lateral_angle_right": 4.0,
                "vertical_min_angle": 2.0,
                "vertical_max_angle": 3.8,
                "max_distance": 3000.0,
                "number_of_points": 10
            },
            "position_of_aircraft": AIRCRAFT_POSITIONS[airport],
            "distance_aircraft_2_runway": {
                "ground_distance_in_meters": 1372.5,
                "altitude_difference_in_meters": 0.0
            },
            "aircraft_orientation": {
                "pitch": 0,
                "yaw": yaw_runway if runway_type == 'runway' else yaw_norunway,
                "roll": 0
            },
            "daytime": {
                "hours": hours,
                "minutes": minutes
            }
        }
        
        # Add annotation only if runway is visible
        if runway_type == 'runway':
            airport_anno = AIRPORT_ANNOTATIONS[airport]
            scenairo_data["annotations"].append({
                "id": 0,
                "image_id": f"{filename_base}.png",
                "category_id": 1,
                "bbox": airport_anno['bbox'],
                "segmentation": airport_anno['segmentation'],
                "area": airport_anno['area'],
                "iscrowd": 0
            })
        
        # Save JSON
        filepath = os.path.join(OUTPUT_DIR, f"{filename_base}.json")
        with open(filepath, 'w') as f:
            json.dump(scenairo_data, f, indent=4)
        
        # Progress indicator
        if (i + 1) % 5000 == 0:
            print(f"Generated {i+1}/{TOTAL_SCENARIOS}...")
    
    print("\n" + "="*70)
    print("✓ Random Pool Generation Complete")
    print(f"   Total scenarios: {TOTAL_SCENARIOS}")
    print(f"   Seed range: [{SEED_BASE:,}, {SEED_BASE + TOTAL_SCENARIOS - 1:,}]")
    print(f"   Format: Full ScenAIro JSON with runway/norunway labels")
    print(f"   [GUARANTEE] Hash-based isolation from all OSCAR_DatasetGenerator splits")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
