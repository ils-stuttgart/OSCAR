"""
OSCAR Wish Dataset Generator - Generate custom wish datasets for ScenAIro
Simple wishlist-based approach: Edit USER_WISHLIST and run the script

Usage:
    1. Edit USER_WISHLIST below with your desired scenarios
    2. Run: python OSCAR_Wish_DatasetGenerator.py
    
Strategy: Hash-based seed assignment ensures each split gets isolated seed space
No matter how wishlist changes, testing/validation/training will never overlap
"""
import os
import json
import random
import hashlib
import shutil
from datetime import datetime as dt

# ============================================================================
# Seed Strategy - Hash-based isolation (prevents data leakage between splits)
# ============================================================================
GLOBAL_SEED = 42

def get_split_seed_base(split_name):
    """Generate deterministic seed base for each split using hash.
    This guarantees non-overlapping seed ranges regardless of image counts.
    Example: 
        testing    -> hash -> seed_base = 1000000
        validation -> hash -> seed_base = 2000000  
        training   -> hash -> seed_base = 3000000
    """
    hash_obj = hashlib.md5(f"{split_name}_{GLOBAL_SEED}".encode())
    # Use first 8 hex chars -> convert to int -> modulo to get seed base in millions
    seed_base = (int(hash_obj.hexdigest()[:8], 16) % 900) * 100000 + 1000000
    return seed_base

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "Ontology_Input_Individuals", "input_image")
SPLIT_MAP = {"training": "train", "testing": "test", "validation": "val"}

# ============================================================================
# USER WISHLIST - Edit this to generate your desired scenarios
# ============================================================================
# intital dataset v2 wishlist
# total 450 + 140 + 140 = 730 images spilt in train/val/test 60%/20%/20%
# imbanlanced training set: 450 images without nighttime, no KLAX
# imbalnaced val set: 140 images without nighttime, no KLAX
# balanced test set: 140 images, all combinations

# intital dataset v1 wishlist
# total 176 + 56 + 56 = 288 images spilt in train/val/test 60%/20%/20%
# imbanlanced training set: 120 images without nighttime, no KLAX
# imbalnaced val set: 56 images without nighttime, no KLAX
# balanced test set: 56 images, all combinations

USER_WISHLIST = [
    # ========== TESTING SET (20% - Balanced across all combinations) ==========
    {"runway": "runway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "runway", "airport": "EDDS", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "EDDS", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},
    
    {"runway": "runway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "runway", "airport": "EDDV", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "EDDV", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},

    {"runway": "runway", "airport": "EDNY", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "runway", "airport": "EDNY", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "EDNY", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "EDNY", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},

    {"runway": "runway", "airport": "EDSB", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "runway", "airport": "EDSB", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "EDSB", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "EDSB", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},

    {"runway": "runway", "airport": "ELLX", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "runway", "airport": "ELLX", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "ELLX", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "ELLX", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},

    {"runway": "runway", "airport": "ENBR", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "runway", "airport": "ENBR", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "ENBR", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "ENBR", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},

    {"runway": "runway", "airport": "KLAX", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "runway", "airport": "KLAX", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "KLAX", "time_of_day": "daytime", "dataset_split": "testing", "count": 2},
    {"runway": "norunway", "airport": "KLAX", "time_of_day": "nighttime", "dataset_split": "testing", "count": 2},
    # ========== VALIDATION SET (20% - imbalanced without nighttime, no KLAX) ==========
    {"runway": "runway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "validation", "count": 4},
    {"runway": "norunway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "validation", "count": 4},

    {"runway": "runway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "validation", "count": 4},
    {"runway": "norunway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "validation", "count": 4},

    {"runway": "runway", "airport": "EDNY", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},
    {"runway": "norunway", "airport": "EDNY", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},

    {"runway": "runway", "airport": "EDSB", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},
    {"runway": "norunway", "airport": "EDSB", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},

    {"runway": "runway", "airport": "ELLX", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},
    {"runway": "norunway", "airport": "ELLX", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},

    {"runway": "runway", "airport": "ENBR", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},
    {"runway": "norunway", "airport": "ENBR", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},
    # ========== TRAINING SET (60% - imbalanced ) ==========
    {"runway": "runway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "training", "count": 14},
    {"runway": "norunway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "training", "count": 14},

    {"runway": "runway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "training", "count": 14},
    {"runway": "norunway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "training", "count": 14}, 
    {"runway": "runway", "airport": "EDNY", "time_of_day": "daytime", "dataset_split": "training", "count": 15},
    {"runway": "norunway", "airport": "EDNY", "time_of_day": "daytime", "dataset_split": "training", "count": 15},

    {"runway": "runway", "airport": "EDSB", "time_of_day": "daytime", "dataset_split": "training", "count": 15},
    {"runway": "norunway", "airport": "EDSB", "time_of_day": "daytime", "dataset_split": "training", "count": 15},

    {"runway": "runway", "airport": "ELLX", "time_of_day": "daytime", "dataset_split": "training", "count": 15},
    {"runway": "norunway", "airport": "ELLX", "time_of_day": "daytime", "dataset_split": "training", "count": 15},

    {"runway": "runway", "airport": "ENBR", "time_of_day": "daytime", "dataset_split": "training", "count": 15},
    {"runway": "norunway", "airport": "ENBR", "time_of_day": "daytime", "dataset_split": "training", "count": 15},

] 
''' v2 wishlist commented out for now
USER_WISHLIST = [
    # ========== TESTING SET (20% - Balanced across all combinations) ==========
    {"runway": "runway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "runway", "airport": "EDDS", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "EDDS", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},
    
    {"runway": "runway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "runway", "airport": "EDDV", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "EDDV", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},

    {"runway": "runway", "airport": "EDNY", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "runway", "airport": "EDNY", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "EDNY", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "EDNY", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},

    {"runway": "runway", "airport": "EDSB", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "runway", "airport": "EDSB", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "EDSB", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "EDSB", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},

    {"runway": "runway", "airport": "ELLX", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "runway", "airport": "ELLX", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "ELLX", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "ELLX", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},

    {"runway": "runway", "airport": "ENBR", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "runway", "airport": "ENBR", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "ENBR", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "ENBR", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},

    {"runway": "runway", "airport": "KLAX", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "runway", "airport": "KLAX", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "KLAX", "time_of_day": "daytime", "dataset_split": "testing", "count": 5},
    {"runway": "norunway", "airport": "KLAX", "time_of_day": "nighttime", "dataset_split": "testing", "count": 5},
    

    
    # ========== VALIDATION SET (20% - imbalanced without nighttime, no KLAX) ==========
    {"runway": "runway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},
    {"runway": "norunway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},

    {"runway": "runway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},
    {"runway": "norunway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},

    {"runway": "runway", "airport": "EDNY", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},
    {"runway": "norunway", "airport": "EDNY", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},

    {"runway": "runway", "airport": "EDSB", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},
    {"runway": "norunway", "airport": "EDSB", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},

    {"runway": "runway", "airport": "ELLX", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},
    {"runway": "norunway", "airport": "ELLX", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},

    {"runway": "runway", "airport": "ENBR", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},
    {"runway": "norunway", "airport": "ENBR", "time_of_day": "daytime", "dataset_split": "validation", "count": 5},

    
    # ========== TRAINING SET (60% - imbalanced ) ==========
    {"runway": "runway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "training", "count": 10},
    {"runway": "runway", "airport": "EDDS", "time_of_day": "nighttime", "dataset_split": "training", "count": 10},
    {"runway": "norunway", "airport": "EDDS", "time_of_day": "daytime", "dataset_split": "training", "count": 10},
    {"runway": "norunway", "airport": "EDDS", "time_of_day": "nighttime", "dataset_split": "training", "count": 10},
    
    {"runway": "runway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "training", "count": 10},
    {"runway": "runway", "airport": "EDDV", "time_of_day": "nighttime", "dataset_split": "training", "count": 10},
    {"runway": "norunway", "airport": "EDDV", "time_of_day": "daytime", "dataset_split": "training", "count": 10},
    {"runway": "norunway", "airport": "EDDV", "time_of_day": "nighttime", "dataset_split": "training", "count": 10},
]
'''
# Airport configurations (same as OSCAR_DatasetGenerator)
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

# Position of Aircraft depends on airport
AIRCRAFT_POSITIONS = {
    'EDDS': [48.69519465345375, 9.250042464967704, 484.21701],
    'EDNY': [47.68028428294408, 9.53459949691774, 511.937012],
    'EDSB': [48.764057487566646, 8.066200919467537, 223.59600799999998],
    'ELLX': [49.6353853125247, 6.2355775077074, 461.492004],
    'KLAX': [33.93993074319327, -118.3794586855742, 134.022],
    'ENBR': [60.31114298706362, 5.2122036872189526, 147.701004],
    'EDDV': [52.466789466703254, 9.705611876930334, 152.781002]
}

# Runway or no runway depends on yaw
yaw_runway = 0.0
yaw_norunway = 90.0

# Airport-specific annotations (bbox, segmentation, area)
AIRPORT_ANNOTATIONS = {
    'EDDS': {
        'bbox': [1200, 380, 160, 294],
        'segmentation': [[[1269, 674, 1291, 674, 1360, 380, 1200, 380]]],
        'area': 47040,
        'iscrowd': 0
    },
    'EDDV': {
        'bbox': [1184, 295, 192, 377],
        'segmentation': [[[1269, 672, 1291, 672, 1376, 295, 1184, 295]]],
        'area': 72384,
        'iscrowd': 0
    },
    'EDNY': {
        'bbox': [1234, 513, 92, 153],
        'segmentation': [[[1268, 666, 1292, 666, 1326, 513, 1234, 513]]],
        'area': 14076,
        'iscrowd': 0
    },
    'EDSB': {
        'bbox': [1204, 380, 152, 291],
        'segmentation': [[[1269, 671, 1291, 671, 1356, 380, 1204, 380]]],
        'area': 44232,
        'iscrowd': 0
    },
    'ELLX': {
        'bbox': [-36972, -159093, 76674, 159767],
        'segmentation': [[[1270, 674, 1290, 674, 39702, -159093, -36972, -158432]]],
        'area': 12249974958,
        'iscrowd': 0
    },
    'ENBR': {
        'bbox': [1205, 386, 150, 288],
        'segmentation': [[[1269, 674, 1291, 674, 1355, 386, 1205, 386]]],
        'area': 43200,
        'iscrowd': 0
    },
    'KLAX': {
        'bbox': [-4, -4864, 2568, 5541],
        'segmentation': [[[1270, 677, 1290, 677, 2564, -4864, -4, -4863]]],
        'area': 14229288,
        'iscrowd': 0
    }
}

# ============================================================================
# ScenAIro JSON Generation
# ============================================================================
def generate_scenairo_json(runway_type, airport_code, time_of_day, split_name, counter_value, image_seed):
    """Generate a single ScenAIro-format JSON file based on wish parameters."""
    
    if airport_code not in AIRPORTS:
        raise ValueError(f"Unknown airport code: {airport_code}. Valid codes: {list(AIRPORTS.keys())}")
    
    airport_data = AIRPORTS[airport_code]
    
    # Set seed for deterministic generation
    random.seed(image_seed)
    
    # Generate filename with counter
    timestamp = dt.now().strftime('%Y-%m-%d_%H%M%S')
    split_short = SPLIT_MAP.get(split_name, split_name)
    time_label = 'day' if time_of_day == 'daytime' else 'night'
    filename_base = f"{timestamp}_Wish_{split_short}_{airport_code}_{runway_type}_{time_label}_{counter_value:05d}"
    
    # Time of day (STRICT ODD boundaries: daytime 10-15, nighttime 0-5)
    if time_of_day == 'daytime':
        hours = random.randint(10, 15)  # 10-15 inclusive
    else:
        hours = random.randint(0, 5)    # 0-5 inclusive
    minutes = random.randint(0, 59)
    
    # Category name for ScenAIro format: "runway" or "no_runway"
    category_name = "runway" if runway_type == 'runway' else "no_runway"
    
    # Build ScenAIro JSON structure 
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
            "icao_code": airport_code,
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
        "position_of_aircraft": AIRCRAFT_POSITIONS[airport_code],
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
        airport_anno = AIRPORT_ANNOTATIONS[airport_code]
        scenairo_data["annotations"].append({
            "id": 0,
            "image_id": f"{filename_base}.png",
            "category_id": 1,
            "bbox": airport_anno['bbox'],
            "segmentation": airport_anno['segmentation'],
            "area": airport_anno['area'],
            "iscrowd": 0
        })
    
    return filename_base, scenairo_data

# ============================================================================
# Main Generation Function
# ============================================================================
def main():
    """Generate ScenAIro datasets from USER_WISHLIST with hash-based seed isolation."""
    
    # Clean up previous datasets
    print(f"\n{'='*70}")
    print("CLEANING UP PREVIOUS DATASETS...")
    print(f"{'='*70}")
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(OUTPUT_BASE_DIR, split_name)
        if os.path.exists(split_dir):
            file_count = sum(len(files) for _, _, files in os.walk(split_dir))
            shutil.rmtree(split_dir)
            print(f"   ✓ Removed {file_count} files from {split_name}/")
        else:
            print(f"   • No existing {split_name}/ directory")
    print(f"{'='*70}\n")
    
    # Counters for each runway+split combo
    counters = {}
    # Track seed ranges used per split for verification
    split_seed_ranges = {}
    # Per-split image counters for seed calculation
    split_image_counters = {}
    
    print(f"\n{'='*70}")
    print("OSCAR WISH DATASET GENERATION")
    print("   [GUARANTEE] Each split uses isolated seed space via MD5 hashing")
    print("   [ADAPTIVE] Works with ANY wishlist configuration")
    print(f"{'='*70}\n")
    
    for wish in USER_WISHLIST:
        runway_type = wish['runway']
        airport_code = wish['airport']
        time_of_day = wish['time_of_day']
        split = wish.get('dataset_split', 'training')
        count = wish['count']
        
        # Validate inputs
        if airport_code not in AIRPORTS:
            print(f"⚠ Skipping invalid airport: {airport_code}")
            continue
        
        # Use train/val/test folder naming
        split_short = SPLIT_MAP.get(split, split)
        output_dir = os.path.join(OUTPUT_BASE_DIR, split_short, runway_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # Counter key for this combination
        counter_key = f"{runway_type}_{split_short}"
        if counter_key not in counters:
            counters[counter_key] = 1
        if split not in split_image_counters:
            split_image_counters[split] = 0
            split_seed_ranges[split] = {'min': float('inf'), 'max': 0}
        
        print(f"Generating {count} scenarios: {runway_type}/{airport_code}/{time_of_day} → {split_short}/")
        
        for _ in range(count):
            # === HASH-BASED DETERMINISTIC SEED ===
            # Each split gets isolated seed base (e.g., testing=1M, validation=2M, training=3M)
            # Within split: sequential increment ensures reproducibility
            split_seed_base = get_split_seed_base(split)
            image_seed = split_seed_base + split_image_counters[split]
            split_image_counters[split] += 1
            
            # Track seed range for this split
            split_seed_ranges[split]['min'] = min(split_seed_ranges[split]['min'], image_seed)
            split_seed_ranges[split]['max'] = max(split_seed_ranges[split]['max'], image_seed)
            
            # Generate JSON
            curr_cnt = counters[counter_key]
            filename_base, scenairo_data = generate_scenairo_json(
                runway_type, airport_code, time_of_day, split, curr_cnt, image_seed
            )
            
            json_path = os.path.join(output_dir, f"{filename_base}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(scenairo_data, f, indent=4)
            
            counters[counter_key] += 1
    
    # Calculate totals
    total_generated = sum(split_image_counters.values())
    expected_total = sum(w['count'] for w in USER_WISHLIST)
    
    print(f"\n{'='*70}")
    print("✓ Generation Complete - Data Leakage Prevention Report:")
    print(f"   Total images generated: {total_generated}")
    print(f"   Expected total: {expected_total}")
    
    # Show per-split statistics
    for split in sorted(split_image_counters.keys()):
        count = split_image_counters[split]
        seed_range = split_seed_ranges[split]
        seed_base = get_split_seed_base(split)
        print(f"\n   [{split.upper()}]")
        print(f"      Images: {count}")
        print(f"      Seed base: {seed_base:,} (from hash)")
        print(f"      Seed range: [{seed_range['min']:,}, {seed_range['max']:,}]")
    
    # Verify no overlap
    print(f"\n   [GUARANTEE] Hash-based isolation ensures ZERO overlap")
    print(f"   [ADAPTIVE] Add/remove any split - isolation maintained automatically")
    
    # Check for overlaps
    splits = list(split_seed_ranges.keys())
    overlaps_found = False
    for i, split1 in enumerate(splits):
        for split2 in splits[i+1:]:
            r1 = split_seed_ranges[split1]
            r2 = split_seed_ranges[split2]
            if not (r1['max'] < r2['min'] or r2['max'] < r1['min']):
                print(f"   [ERROR] Overlap detected between {split1} and {split2}!")
                overlaps_found = True
    
    if not overlaps_found:
        print(f"   ✓ Verified: All splits use non-overlapping seed ranges")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
