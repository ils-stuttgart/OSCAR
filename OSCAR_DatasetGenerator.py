"""
OSCAR Dataset Generator - Creates recommendation datasets for ScenAIro
Usage: python OSCAR_DatasetGenerator.py --iteration 1 --seeds 61 116
Output: Ontology_Input_Individuals/output_image/Rec{N}-scenairo-seed{X}/
"""
import os
import json
import random
import shutil
import argparse
from datetime import datetime as dt
from collections import defaultdict
# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUERY_RESULTS_DIR = os.path.join(SCRIPT_DIR, "query_result")
EXPERIMENTS_DIR = os.path.join(SCRIPT_DIR, "OSCAR_Experiments")

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
# Runway or No_runway depends on yaw angle
YAW_RUNWAY = 0.0
YAW_NORUNWAY = 90.0

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

CONFIG = {
    'base_percentage': 0.08,  # 8% of cumulative training size
    'min_images': 5,
    'max_images': 30, #prevent crash from SceAIro & save time, to generate 30 images need already 17 min. (sleep(20))
    'use_dynamic': True,
    
    'tier1_allocation': 0.6,  # 60% from Q14 co-occurrence weaknesses
    'tier2_allocation': 0.2,  # 20% from Q21 critical errors
    'tier3_allocation': 0.15, # 15% from Q13 general weaknesses
    'tier4_allocation': 0.05, # 5% from Q8 OOD gaps
    
    'confidence_weight_enabled': True,
    'confidence_threshold_critical': 0.70,
}

# Initial training size detection
_initial_training_size = None

def get_initial_training_size():
    """Auto-detect initial training size from input_image/train folder."""
    global _initial_training_size
    if _initial_training_size is not None:
        return _initial_training_size
    
    train_dir = os.path.join(SCRIPT_DIR, "Ontology_Input_Individuals", "input_image", "train")
    if not os.path.exists(train_dir):
        _initial_training_size = 120  # Default for OSCAR
        return _initial_training_size
    
    count = sum(len([f for f in os.listdir(os.path.join(train_dir, d)) if f.endswith('.json')]) 
                for d in ['norunway', 'runway'] if os.path.exists(os.path.join(train_dir, d)))
    
    _initial_training_size = count if count > 0 else 120
    print(f"[AUTO-DETECT] Initial training size: {_initial_training_size}")
    return _initial_training_size

def get_cumulative_training_size(iteration_num):
    total = get_initial_training_size()
    if iteration_num > 1:
        for _ in range(1, iteration_num):
            increment = round(total * CONFIG['base_percentage'])
            total += max(CONFIG['min_images'], min(CONFIG['max_images'], increment))
    return total

def get_image_limit(iteration_num):
    if not CONFIG['use_dynamic']:
        return 10
    current_size = get_cumulative_training_size(iteration_num)
    limit = max(CONFIG['min_images'], min(CONFIG['max_images'], round(current_size * CONFIG['base_percentage'])))
    print(f"[ADAPTIVE] Iter {iteration_num}: cumulative={current_size} × 8% = {limit} images")
    return limit
# ============================================================================
# Query Result Extraction
# ============================================================================
def extract_candidates_improved(query_results, limit):
    """Extract and prioritize candidates from query results."""
    candidates = []
    
    # Extract results section
    results = query_results.get('results', query_results)
    
    # Tier 1: Q14 Co-occurrence weaknesses (multi-dimensional)
    if 'Q14_CNN_Cooccurrence_Weaknesses' in results:
        for item in results['Q14_CNN_Cooccurrence_Weaknesses']:
            weakness = item[0] if isinstance(item, list) else item.get('weakness', '')
            runway_type, airport, time_of_day = parse_weakness_name(weakness)
            if runway_type and airport and time_of_day:
                candidates.append({
                    'runway': runway_type,
                    'airport': airport,
                    'time_of_day': time_of_day,
                    'tier': 1,
                    'source': 'Q14',
                    'confidence': None
                })
    
    # Tier 2: Q21 Critical errors (high confidence failures)
    if 'Q21_Critical_Errors_HighConfidence' in results:
        for item in results['Q21_Critical_Errors_HighConfidence']:
            if isinstance(item, list) and len(item) >= 7:
                runway_label = item[1] 
                airport_code = item[2]  
                time_label = item[3]  
                confidence = float(item[6])
                
                candidates.append({
                    'runway': runway_label,
                    'airport': airport_code,
                    'time_of_day': time_label,
                    'tier': 2,
                    'source': 'Q21',
                    'confidence': confidence
                })
    
    # Tier 3: Q13 General weaknesses (1D weaknesses)
    if 'Q13_CNN_1D_Weaknesses' in results:
        for item in results['Q13_CNN_1D_Weaknesses']:
            weakness = item[0] if isinstance(item, list) else item.get('weakness', '')
            dimension = parse_1d_weakness(weakness)
            if dimension:
                candidates.append({
                    **dimension,
                    'tier': 3,
                    'source': 'Q13',
                    'confidence': None
                })
    
    # Tier 4: Q8 OOD gaps (unseen combinations)
    if 'Q8_OOD_Combinations_NotInTraining' in results:
        for item in results['Q8_OOD_Combinations_NotInTraining']:
            # Format: [runway_obj, airport_obj, time_obj, count]
            # Q8 uses object properties: "with_runway", "klax_airport", "day_time"
            if isinstance(item, list) and len(item) >= 3:
                runway_label = 'runway' if item[0] == 'with_runway' else 'norunway'
                airport_code = item[1].replace('_airport', '').upper()
                time_label = 'daytime' if item[2] == 'day_time' else 'nighttime'
                
                candidates.append({
                    'runway': runway_label,
                    'airport': airport_code,
                    'time_of_day': time_label,
                    'tier': 4,
                    'source': 'Q8',
                    'confidence': None
                })
    
    # Apply tier-based allocation
    final_queue = []
    allocations = {
        1: int(limit * CONFIG['tier1_allocation']),
        2: int(limit * CONFIG['tier2_allocation']),
        3: int(limit * CONFIG['tier3_allocation']),
        4: int(limit * CONFIG['tier4_allocation'])
    }
    
    for tier in [1, 2, 3, 4]:
        tier_candidates = [c for c in candidates if c['tier'] == tier]
        tier_limit = allocations[tier]
        
        # Confidence-based sampling for tier 2
        if tier == 2 and CONFIG['confidence_weight_enabled']:
            tier_candidates = sorted(tier_candidates, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Sample with replacement if needed
        if len(tier_candidates) < tier_limit:
            sampled = tier_candidates * (tier_limit // len(tier_candidates) + 1) if tier_candidates else []
            sampled = sampled[:tier_limit]
        else:
            sampled = random.sample(tier_candidates, tier_limit)
        
        final_queue.extend(sampled)
    
    while len(final_queue) < limit and candidates:
        final_queue.append(random.choice(candidates))
    
    return final_queue[:limit]

def parse_weakness_name(weakness):
    """Parse co-occurrence weakness name to extract ODD dimensions.
    Examples: PoorOnRunwayEDDSNighttime -> (runway, EDDS, nighttime)
    """
    runway_type = airport = time_of_day = None
    if 'NoRunway' in weakness:
        runway_type = 'norunway'
        remainder = weakness.split('NoRunway')[1]
    elif 'Runway' in weakness:
        runway_type = 'runway'
        remainder = weakness.split('Runway')[1]
    else:
        return None, None, None
    
    # Extract airport code and time
    for code in AIRPORTS.keys():
        if code in remainder:
            airport = code
            time_part = remainder.split(code)[1]
            
            # Extract time of day
            if 'Daytime' in time_part:
                time_of_day = 'daytime'
            elif 'Nighttime' in time_part:
                time_of_day = 'nighttime'
            break
    
    return runway_type, airport, time_of_day

def parse_1d_weakness(weakness):
    airports = list(AIRPORTS.keys())
    runways = ['runway', 'norunway']
    times = ['daytime', 'nighttime']
    
    if 'WithRunway' in weakness:
        return {'runway': 'runway', 'airport': random.choice(airports), 'time_of_day': random.choice(times)}
    elif 'NoRunway' in weakness:
        return {'runway': 'norunway', 'airport': random.choice(airports), 'time_of_day': random.choice(times)}
    elif 'Airport' in weakness:
        for code in airports:
            if code in weakness:
                return {'runway': random.choice(runways), 'airport': code, 'time_of_day': random.choice(times)}
    elif 'Daytime' in weakness:
        return {'runway': random.choice(runways), 'airport': random.choice(airports), 'time_of_day': 'daytime'}
    elif 'Nighttime' in weakness:
        return {'runway': random.choice(runways), 'airport': random.choice(airports), 'time_of_day': 'nighttime'}
    return None
# ============================================================================
# ScenAIro JSON Generation
# ============================================================================
def generate_scenairo_json(candidate, iteration_num, sequence_id):
    """Generate a single ScenAIro-format JSON file."""
    airport_code = candidate['airport']
    airport_data = AIRPORTS[airport_code]
    runway_type = candidate['runway']
    time_of_day = candidate['time_of_day']
    
    timestamp = dt.now().strftime('%Y-%m-%d_%H%M%S')
    time_label = 'day' if time_of_day == 'daytime' else 'night'
    filename_base = f"{timestamp}_{airport_code}_{runway_type}_{time_label}_rec{iteration_num}_{sequence_id:03d}"
    
    hours = random.randint(10, 15) if time_of_day == 'daytime' else random.randint(0, 5)
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
            "yaw": YAW_RUNWAY if runway_type == 'runway' else YAW_NORUNWAY,
            "roll": 0
        },
        "daytime": {
            "hours": hours,
            "minutes": minutes
        }
    }
    
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

def generate_dataset(query_results, output_dir, iteration_num, limit):
    """Generate ScenAIro-format dataset from query results."""
    print(f"\n{'='*70}")
    print(f"OSCAR DATASET GENERATION - Iteration {iteration_num}")
    print(f"Target: {limit} scenarios for ScenAIro")
    print(f"{'='*70}")
    
    for d in ['runway', 'norunway']:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)
    
    final_queue = extract_candidates_improved(query_results, limit)
    tier_counts = defaultdict(int)
    for p in final_queue:
        tier_counts[p['tier']] += 1
    
    print(f"\n[GENERATION] Tier distribution:")
    for tier in sorted(tier_counts.keys()):
        pct = tier_counts[tier] / len(final_queue) * 100 if final_queue else 0
        print(f"   Tier {tier}: {tier_counts[tier]} ({pct:.1f}%)")
    
    generated_files = []
    for i, candidate in enumerate(final_queue, 1):
        filename_base, scenairo_data = generate_scenairo_json(candidate, iteration_num, i)
        runway_type = candidate['runway']
        json_path = os.path.join(output_dir, runway_type, f"{filename_base}.json")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scenairo_data, f, indent=4)
        
        generated_files.append({
            'filename': f"{runway_type}/{filename_base}.json",
            'runway': candidate['runway'],
            'airport': candidate['airport'],
            'time_of_day': candidate['time_of_day'],
            'tier': candidate['tier'],
            'source': candidate['source']
        })
    
    summary = {
        'total_scenarios': len(generated_files),
        'target_limit': limit,
        'iteration': iteration_num,
        'tier_distribution': dict(tier_counts),
        'tier_percentages': {f'tier_{k}': v/len(final_queue)*100 for k, v in tier_counts.items()} if final_queue else {},
        'config': CONFIG,
        'scenarios': generated_files
    }
    
    with open(os.path.join(output_dir, f"summary_rec{iteration_num}_metadata.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Generated {len(generated_files)} scenarios")
    print(f"   Tier 1 (Q14): {tier_counts.get(1, 0)} ({tier_counts.get(1, 0)/limit*100:.1f}%)")
    print(f"   Tier 2 (Q21): {tier_counts.get(2, 0)} ({tier_counts.get(2, 0)/limit*100:.1f}%)")
    print(f"   Tier 3 (Q13): {tier_counts.get(3, 0)} ({tier_counts.get(3, 0)/limit*100:.1f}%)")
    print(f"   Tier 4 (Q8):  {tier_counts.get(4, 0)} ({tier_counts.get(4, 0)/limit*100:.1f}%)")
    print(f"✓ Summary saved: summary_rec{iteration_num}_metadata.json")
    print(f"{'='*70}")

def main(iteration_num=1, seed=42):
    """Generate ontology-guided JSON files (only JSONs, no images)."""
    seed_suffix = f"_seed{seed}"
    query_path = os.path.join(QUERY_RESULTS_DIR, f"querying_result{iteration_num}{seed_suffix}.json")
    
    # New structure: OSCAR_Experiments/seed_XX/Ontology/datasets/RecN-scenairo-seedXX/
    output_dir = os.path.join(EXPERIMENTS_DIR, f"seed_{seed}", "Ontology", "datasets", f"Rec{iteration_num}-scenairo-seed{seed}")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    if not os.path.exists(query_path):
        print(f"✗ Query results not found: {query_path}\n  Run OSCAR_Query.py first.")
        return
    
    with open(query_path, 'r', encoding='utf-8') as f:
        query_results = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"Generating Ontology-Guided JSON Files (Iteration {iteration_num}, Seed {seed})")
    print(f"Output: {output_dir}")
    print(f"Note: Only JSON files generated. Render images separately in ScenAIro/MSFS.")
    print(f"{'='*70}\n")
    
    generate_dataset(query_results, output_dir, iteration_num, get_image_limit(iteration_num))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OSCAR Dataset Generator - Ontology-guided JSON generation')
    parser.add_argument('--iteration', type=int, default=1, help='Iteration number')
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated seed numbers (e.g., 12,42,88)')
    parser.add_argument('--min-images', type=int, help='Minimum images (ignored, for compatibility)')
    parser.add_argument('--max-images', type=int, help='Maximum images (ignored, for compatibility)')
    args = parser.parse_args()
    
    seeds = [s.strip() for s in args.seeds.split(',')]
    for seed_str in seeds:
        seed_num = int(seed_str)
        print(f"\n{'#'*80}")
        print(f"# Processing seed: {seed_num}")
        print(f"{'#'*80}")
        main(args.iteration, seed_num)
