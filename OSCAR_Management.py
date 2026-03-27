"""
Usage: python OSCAR_Management.py --iteration 1 --seeds 61 42 116
"""
from owlready2 import *
import os
import json
import argparse
from datetime import datetime

# Configuration paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OWL_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Ontology_Owl_files")
EXPERIMENTS_DIR = os.path.join(SCRIPT_DIR, "OSCAR_Experiments")
# Legacy: Keep for initial training images (train/val/test)
INPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "Ontology_Input_Individuals")
INPUT_DIR = os.path.join(INPUT_BASE_DIR, "input_image")
os.makedirs(OWL_OUTPUT_DIR, exist_ok=True)
onto = get_ontology("http://example.org/OSCAR.owl")

TIME_OF_DAY_THRESHOLDS = {
    'daytime_start': 10, 'daytime_end': 16,       # 10:00 - 15:59 (inclusive of hour 15)
    'nighttime_start': 0, 'nighttime_end': 6      # 00:00 - 05:59 (inclusive of hour 5)
}

with onto:
    class Image(Thing): 
        comment = ["Image captured during landing approach"]
    
    # =========================================================================
    # ODD1: Runway Classes (Target Label + Runway Properties)
    # =========================================================================
    class Runway(Thing): 
        comment = ["Runway classification and properties"]
    class WithRunway(Runway): 
        comment = ["Runway visible"]
    class NoRunway(Runway): 
        comment = ["No runway visible"]
    AllDisjoint([WithRunway, NoRunway])
    with_runway = WithRunway("with_runway")
    no_runway = NoRunway("no_runway")
    
    class Airport(Thing): pass
    class EDDS(Airport): pass
    class EDDV(Airport): pass
    class EDNY(Airport): pass
    class EDSB(Airport): pass
    class ELLX(Airport): pass
    class ENBR(Airport): pass
    class KLAX(Airport): pass

    class TimeOfDay(Thing): pass
    class Daytime(TimeOfDay): pass
    class Nighttime(TimeOfDay): pass
    AllDisjoint([Daytime, Nighttime])

    class CNN(Thing): pass
    class Output(Thing): pass
    
    class CNNWeakness(Thing): pass
    class CNN1DWeakness(CNNWeakness): pass
    class RunwayWeakness(CNN1DWeakness): pass
    class PoorOnWithRunway(RunwayWeakness): pass
    class PoorOnNoRunway(RunwayWeakness): pass
    AllDisjoint([PoorOnWithRunway, PoorOnNoRunway])
    class AirportWeakness(CNN1DWeakness): pass
    class PoorOnEDDSAirport(AirportWeakness): pass
    class PoorOnEDDVAirport(AirportWeakness): pass
    class PoorOnEDNYAirport(AirportWeakness): pass
    class PoorOnEDSBAirport(AirportWeakness): pass
    class PoorOnELLXAirport(AirportWeakness): pass
    class PoorOnENBRAirport(AirportWeakness): pass
    class PoorOnKLAXAirport(AirportWeakness): pass
    AllDisjoint([PoorOnEDDSAirport, PoorOnEDDVAirport, PoorOnEDNYAirport, PoorOnEDSBAirport, PoorOnELLXAirport, PoorOnENBRAirport, PoorOnKLAXAirport])
    class TimeOfDayWeakness(CNN1DWeakness): pass
    class PoorOnDaytime(TimeOfDayWeakness): pass
    class PoorOnNighttime(TimeOfDayWeakness): pass
    AllDisjoint([PoorOnDaytime, PoorOnNighttime])
    
    class CooccurrenceWeakness(CNNWeakness): pass
    class PoorOnRunwayEDDSNighttime(CooccurrenceWeakness): pass
    class PoorOnRunwayEDDSDaytime(CooccurrenceWeakness): pass
    class PoorOnRunwayEDDVNighttime(CooccurrenceWeakness): pass
    class PoorOnRunwayEDDVDaytime(CooccurrenceWeakness): pass
    class PoorOnRunwayEDNYNighttime(CooccurrenceWeakness): pass
    class PoorOnRunwayEDNYDaytime(CooccurrenceWeakness): pass
    class PoorOnRunwayEDSBNighttime(CooccurrenceWeakness): pass
    class PoorOnRunwayEDSBDaytime(CooccurrenceWeakness): pass
    class PoorOnRunwayELLXNighttime(CooccurrenceWeakness): pass
    class PoorOnRunwayELLXDaytime(CooccurrenceWeakness): pass   
    class PoorOnRunwayENBRNighttime(CooccurrenceWeakness): pass
    class PoorOnRunwayENBRDaytime(CooccurrenceWeakness): pass
    class PoorOnRunwayKLAXNighttime(CooccurrenceWeakness): pass
    class PoorOnRunwayKLAXDaytime(CooccurrenceWeakness): pass

    class PoorOnNoRunwayEDDSNighttime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayEDDSDaytime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayEDDVNighttime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayEDDVDaytime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayEDNYNighttime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayEDNYDaytime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayEDSBNighttime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayEDSBDaytime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayELLXNighttime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayELLXDaytime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayENBRNighttime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayENBRDaytime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayKLAXNighttime(CooccurrenceWeakness): pass
    class PoorOnNoRunwayKLAXDaytime(CooccurrenceWeakness): pass
    AllDisjoint([PoorOnRunwayEDDSNighttime, PoorOnRunwayEDDSDaytime, PoorOnRunwayEDDVNighttime, PoorOnRunwayEDDVDaytime,
                  PoorOnRunwayEDNYNighttime, PoorOnRunwayEDNYDaytime, PoorOnRunwayEDSBNighttime, PoorOnRunwayEDSBDaytime,
                  PoorOnRunwayELLXNighttime, PoorOnRunwayELLXDaytime, PoorOnRunwayENBRNighttime, PoorOnRunwayENBRDaytime,
                  PoorOnRunwayKLAXNighttime, PoorOnRunwayKLAXDaytime, PoorOnNoRunwayEDDSNighttime, PoorOnNoRunwayEDDSDaytime,
                  PoorOnNoRunwayEDDVNighttime, PoorOnNoRunwayEDDVDaytime, PoorOnNoRunwayEDNYNighttime, PoorOnNoRunwayEDNYDaytime,
                  PoorOnNoRunwayEDSBNighttime, PoorOnNoRunwayEDSBDaytime, PoorOnNoRunwayELLXNighttime, PoorOnNoRunwayELLXDaytime,
                  PoorOnNoRunwayENBRNighttime, PoorOnNoRunwayENBRDaytime, PoorOnNoRunwayKLAXNighttime, PoorOnNoRunwayKLAXDaytime])

    # Issues and Recommendations
    class Issue(Thing): pass
    class Overfitting(Issue): pass
    class Underfitting(Issue): pass
    
    class GenerationRecommendation(Thing): pass
    class Rec_Training_Image(Image, GenerationRecommendation):
        comment = ["Generated images to address specific CNN weaknesses"]
    # Training State Management
    class TrainingState(Thing): pass
    class OptimalState(TrainingState): 
        comment = ["Stop Training: CNN meets all performance criteria"]
    class StagnationState(TrainingState): 
        comment = ["Stop Training: No improvement despite new data"]
    class NeedsImprovementState(TrainingState): 
        comment = ["Continue Training: Weaknesses persist"]
    AllDisjoint([OptimalState, StagnationState, NeedsImprovementState])

with onto:
    # -------------------------------------------------------------------------
    # Image → ODD Properties
    # -------------------------------------------------------------------------
    class hasRunway(ObjectProperty):
        domain = [Image]
        range = [Runway]
        comment = ["Links Image to its runway classification (with/without)"]
    
    class hasAirport(ObjectProperty):
        domain = [Image]
        range = [Airport]
        comment = ["Links Image to the airport where it was captured"]
   
    class hasTimeOfDay(ObjectProperty):
        domain = [Image]
        range = [TimeOfDay]
        comment = ["Links Image to time of day category"]
    # -------------------------------------------------------------------------
    # Output Properties
    # -------------------------------------------------------------------------
    class isSource(ObjectProperty):
        domain = [Output]
        range = [Image]
        comment = ["The Image this Output is a prediction for"]
    
    class isCalculatedBy(ObjectProperty):
        domain = [Output]
        range = [CNN]
        comment = ["The CNN that calculated this Output"]
    
    class hasGroundTruth(ObjectProperty, FunctionalProperty):
        domain = [Image]
        range = [Runway]
        comment = ["Ground truth runway classification (with_runway or no_runway individual)"]
    
    class hasPredictedValue(ObjectProperty, FunctionalProperty):
        domain = [Output]
        range = [Runway]
        comment = ["CNN prediction (with_runway or no_runway individual)"]
    class isTrainingOn(ObjectProperty):
        domain = [CNN]
        range = [Image]
    class isValidatingOn(ObjectProperty):
        domain = [CNN]
        range = [Image]
    class isTestingOn(ObjectProperty):
        domain = [CNN]
        range = [Image]
    class isTrainedBy(ObjectProperty):
        domain = [Image]
        range = [CNN]
        inverse_property = isTrainingOn
    class isValidatedBy(ObjectProperty):
        domain = [Image]
        range = [CNN]
        inverse_property = isValidatingOn
    class isTestedBy(ObjectProperty):
        domain = [Image]
        range = [CNN]
        inverse_property = isTestingOn
    # -------------------------------------------------------------------------
    # CNN Weakness Properties
    # -------------------------------------------------------------------------
    class hasFailureOn(ObjectProperty):
        domain = [CNN]
        range = [Image]
        comment = ["CNN misclassified this Image"]
    class has1DWeakness(ObjectProperty):
        domain = [CNN]
        range = [CNN1DWeakness]
        comment = ["CNN has this 1D weakness (inferred by SWRL)"]
    class hasCooccurrenceWeakness(ObjectProperty):
        domain = [CNN]
        range = [CooccurrenceWeakness]
        comment = ["CNN has this multi-dimensional weakness"]
    
    # -------------------------------------------------------------------------
    # Training State and Issue Properties
    # -------------------------------------------------------------------------
    class hasTrainingState(ObjectProperty):
        domain = [CNN]
        range = [TrainingState]
    class hasOverfittingIssue(ObjectProperty):
        domain = [CNN]
        range = [Overfitting]

with onto:
    class hasImageId(DataProperty, FunctionalProperty):
        domain = [Image]
        range = [str]
        comment = ["Unique identifier for the image"]
    
    class hasFileName(DataProperty, FunctionalProperty):
        domain = [Image]
        range = [str]
        comment = ["Image filename"]
    
    class hasImageWidth(DataProperty, FunctionalProperty):
        domain = [Image]
        range = [int]
        comment = ["Image width in pixels"]
    
    class hasImageHeight(DataProperty, FunctionalProperty):
        domain = [Image]
        range = [int]
        comment = ["Image height in pixels"]
    
    # -------------------------------------------------------------------------
    # ODD Label Data Properties (for SPARQL queries)
    # -------------------------------------------------------------------------
    class imageRunwayLabel(DataProperty, FunctionalProperty):
        domain = [Image]
        range = [str]
        comment = ["Human-readable runway label: 'runway' or 'norunway'"]
    
    class imageAirportCode(DataProperty, FunctionalProperty):
        domain = [Image]
        range = [str]
        comment = ["ICAO airport code: 'EDDS', 'KLAX', etc."]
    
    class imageTimeLabel(DataProperty, FunctionalProperty):
        domain = [Image]
        range = [str]
        comment = ["Time of day label: 'daytime' or 'nighttime'"]
    
    # -------------------------------------------------------------------------
    # ODD1: Runway Data Properties
    # -------------------------------------------------------------------------
    class hasRunwayName(DataProperty, FunctionalProperty):
        domain = [Runway]
        range = [str]
        comment = ["Runway name (e.g., in EDDS '25')"]
    
    class hasAirportName(DataProperty, FunctionalProperty):
        domain = [Airport]
        range = [str]
        comment = ["Airport name (e.g., 'Stuttgart')"]
    
    class hasICAOCode(DataProperty, FunctionalProperty):
        domain = [Airport]
        range = [str]
        comment = ["ICAO airport code (e.g., 'EDDS')"]
    
    class hasRunwayWidth(DataProperty, FunctionalProperty):
        domain = [Runway]
        range = [float]
        comment = ["Runway width in meters"]
    
    class hasRunwayLength(DataProperty, FunctionalProperty):
        domain = [Runway]
        range = [float]
        comment = ["Runway length in meters"]
    
    class hasRunwayHeading(DataProperty, FunctionalProperty):
        domain = [Runway]
        range = [float]
        comment = ["Runway heading in degrees"]
    
    class hasRunwayCenterLatitude(DataProperty, FunctionalProperty):
        domain = [Runway]
        range = [float]
        comment = ["Runway center latitude"]
    
    class hasRunwayCenterLongitude(DataProperty, FunctionalProperty):
        domain = [Runway]
        range = [float]
        comment = ["Runway center longitude"]
    
    class hasRunwayCenterAltitude(DataProperty, FunctionalProperty):
        domain = [Runway]
        range = [float]
        comment = ["Runway center altitude in meters"]
    
    class hasRunwayStartHeight(DataProperty, FunctionalProperty):
        domain = [Runway]
        range = [float]
        comment = ["Runway start height in meters"]
    
    class hasRunwayEndHeight(DataProperty, FunctionalProperty):
        domain = [Runway]
        range = [float]
        comment = ["Runway end height in meters"]

    # -------------------------------------------------------------------------
    # ODD4: Time Data Properties
    # -------------------------------------------------------------------------
    class hasCaptureHour(DataProperty, FunctionalProperty):
        domain = [Image]
        range = [int]
        comment = ["Hour of capture (0-23)"]
    
    class hasCaptureMinute(DataProperty, FunctionalProperty):
        domain = [Image]
        range = [int]
        comment = ["Minute of capture (0-59)"]
    
    # -------------------------------------------------------------------------
    # Output Data Properties
    # -------------------------------------------------------------------------
    class hasConfidenceScore(DataProperty, FunctionalProperty):
        domain = [Output]
        range = [float]
        comment = ["CNN confidence score (0.0 to 1.0)"]
    
    # -------------------------------------------------------------------------
    # CNN Performance Data Properties
    # -------------------------------------------------------------------------
    class hasTrainingAccuracy(DataProperty, FunctionalProperty):
        domain = [CNN]
        range = [float]
    class hasValidationAccuracy(DataProperty, FunctionalProperty):
        domain = [CNN]
        range = [float]
    class hasTestingAccuracy(DataProperty, FunctionalProperty):
        domain = [CNN]
        range = [float]
    class hasTrainingLoss(DataProperty, FunctionalProperty):
        domain = [CNN]
        range = [float]
    class hasValidationLoss(DataProperty, FunctionalProperty):
        domain, range = [CNN], [float]
    class hasTestingLoss(DataProperty, FunctionalProperty):
        domain, range = [CNN], [float]
    class hasAccuracyGap(DataProperty, FunctionalProperty):
        domain, range = [CNN], [float]
    class accuracyImprovement(DataProperty, FunctionalProperty):
        domain, range = [CNN], [float]
    class previousAccuracy(DataProperty, FunctionalProperty):
        domain, range = [CNN], [float]
# constant instances
with onto:
    day_time = Daytime("day_time")
    night_time = Nighttime("night_time")
    with_runway = WithRunway("with_runway")
    no_runway = NoRunway("no_runway")
    
    edds_airport = EDDS("edds_airport")
    eddv_airport = EDDV("eddv_airport")
    edny_airport = EDNY("edny_airport")
    edsb_airport = EDSB("edsb_airport")
    ellx_airport = ELLX("ellx_airport")
    enbr_airport = ENBR("enbr_airport")
    klax_airport = KLAX("klax_airport")
    
    PoorOnWithRunway_Weakness = PoorOnWithRunway("PoorOnWithRunway_Weakness")
    PoorOnNoRunway_Weakness = PoorOnNoRunway("PoorOnNoRunway_Weakness")
    PoorOnEDDSAirport_Weakness = PoorOnEDDSAirport("PoorOnEDDSAirport_Weakness")
    PoorOnEDDVAirport_Weakness = PoorOnEDDVAirport("PoorOnEDDVAirport_Weakness")
    PoorOnEDNYAirport_Weakness = PoorOnEDNYAirport("PoorOnEDNYAirport_Weakness")
    PoorOnEDSBAirport_Weakness = PoorOnEDSBAirport("PoorOnEDSBAirport_Weakness")
    PoorOnELLXAirport_Weakness = PoorOnELLXAirport("PoorOnELLXAirport_Weakness")
    PoorOnENBRAirport_Weakness = PoorOnENBRAirport("PoorOnENBRAirport_Weakness")
    PoorOnKLAXAirport_Weakness = PoorOnKLAXAirport("PoorOnKLAXAirport_Weakness") 
    PoorOnDaytime_Weakness = PoorOnDaytime("PoorOnDaytime_Weakness")
    PoorOnNighttime_Weakness = PoorOnNighttime("PoorOnNighttime_Weakness")
    
    # Co-occurrence Weakness instances 
    PoorOnRunwayEDDSNighttime_Weakness = PoorOnRunwayEDDSNighttime("PoorOnRunwayEDDSNighttime_Weakness")
    PoorOnRunwayEDDSDaytime_Weakness = PoorOnRunwayEDDSDaytime("PoorOnRunwayEDDSDaytime_Weakness")
    PoorOnRunwayEDDVNighttime_Weakness = PoorOnRunwayEDDVNighttime("PoorOnRunwayEDDVNighttime_Weakness")
    PoorOnRunwayEDDVDaytime_Weakness = PoorOnRunwayEDDVDaytime("PoorOnRunwayEDDVDaytime_Weakness")
    PoorOnRunwayEDNYNighttime_Weakness = PoorOnRunwayEDNYNighttime("PoorOnRunwayEDNYNighttime_Weakness")
    PoorOnRunwayEDNYDaytime_Weakness = PoorOnRunwayEDNYDaytime("PoorOnRunwayEDNYDaytime_Weakness")
    PoorOnRunwayEDSBNighttime_Weakness = PoorOnRunwayEDSBNighttime("PoorOnRunwayEDSBNighttime_Weakness")
    PoorOnRunwayEDSBDaytime_Weakness = PoorOnRunwayEDSBDaytime("PoorOnRunwayEDSBDaytime_Weakness")
    PoorOnRunwayELLXNighttime_Weakness = PoorOnRunwayELLXNighttime("PoorOnRunwayELLXNighttime_Weakness")
    PoorOnRunwayELLXDaytime_Weakness = PoorOnRunwayELLXDaytime("PoorOnRunwayELLXDaytime_Weakness")
    PoorOnRunwayENBRNighttime_Weakness = PoorOnRunwayENBRNighttime("PoorOnRunwayENBRNighttime_Weakness")
    PoorOnRunwayENBRDaytime_Weakness = PoorOnRunwayENBRDaytime("PoorOnRunwayENBRDaytime_Weakness")
    PoorOnRunwayKLAXNighttime_Weakness = PoorOnRunwayKLAXNighttime("PoorOnRunwayKLAXNighttime_Weakness")
    PoorOnRunwayKLAXDaytime_Weakness = PoorOnRunwayKLAXDaytime("PoorOnRunwayKLAXDaytime_Weakness")
    PoorOnNoRunwayEDDSNighttime_Weakness = PoorOnNoRunwayEDDSNighttime("PoorOnNoRunwayEDDSNighttime_Weakness")
    PoorOnNoRunwayEDDSDaytime_Weakness = PoorOnNoRunwayEDDSDaytime("PoorOnNoRunwayEDDSDaytime_Weakness")
    PoorOnNoRunwayEDDVNighttime_Weakness = PoorOnNoRunwayEDDVNighttime("PoorOnNoRunwayEDDVNighttime_Weakness")
    PoorOnNoRunwayEDDVDaytime_Weakness = PoorOnNoRunwayEDDVDaytime("PoorOnNoRunwayEDDVDaytime_Weakness")
    PoorOnNoRunwayEDNYNighttime_Weakness = PoorOnNoRunwayEDNYNighttime("PoorOnNoRunwayEDNYNighttime_Weakness")
    PoorOnNoRunwayEDNYDaytime_Weakness = PoorOnNoRunwayEDNYDaytime("PoorOnNoRunwayEDNYDaytime_Weakness")
    PoorOnNoRunwayEDSBNighttime_Weakness = PoorOnNoRunwayEDSBNighttime("PoorOnNoRunwayEDSBNighttime_Weakness")
    PoorOnNoRunwayEDSBDaytime_Weakness = PoorOnNoRunwayEDSBDaytime("PoorOnNoRunwayEDSBDaytime_Weakness")
    PoorOnNoRunwayELLXNighttime_Weakness = PoorOnNoRunwayELLXNighttime("PoorOnNoRunwayELLXNighttime_Weakness")
    PoorOnNoRunwayELLXDaytime_Weakness = PoorOnNoRunwayELLXDaytime("PoorOnNoRunwayELLXDaytime_Weakness")
    PoorOnNoRunwayENBRNighttime_Weakness = PoorOnNoRunwayENBRNighttime("PoorOnNoRunwayENBRNighttime_Weakness")
    PoorOnNoRunwayENBRDaytime_Weakness = PoorOnNoRunwayENBRDaytime("PoorOnNoRunwayENBRDaytime_Weakness")
    PoorOnNoRunwayKLAXNighttime_Weakness = PoorOnNoRunwayKLAXNighttime("PoorOnNoRunwayKLAXNighttime_Weakness")
    PoorOnNoRunwayKLAXDaytime_Weakness = PoorOnNoRunwayKLAXDaytime("PoorOnNoRunwayKLAXDaytime_Weakness")

    # Training States
    optimal_state = OptimalState("optimal_state")
    stagnation_state = StagnationState("stagnation_state")
    needs_improvement_state = NeedsImprovementState("needs_improvement_state")

    # Issue Individuals
    overfitting_issue = Overfitting("overfitting_issue")
    underfitting_issue = Underfitting("underfitting_issue")

def classify_time_of_day(hour):
    if TIME_OF_DAY_THRESHOLDS['daytime_start'] <= hour < TIME_OF_DAY_THRESHOLDS['daytime_end']:
        return onto.day_time
    elif TIME_OF_DAY_THRESHOLDS['nighttime_start'] <= hour < TIME_OF_DAY_THRESHOLDS['nighttime_end']:
        return onto.night_time
    return None

def get_airport_instance(icao_code):
    """Map ICAO code to airport instance."""
    mapping = {
        'EDDS': onto.edds_airport,
        'EDDV': onto.eddv_airport,
        'EDNY': onto.edny_airport,
        'EDSB': onto.edsb_airport,
        'ELLX': onto.ellx_airport,
        'ENBR': onto.enbr_airport,
        'KLAX': onto.klax_airport
    }
    # Try to find existing or create new
    airport = mapping.get(icao_code.upper())
    if not airport:
        # Check if dynamically created
        airport = onto.search_one(iri=f"*{icao_code.lower()}_airport")
    return airport

def create_image_from_scenAIro_json(json_path, cnn_to_link=None, split='test', is_recommendation=False):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract image metadata
    image_info = data['images'][0]
    image_id = image_info['id'].replace('.png', '').replace('.', '_').replace('-', '_')
    
    with onto:
        # Create Image individual (or Rec_Training_Image for recommendations)
        if is_recommendation and split == 'train':
            img = Rec_Training_Image(f"img_{image_id}")
        else:
            img = Image(f"img_{image_id}")
        img.hasImageId = image_info['id']
        img.hasFileName = image_info['file_name']
        img.hasImageWidth = image_info['width']
        img.hasImageHeight = image_info['height']
        
        # ODD1: Set ground truth based on category
        category = data.get('categories', [{}])[0]
        if category.get('name') == 'runway':
            img.hasGroundTruth = onto.with_runway
            img.hasRunway = [onto.with_runway]
            img.imageRunwayLabel = "runway"
        else:
            img.hasGroundTruth = onto.no_runway
            img.hasRunway = [onto.no_runway]
            img.imageRunwayLabel = "norunway"
        
        # Parse annotations (bounding box) - only for runway images
        if data.get('annotations'):
            ann = data['annotations'][0]
            bbox = ann.get('bbox', [0, 0, 0, 0])
            img.hasBBoxX = float(bbox[0])
            img.hasBBoxY = float(bbox[1])
            img.hasBBoxWidth = float(bbox[2])
            img.hasBBoxHeight = float(bbox[3])
            img.hasBBoxArea = float(ann.get('area', 0))
        
        # ODD2: Parse runway_data (Airport)
        runway_data = data.get('runway_data', {})
        if runway_data:
            icao_code = runway_data.get('icao_code', 'EDDS')
            airport_inst = get_airport_instance(icao_code)
            
            if airport_inst:
                img.hasAirport = [airport_inst]
                img.imageAirportCode = icao_code
            
            # Store runway properties as data properties
            img.hasRunwayName = runway_data.get('runway_name', '00')
            img.hasRunwayWidth = float(runway_data.get('runway_width', 45.0))
            img.hasRunwayLength = float(runway_data.get('runway_length', 3000.0))
        
        # ODD3: Parse daytime (Time of Day)
        daytime_data = data.get('daytime', {})
        if daytime_data:
            hour = int(daytime_data.get('hours', 12))
            minute = int(daytime_data.get('minutes', 0))
            img.hasCaptureHour = hour
            img.hasCaptureMinute = minute
            
            # Classify time of day
            time_inst = classify_time_of_day(hour)
            if time_inst:
                img.hasTimeOfDay = [time_inst]
                img.imageTimeLabel = 'daytime' if time_inst == onto.day_time else 'nighttime'
        
        # Link to CNN based on dataset split
        if cnn_to_link:
            if split == 'train':
                img.isTrainedBy = [cnn_to_link]
            elif split == 'val':
                img.isValidatedBy = [cnn_to_link]
            elif split == 'test':
                img.isTestedBy = [cnn_to_link]
        
        return img

def populate_cnn_results(cnn_result_json, cnn_instance):
    if isinstance(cnn_result_json, str):
        if not os.path.exists(cnn_result_json):
            print(f"[!] CNN results file not found: {cnn_result_json}")
            return
        
        with open(cnn_result_json, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = cnn_result_json
    
    # Extract misclassifications only
    predictions = []
    
    if 'misclassifications' in results:
        misclass = results['misclassifications']
        
        # Training: dict with 'original_dataset' and optionally 'recommendation_dataset'
        if 'training' in misclass and isinstance(misclass['training'], dict):
            for category in ['original_dataset', 'recommendation_dataset']:
                if category in misclass['training']:
                    predictions.extend(misclass['training'][category])
        
        # Validation and testing: direct lists
        for split in ['validation', 'testing']:
            if split in misclass and isinstance(misclass[split], list):
                predictions.extend(misclass[split])
    
    if not predictions:
        print("[!] No misclassifications found in CNN results")
        return
    
    with onto:
        created_count = 0
        for pred in predictions:
            filename = pred.get('filename', '')
            if not filename:
                continue
            
            # Remove _from_json suffix if present (CNN adds this during prediction)
            filename_base = filename.replace('_from_json.png', '.png').replace('.png', '').replace('.json', '')
            
            # Convert filename to image ID format (same as create_image_from_scenAIro_json)
            image_id = filename_base.replace('.', '_').replace('-', '_')
            
            # Find corresponding Image individual
            img = onto.search_one(iri=f"*img_{image_id}")
            if not img:
                # Try without img_ prefix
                img = onto.search_one(iri=f"*{image_id}")
            
            if not img:
                # Debug: show what we're looking for
                if created_count == 0:
                    print(f"  \u26a0 Sample lookup failed: {filename} -> img_{image_id}")
                continue
            
            # Create Output individual (link misclassification to image)
            output_name = f"output_{image_id}_{cnn_instance.name}"
            output = Output(output_name)
            output.isSource = [img]
            output.isCalculatedBy = [cnn_instance]
            
            # Convert string prediction to object property
            pred_value = pred.get('predicted', 'unknown')
            if pred_value == 'runway':
                output.hasPredictedValue = onto.with_runway
            elif pred_value == 'norunway':
                output.hasPredictedValue = onto.no_runway
            
            output.hasConfidenceScore = float(pred.get('confidence', 0.0))
            created_count += 1
        
        print(f"  \u2713 Created {created_count} Output individuals")
        return created_count

def create_cnn_individual(iteration_num, cnn_metrics):
    with onto:
        # Iteration 1 uses Base model, later iterations use Rec naming
        if iteration_num == 1:
            cnn_name = "CNN_RunwayDetector_Base"
        else:
            cnn_name = f"CNN_RunwayDetector_Rec{iteration_num - 1}"
        
        cnn = CNN(cnn_name)
        
        # Performance metrics
        train_acc = float(cnn_metrics.get('train_accuracy', 0.0))
        val_acc = float(cnn_metrics.get('val_accuracy', 0.0))
        test_acc = float(cnn_metrics.get('test_accuracy', 0.0))
        train_loss = float(cnn_metrics.get('train_loss', 0.0))
        val_loss = float(cnn_metrics.get('val_loss', 0.0))
        test_loss = float(cnn_metrics.get('test_loss', 0.0))

        cnn.hasTrainingAccuracy = train_acc
        cnn.hasValidationAccuracy = val_acc
        cnn.hasTestingAccuracy = test_acc
        cnn.hasTrainingLoss = train_loss
        cnn.hasValidationLoss = val_loss
        cnn.hasTestingLoss = test_loss
        
        # Accuracy gap (overfitting indicator)
        accuracy_gap = float(train_acc - test_acc)
        cnn.hasAccuracyGap = accuracy_gap
        if train_acc > 0.95 and test_acc < 0.80:
            cnn.hasOverfittingIssue = [onto.overfitting_issue]
        # Priority: optimal > stagnation > needs_improvement.
        if test_acc >= 1.0:
            cnn.hasTrainingState = [onto.optimal_state]
        else:
            cnn.hasTrainingState = [onto.needs_improvement_state]
        
        # Improvement tracking (if previous iteration exists)
        if iteration_num > 1:
            prev_acc = float(cnn_metrics.get('previous_test_accuracy', test_acc))
            cnn.previousAccuracy = float(prev_acc)
            improvement = float(test_acc - prev_acc)
            cnn.accuracyImprovement = improvement

            if test_acc < 1.0 and improvement <= 0:
                cnn.hasTrainingState = [onto.stagnation_state]
        
        return cnn

def populate_from_json_directory(json_dir, cnn_instance, split='test', is_recommendation=False):
    import glob
    if not os.path.exists(json_dir):
        print(f"[!] Directory not found: {json_dir}")
        return 0
    
    json_files = []
    for label_dir in ['norunway', 'runway']:
        label_path = os.path.join(json_dir, label_dir)
        if os.path.exists(label_path):
            json_files.extend(glob.glob(os.path.join(label_path, "*.json")))
    
    split_label = "REC" if is_recommendation else split.upper()
    print(f"\n{split_label} Set: Found {len(json_files)} JSON files")
    
    success_count = 0
    for json_path in json_files:
        try:
            img = create_image_from_scenAIro_json(json_path, cnn_instance, split, is_recommendation)
            if success_count < 3:  # Show first 3 only
                print(f"  [+] {img.name}")
            success_count += 1
        except Exception as e:
            print(f"  [!] Error parsing {os.path.basename(json_path)}: {e}")
    
    if success_count > 3:
        print(f"  ... and {success_count - 3} more images")
    
    return success_count

def main(iteration_num, seed_suffix=''):
    global onto
    print("="*80)
    print(f"OSCAR ONTOLOGY - Iteration {iteration_num}")
    print("="*80)
    
    # Extract seed value from seed_suffix (e.g., "_seed61" -> 61)
    seed_value = seed_suffix.replace('_seed', '') if seed_suffix else '42'
    
    if iteration_num == 1:
        onto = get_ontology("http://example.org/OSCAR.owl")
        print(">> New ontology")
    else:
        prev_owl_path = os.path.join(OWL_OUTPUT_DIR, f"OSCAR{iteration_num-1}{seed_suffix}.owl")
        if os.path.exists(prev_owl_path):
            onto = get_ontology(prev_owl_path).load()
            print(f">> Loaded previous ontology: {os.path.basename(prev_owl_path)}")
            print(f"  - Existing CNNs: {len(list(onto.CNN.instances()))}")
            print(f"  - Existing Images: {len(list(onto.Image.instances()))}")
        else:
            print(f"!! Previous ontology not found: {prev_owl_path}")
            print("  Creating new ontology instead")
            onto = get_ontology("http://example.org/OSCAR.owl")
    
    # New structure: OSCAR_Experiments/seed_XX/Ontology/results/
    CNN_INPUT_DIR = os.path.join(EXPERIMENTS_DIR, f"seed_{seed_value}", "Ontology", "results")
    
    cnn_result_candidates = [
        os.path.join(CNN_INPUT_DIR, f"CNN_with_Rec{iteration_num-1}_seed{seed_value}.json") if iteration_num > 1 else None,
        os.path.join(CNN_INPUT_DIR, f"CNN_Base_seed{seed_value}.json"),
        os.path.join(CNN_INPUT_DIR, f"cnn_results_iter{iteration_num}_seed{seed_value}.json"),
        os.path.join(CNN_INPUT_DIR, f"CNN_Rec{iteration_num}_seed{seed_value}.json")
    ]
    cnn_result_candidates = [c for c in cnn_result_candidates if c is not None]
    
    cnn_result_path = None
    for path in cnn_result_candidates:
        if os.path.exists(path):
            cnn_result_path = path
            break
    
    if cnn_result_path:
        with open(cnn_result_path, 'r', encoding='utf-8') as f:
            cnn_data = json.load(f)
        
        if 'performance_metrics' in cnn_data:
            perf = cnn_data['performance_metrics']
            cnn_metrics = {
                'train_accuracy': perf['training']['accuracy'],
                'val_accuracy': perf['validation']['accuracy'],
                'test_accuracy': perf['testing']['accuracy'],
                'train_loss': perf['training']['loss'],
                'val_loss': perf['validation']['loss'],
                'test_loss': perf['testing']['loss']
            }
        else:
            cnn_metrics = cnn_data.get('metrics', {})
        print(f"[+] CNN results: {os.path.basename(cnn_result_path)}")
    else:
        print(f"[!] CNN results not found")
        cnn_metrics = {'train_accuracy': 0.85, 'val_accuracy': 0.80, 'test_accuracy': 0.75,
                       'train_loss': 0.3, 'val_loss': 0.4, 'test_loss': 0.5}
    
    cnn_instance = create_cnn_individual(iteration_num, cnn_metrics)
    print(f"[+] CNN: {cnn_instance.name} | Test Acc: {cnn_instance.hasTestingAccuracy:.2%} | Gap: {cnn_instance.hasAccuracyGap:.2%}")
    
    total_images = 0
    if iteration_num == 1:
        image_base = os.path.join(INPUT_BASE_DIR, "input_image")
        for split in ['train', 'val', 'test']:
            total_images += populate_from_json_directory(os.path.join(image_base, split), cnn_instance, split)
    else:
        print(f"\n[+] Reusing images from previous iteration")
        
        # Link existing training images to new CNN
        for img in onto.Image.instances():
            if hasattr(img, 'isTrainedBy') and cnn_instance not in img.isTrainedBy:
                img.isTrainedBy.append(cnn_instance)
            if hasattr(img, 'isValidatedBy') and cnn_instance not in img.isValidatedBy:
                img.isValidatedBy.append(cnn_instance)
            if hasattr(img, 'isTestedBy') and cnn_instance not in img.isTestedBy:
                img.isTestedBy.append(cnn_instance)
        for rec_img in onto.Rec_Training_Image.instances():
            if hasattr(rec_img, 'isTrainedBy') and cnn_instance not in rec_img.isTrainedBy:
                rec_img.isTrainedBy.append(cnn_instance)
        total_images = len(list(onto.Image.instances())) + len(list(onto.Rec_Training_Image.instances()))
    
    # Load NEW recommendation images from output_image (iteration 2+)
    # These are the recommendations generated in the PREVIOUS iteration
    if iteration_num > 1:
        # New structure: OSCAR_Experiments/seed_XX/Ontology/datasets/
        output_image_base = os.path.join(EXPERIMENTS_DIR, f"seed_{seed_value}", "Ontology", "datasets")
        rec_folder_name = f"Rec{iteration_num-1}-scenairo-seed{seed_value}"
        rec_folder_path = os.path.join(output_image_base, rec_folder_name)
        if os.path.exists(rec_folder_path):
            print(f"\n Loading {rec_folder_name}")
            total_images += populate_from_json_directory(rec_folder_path, cnn_instance, 'train', is_recommendation=True)
        else:
            print(f"\n[!] Not found: {rec_folder_path}")
    
    predictions_added = False
    if cnn_result_path and os.path.exists(cnn_result_path):
        try:
            with open(cnn_result_path, 'r', encoding='utf-8') as f:
                cnn_full_data = json.load(f)
            
            # Check if misclassifications exist in the CNN_Base format
            has_predictions = False
            if 'misclassifications' in cnn_full_data:
                misclass = cnn_full_data['misclassifications']
                
                # Check training (has original_dataset key)
                if 'training' in misclass and isinstance(misclass['training'], dict):
                    has_predictions = len(misclass['training'].get('original_dataset', [])) > 0
                if not has_predictions:
                    for split in ['validation', 'testing']:
                        if split in misclass and len(misclass[split]) > 0:
                            has_predictions = True
                            break
            
            if has_predictions:
                print(f"\n Populating predictions...")
                count = populate_cnn_results(cnn_full_data, cnn_instance)
                predictions_added = (count > 0)
            else:
                print(f"\n[!] No misclassifications found")
        except Exception as e:
            print(f"\n[!] Could not load predictions: {e}")
    else:
        print(f"\n[!] CNN results not found")
    
    owl_dir = os.path.join(SCRIPT_DIR, "Ontology_Owl_files")
    os.makedirs(owl_dir, exist_ok=True)
    output_path = os.path.join(owl_dir, f"OSCAR{iteration_num}{seed_suffix}.owl")
    onto.save(file=output_path, format="rdfxml")
    
    print(f"\n{'='*80}")
    print(f"[+] Saved: {output_path}")
    print(f"  Individuals: {len(list(onto.individuals()))}, Images: {len(list(onto.Image.instances()))}, Outputs: {len(list(onto.Output.instances()))}, CNNs: {len(list(onto.CNN.instances()))}")
    print("="*80)
    
    if not predictions_added and iteration_num == 1:
        print("\\n[INFO] Next: Train CNN, generate predictions, re-run Management.py")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OSCAR Ontology Management')
    parser.add_argument('--iteration', type=int, default=1, help='Iteration number')
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated seed numbers (e.g., 12,42,88)')
    parser.add_argument('--skip-population', action='store_true', help='Only create empty ontology')
    args = parser.parse_args()
    
    if args.skip_population:
        output_file = os.path.join(OWL_OUTPUT_DIR, "OSCAR.owl")
        onto.save(file=output_file, format="rdfxml")
        print(f"[+] Empty ontology saved: {output_file}")
    else:
        seeds = [s.strip() for s in args.seeds.split(',')]
        for seed in seeds:
            seed_suffix = f"_seed{seed}"
            print(f"\n{'#'*80}")
            print(f"# Processing seed: {seed}")
            print(f"{'#'*80}")
            main(iteration_num=args.iteration, seed_suffix=seed_suffix)
