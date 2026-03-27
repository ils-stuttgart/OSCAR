"""
OSCAR - SWRL Rules Module (Simplified 3D ODD)
================================================================================
Usage:
    # Iteration 1 - Initial CNN training
    python OSCAR_Rule.py --iteration 1 --seed 42
    # Iteration 2 - Cumulative retraining with Rec1 data
    python OSCAR_Rule.py --iteration 2 --seed 42
================================================================================
"""

from owlready2 import *
import os
import sys

def define_swrl_rules(onto):
    """Define all SWRL rules for weakness detection in runway identification."""
    
    print("\n" + "="*80)
    print("SWRL RULES - Defining 44 Rules for 3D ODD (Runway × Airport × Time)")
    print("="*80)
    
    with onto:
        # ====================================================================
        # RULE 1: Misclassification Detection (PRIMARY RULE)
        # ====================================================================
        rule_1 = Imp()
        rule_1.label = ["R1-Misclassification_Detection"]
        rule_1.comment = ["Detect all misclassified images: groundTruth != predictedValue"]
        rule_1.set_as_rule("""
            CNN(?cnn),
            Image(?img),
            Output(?out),
            isCalculatedBy(?out, ?cnn),
            isSource(?out, ?img),
            hasGroundTruth(?img, ?gt),
            hasPredictedValue(?out, ?pred),
            differentFrom(?gt, ?pred)
            -> hasFailureOn(?cnn, ?img)
        """)
        print("✓ R1: Misclassification Detection (Primary - uses differentFrom)")
        
        # ====================================================================
        # RULES 2a-2b: ODD1 - Runway Classification Weaknesses
        # ====================================================================
        rule_2a = Imp()
        rule_2a.label = ["R2a-PoorOnWithRunway_Weakness"]
        rule_2a.comment = ["Detect false negatives: CNN misses runways"]
        rule_2a.set_as_rule("""
            CNN(?cnn),
            Image(?img),
            hasFailureOn(?cnn, ?img),
            hasRunway(?img, with_runway)
            -> has1DWeakness(?cnn, PoorOnWithRunway_Weakness)
        """)
        print("✓ R2a: PoorOnWithRunway (False Negatives)")
        
        rule_2b = Imp()
        rule_2b.label = ["R2b-PoorOnNoRunway_Weakness"]
        rule_2b.comment = ["Detect false positives: CNN hallucinates runways"]
        rule_2b.set_as_rule("""
            CNN(?cnn),
            Image(?img),
            hasFailureOn(?cnn, ?img),
            hasRunway(?img, no_runway)
            -> has1DWeakness(?cnn, PoorOnNoRunway_Weakness)
        """)
        print("✓ R2b: PoorOnNoRunway (False Positives)")
        
        # ====================================================================
        # RULES 3a-3g: ODD2 - Airport Weaknesses (7 airports)
        # ====================================================================
        print("\n--- ODD2: Airport Weaknesses (7 rules) ---")
        
        rule_3a = Imp()
        rule_3a.label = ["R3a-PoorOnEDDSAirport_Weakness"]
        rule_3a.set_as_rule("""
            CNN(?cnn), Image(?img),
            hasFailureOn(?cnn, ?img),
            hasAirport(?img, edds_airport)
            -> has1DWeakness(?cnn, PoorOnEDDSAirport_Weakness)
        """)
        print("✓ R3a: PoorOnEDDS (Stuttgart)")
        
        rule_3b = Imp()
        rule_3b.label = ["R3b-PoorOnEDDVAirport_Weakness"]
        rule_3b.set_as_rule("""
            CNN(?cnn), Image(?img),
            hasFailureOn(?cnn, ?img),
            hasAirport(?img, eddv_airport)
            -> has1DWeakness(?cnn, PoorOnEDDVAirport_Weakness)
        """)
        print("✓ R3b: PoorOnEDDV (Hanover)")
        
        rule_3c = Imp()
        rule_3c.label = ["R3c-PoorOnEDNYAirport_Weakness"]
        rule_3c.set_as_rule("""
            CNN(?cnn), Image(?img),
            hasFailureOn(?cnn, ?img),
            hasAirport(?img, edny_airport)
            -> has1DWeakness(?cnn, PoorOnEDNYAirport_Weakness)
        """)
        print("✓ R3c: PoorOnEDNY (Friedrichshafen)")
        
        rule_3d = Imp()
        rule_3d.label = ["R3d-PoorOnEDSBAirport_Weakness"]
        rule_3d.set_as_rule("""
            CNN(?cnn), Image(?img),
            hasFailureOn(?cnn, ?img),
            hasAirport(?img, edsb_airport)
            -> has1DWeakness(?cnn, PoorOnEDSBAirport_Weakness)
        """)
        print("✓ R3d: PoorOnEDSB (Karlsruhe)")
        
        rule_3e = Imp()
        rule_3e.label = ["R3e-PoorOnELLXAirport_Weakness"]
        rule_3e.set_as_rule("""
            CNN(?cnn), Image(?img),
            hasFailureOn(?cnn, ?img),
            hasAirport(?img, ellx_airport)
            -> has1DWeakness(?cnn, PoorOnELLXAirport_Weakness)
        """)
        print("✓ R3e: PoorOnELLX (Luxembourg)")
        
        rule_3f = Imp()
        rule_3f.label = ["R3f-PoorOnENBRAirport_Weakness"]
        rule_3f.set_as_rule("""
            CNN(?cnn), Image(?img),
            hasFailureOn(?cnn, ?img),
            hasAirport(?img, enbr_airport)
            -> has1DWeakness(?cnn, PoorOnENBRAirport_Weakness)
        """)
        print("✓ R3f: PoorOnENBR (Bergen)")
        
        rule_3g = Imp()
        rule_3g.label = ["R3g-PoorOnKLAXAirport_Weakness"]
        rule_3g.set_as_rule("""
            CNN(?cnn), Image(?img),
            hasFailureOn(?cnn, ?img),
            hasAirport(?img, klax_airport)
            -> has1DWeakness(?cnn, PoorOnKLAXAirport_Weakness)
        """)
        print("✓ R3g: PoorOnKLAX (Los Angeles)")
        
        # ====================================================================
        # RULES 4a-4b: ODD3 - Time of Day Weaknesses
        # ====================================================================
        print("\n--- ODD3: Time of Day Weaknesses (2 rules) ---")
        
        rule_4a = Imp()
        rule_4a.label = ["R4a-PoorOnDaytime_Weakness"]
        rule_4a.set_as_rule("""
            CNN(?cnn), Image(?img),
            hasFailureOn(?cnn, ?img),
            hasTimeOfDay(?img, day_time)
            -> has1DWeakness(?cnn, PoorOnDaytime_Weakness)
        """)
        print("✓ R4a: PoorOnDaytime")
        
        rule_4b = Imp()
        rule_4b.label = ["R4b-PoorOnNighttime_Weakness"]
        rule_4b.set_as_rule("""
            CNN(?cnn), Image(?img),
            hasFailureOn(?cnn, ?img),
            hasTimeOfDay(?img, night_time)
            -> has1DWeakness(?cnn, PoorOnNighttime_Weakness)
        """)
        print("✓ R4b: PoorOnNighttime")
        
        # ====================================================================
        # RULES 5: Co-occurrence Weaknesses (28 rules: 2×7×2)
        # Detect failures at specific combinations of Runway × Airport × Time
        # ====================================================================
        print("\n--- R5: Co-occurrence Weaknesses (28 rules) ---")
        
        airports = [
            ("edds_airport", "EDDS"),
            ("eddv_airport", "EDDV"),
            ("edny_airport", "EDNY"),
            ("edsb_airport", "EDSB"),
            ("ellx_airport", "ELLX"),
            ("enbr_airport", "ENBR"),
            ("klax_airport", "KLAX")
        ]
        
        rule_num = 0
        for airport_inst, airport_code in airports:
            for time_inst, time_label, time_name in [("day_time", "Daytime", "day"), ("night_time", "Nighttime", "night")]:
                rule_num += 1
                rule = Imp()
                weakness_name = f"PoorOnRunway{airport_code}{time_label}_Weakness"
                rule.label = [f"R5{chr(96+rule_num)}-{weakness_name}"]
                rule.set_as_rule(f"""
                    CNN(?cnn), Image(?img),
                    hasFailureOn(?cnn, ?img),
                    hasRunway(?img, with_runway),
                    hasAirport(?img, {airport_inst}),
                    hasTimeOfDay(?img, {time_inst})
                    -> hasCooccurrenceWeakness(?cnn, {weakness_name})
                """)
                print(f"✓ R5{chr(96+rule_num)}: Runway+{airport_code}+{time_label}")

        for airport_inst, airport_code in airports:
            for time_inst, time_label, time_name in [("day_time", "Daytime", "day"), ("night_time", "Nighttime", "night")]:
                rule_num += 1
                rule = Imp()
                weakness_name = f"PoorOnNoRunway{airport_code}{time_label}_Weakness"
                rule.label = [f"R5{chr(96+rule_num)}-{weakness_name}"]
                rule.set_as_rule(f"""
                    CNN(?cnn), Image(?img),
                    hasFailureOn(?cnn, ?img),
                    hasRunway(?img, no_runway),
                    hasAirport(?img, {airport_inst}),
                    hasTimeOfDay(?img, {time_inst})
                    -> hasCooccurrenceWeakness(?cnn, {weakness_name})
                """)
                print(f"✓ R5{chr(96+rule_num)}: NoRunway+{airport_code}+{time_label}")
        
        # ====================================================================
        # RULES 6a-6d: Training State Detection (4 rules)
        # ====================================================================
        print("\n--- R6: Training State Detection (4 rules) ---")
        
        # R6a: Overfitting Detection
        rule_6a = Imp()
        rule_6a.label = ["R6a-Overfitting_Detection"]
        rule_6a.comment = ["Detect overfitting: high training accuracy but low test accuracy"]
        rule_6a.set_as_rule("""
            CNN(?cnn),
            hasTrainingAccuracy(?cnn, ?trainAcc),
            hasTestingAccuracy(?cnn, ?testAcc),
            greaterThan(?trainAcc, 0.95),
            lessThan(?testAcc, 0.80)
            -> hasOverfittingIssue(?cnn, Overfitting)
        """)
        print("✓ R6a: Overfitting Detection")
        
        # R6b: Optimal State Detection
        rule_6b = Imp()
        rule_6b.label = ["R6b-Optimal_State"]
        rule_6b.comment = ["CNN achieves optimal performance with high test accuracy"]
        rule_6b.set_as_rule("""
            CNN(?cnn),
            hasTestingAccuracy(?cnn, ?testAcc),
            greaterThanOrEqual(?testAcc, 0.95)
            -> hasTrainingState(?cnn, optimal_state)
        """)
        print("✓ R6b: Optimal State")
        
        # R6c: Low Testing Accuracy  
        rule_6c = Imp()
        rule_6c.label = ["R6c-NeedsImprovement_LowAccuracy"]
        rule_6c.comment = ["Low test accuracy requires more training"]
        rule_6c.set_as_rule("""
            CNN(?cnn),
            hasTestingAccuracy(?cnn, ?testAcc),
            lessThan(?testAcc, 0.85)
            -> hasTrainingState(?cnn, needs_improvement_state)
        """)
        print("✓ R6c: Needs Improvement (Low Accuracy)")
        
        # R6d: Overfitting Forces Continuation
        rule_6d = Imp()
        rule_6d.label = ["R6d-NeedsImprovement_Overfitting"]
        rule_6d.comment = ["Overfitting requires intervention"]
        rule_6d.set_as_rule("""
            CNN(?cnn),
            hasOverfittingIssue(?cnn, ?issue)
            -> hasTrainingState(?cnn, needs_improvement_state)
        """)
        print("✓ R6d: Needs Improvement (Overfitting)")
        
        # ====================================================================
        # Summary
        # ====================================================================
        print("\n" + "="*80)
        print("SWRL Rules Summary:")
        print("  - R1: Misclassification detection (1 rule)")
        print("  - R2a-2b: Runway weaknesses (2 rules)")
        print("  - R3a-3g: Airport weaknesses (7 rules)")
        print("  - R4a-4b: Time of Day weaknesses (2 rules)")
        print("  - R5: Co-occurrence weaknesses (28 rules)")
        print("  - R6a-6d: Training state detection (4 rules)")
        print("  Total: 44 SWRL rules for 3D ODD (Runway × Airport × Time)")
        print("="*80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Add SWRL rules to OSCAR ontology')
    parser.add_argument('--iteration', type=int, default=1,
                       help='Iteration number (e.g., 1, 2, 3...)')
    parser.add_argument('--seeds', type=str, default='42',
                       help='Comma-separated seed numbers (e.g., 12,42,88)')
    args = parser.parse_args()
    
    iteration_num = args.iteration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    owl_folder = os.path.join(script_dir, "Ontology_OWL_files")
    
    seeds = [s.strip() for s in args.seeds.split(',')]
    for seed in seeds:
        seed_suffix = f"_seed{seed}"
        base_path = os.path.join(owl_folder, f"OSCAR{iteration_num}{seed_suffix}.owl")      
        rules_path = os.path.join(owl_folder, f"OSCAR{iteration_num}{seed_suffix}_with_rules.owl")  
        
        print("\n" + "="*80)
        print(f"OSCAR - ADDING SWRL RULES - Iteration {iteration_num} - Seed {seed}")
        print("="*80)
        
        if not os.path.exists(base_path):
            print(f"✗ File not found: {base_path}")
            print("  Run OSCAR_Management.py first.")
            continue
        
        onto = get_ontology(base_path).load()
        print(f"✓ Loaded: {base_path}")
        
        # Clear existing rules
        existing_rules = list(onto.rules())
        if existing_rules:
            print(f"  Clearing {len(existing_rules)} existing rules...")
            for rule in existing_rules:
                destroy_entity(rule)
            print("  ✓ Rules cleared")

        # Save clean (no rules) - for Protégé
        onto.save(file=base_path, format="rdfxml")
        print(f"\n✓ Saved CLEAN: {base_path}")
        print("  → Use in Protégé\n")

        # Add rules
        define_swrl_rules(onto)
        
        # Save with rules - for Python reasoning
        onto.save(file=rules_path, format="rdfxml")
        print(f"\n✓ Saved WITH RULES: {rules_path}")
        print(f"  → Use for iteration {iteration_num+1}")