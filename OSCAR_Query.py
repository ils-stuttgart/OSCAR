"""
OSCAR - Query Library (OntoLoop Mapping)
=======================================
Usage:
    # Iteration 1 - Initial CNN training
    python OSCAR_Query.py --iteration 1 --seed 42
    
    # Iteration 2 - Cumulative retraining with Rec1 data
    python OSCAR_Query.py --iteration 2 --seed 42
    
Outputs query_result/querying_result{N}_seed{X}.json with selected queries.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime as dt

from owlready2 import *  
import owlready2.reasoning

# Try importing tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

owlready2.reasoning.JAVA_MEMORY = 15000  # MB for Pellet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OWL_DIR = os.path.join(SCRIPT_DIR, "Ontology_OWL_files")
QUERY_RESULT_DIR = os.path.join(SCRIPT_DIR, "query_result")


MINIMAL_QUERIES = [
    "Q4_Detailed_Failures_in_TestDataset",
    "Q8_OOD_Combinations_NotInTraining",
    "Q13_CNN_1D_Weaknesses",
    "Q14_CNN_Cooccurrence_Weaknesses",
    "Q21_Critical_Errors_HighConfidence",
    "Q15_Final_Stopping_Decision",
    "Q16_Performance_Comparison_AllCNNs",
    "Q24_Training_Dataset_Composition",
]


def get_queries(target_cnn_name):
    prefix = "PREFIX : <http://example.org/OSCAR.owl#>"
    cnn_filter = f'FILTER(STRENDS(STR(?cnn), "{target_cnn_name}"))'

    return {
        # -------------------------------------------------------------
        # Failure analysis
        # -------------------------------------------------------------
        "Q4_Detailed_Failures_in_TestDataset": f"""
            {prefix}
            SELECT ?img ?runwayLabel ?airportCode ?timeLabel ?groundTruthLabel ?predictedLabel ?confidence
            WHERE {{
                ?cnn a :CNN ; :hasFailureOn ?img .
                {cnn_filter}
                ?img a :Image ;
                     :isTestedBy ?cnn ;
                     :hasGroundTruth ?gt .
                OPTIONAL {{ ?img :imageRunwayLabel ?runwayLabel }}
                OPTIONAL {{ ?img :imageAirportCode ?airportCode }}
                OPTIONAL {{ ?img :imageTimeLabel ?timeLabel }}
                ?out a :Output ;
                     :isSource ?img ;
                     :isCalculatedBy ?cnn ;
                     :hasPredictedValue ?pred ;
                     :hasConfidenceScore ?confidence .
                BIND(IF(STRENDS(STR(?gt), "with_runway"), "runway", "norunway") AS ?groundTruthLabel)
                BIND(IF(STRENDS(STR(?pred), "with_runway"), "runway", "norunway") AS ?predictedLabel)
            }}
            ORDER BY DESC(?confidence)
        """,
        # -------------------------------------------------------------
        # Tier 2: Critical Errors (High Confidence Failures)
        # -------------------------------------------------------------
        "Q21_Critical_Errors_HighConfidence": f"""
            {prefix}
            SELECT ?img ?runwayLabel ?airportCode ?timeLabel ?groundTruthLabel ?predictedLabel ?confidenceScore
                   (IF(?confidenceScore > 0.7, "CRITICAL", 
                       IF(?confidenceScore > 0.5, "MODERATE", "BORDERLINE")) AS ?errorSeverity)
            WHERE {{
                ?cnn a :CNN ; :hasFailureOn ?img .
                {cnn_filter}
                ?img a :Image ; 
                     :isTestedBy ?cnn ;
                     :hasGroundTruth ?gt .
                OPTIONAL {{ ?img :imageRunwayLabel ?runwayLabel }}
                OPTIONAL {{ ?img :imageAirportCode ?airportCode }}
                OPTIONAL {{ ?img :imageTimeLabel ?timeLabel }}
                ?output a :Output ; 
                        :isSource ?img ; 
                        :isCalculatedBy ?cnn ; 
                        :hasPredictedValue ?pred ; 
                        :hasConfidenceScore ?confidenceScore .
                BIND(IF(STRENDS(STR(?gt), "with_runway"), "runway", "norunway") AS ?groundTruthLabel)
                BIND(IF(STRENDS(STR(?pred), "with_runway"), "runway", "norunway") AS ?predictedLabel)
                FILTER(?confidenceScore > 0.5)
            }}
            ORDER BY DESC(?confidenceScore)
        """,
        # -------------------------------------------------------------
        # Tier 4: OOD combos present in test but absent in training
        # -------------------------------------------------------------
        "Q8_OOD_Combinations_NotInTraining": f"""
            {prefix}
            SELECT ?runway ?airport ?time (COUNT(DISTINCT ?testImg) AS ?count)
            WHERE {{
                ?cnn a :CNN . {cnn_filter}
                ?testImg a :Image ; :isTestedBy ?cnn ;
                         :hasRunway ?runway ; :hasAirport ?airport ; :hasTimeOfDay ?time .
                FILTER NOT EXISTS {{
                    ?trainImg a :Image ; :isTrainedBy ?cnn ;
                              :hasRunway ?runway ; :hasAirport ?airport ; :hasTimeOfDay ?time .
                }}
            }}
            GROUP BY ?runway ?airport ?time
            HAVING (?count > 0)
            ORDER BY DESC(?count)
        """,
        # -------------------------------------------------------------
        # Tier 3: 1D weaknesses
        # -------------------------------------------------------------
        "Q13_CNN_1D_Weaknesses": f"""
            {prefix}
            SELECT DISTINCT ?weakness
            WHERE {{
                ?cnn a :CNN ; :has1DWeakness ?weakness .
                {cnn_filter}
            }}
            ORDER BY ?weakness
        """,
        # -------------------------------------------------------------
        # Tier 1: Co-occurrence weaknesses
        # -------------------------------------------------------------
        "Q14_CNN_Cooccurrence_Weaknesses": f"""
            {prefix}
            SELECT DISTINCT ?cooccurrenceWeakness
            WHERE {{
                ?cnn a :CNN ; :hasCooccurrenceWeakness ?cooccurrenceWeakness .
                {cnn_filter}
            }}
            ORDER BY ?cooccurrenceWeakness
        """,
        # -------------------------------------------------------------
        # Loop control
        # -------------------------------------------------------------
        "Q15_Final_Stopping_Decision": f"""
            {prefix}
            SELECT ?cnn
                   (IF(BOUND(?state),
                        IF(STRENDS(STR(?state), "optimal_state"), "STOP_EXCELLENCE", "CONTINUE"),
                        IF(?testAcc >= 1.0, "STOP_EXCELLENCE", "CONTINUE")) AS ?finalDecision)
                   (IF(BOUND(?state),
                        CONCAT("TrainingState=", STRAFTER(STR(?state), "#")),
                        CONCAT("FallbackTestAcc=", STR(?testAcc))) AS ?reason)
            WHERE {{
                ?cnn a :CNN .
                {cnn_filter}
                OPTIONAL {{ ?cnn :hasTrainingState ?state . }}
                ?cnn :hasTestingAccuracy ?testAcc .
            }}
            LIMIT 1
        """,

        # -------------------------------------------------------------
        # Performance comparison
        # -------------------------------------------------------------
        "Q16_Performance_Comparison_AllCNNs": f"""
            {prefix}
            SELECT ?cnn
                   ?testingAccuracy ?validationAccuracy ?trainingAccuracy
                   ?accuracyGap
                   (IF(?testingAccuracy > ?trainingAccuracy, "GOOD", "POTENTIAL_OVERFIT") AS ?generalizationStatus)
            WHERE {{
                ?cnn a :CNN .
                OPTIONAL {{ ?cnn :hasTestingAccuracy ?testingAccuracy }}
                OPTIONAL {{ ?cnn :hasValidationAccuracy ?validationAccuracy }}
                OPTIONAL {{ ?cnn :hasTrainingAccuracy ?trainingAccuracy }}
                OPTIONAL {{ ?cnn :hasAccuracyGap ?accuracyGap }}
            }}
            ORDER BY DESC(?testingAccuracy)
        """,

        # -------------------------------------------------------------
        # Training dataset composition
        # -------------------------------------------------------------
        "Q24_Training_Dataset_Composition": f"""
            {prefix}
            SELECT
                (COUNT(DISTINCT ?origImg) AS ?originalCount)
                (COUNT(DISTINCT ?recImg) AS ?recCount)
                (COUNT(DISTINCT ?anyImg) AS ?totalCount)
            WHERE {{
                ?cnn a :CNN .
                {cnn_filter}
                OPTIONAL {{
                    ?origImg a :Image ; :isTrainedBy ?cnn .
                    FILTER NOT EXISTS {{ ?origImg a :Rec_Training_Image }}
                }}
                OPTIONAL {{ ?recImg a :Rec_Training_Image ; :isTrainedBy ?cnn }}
                OPTIONAL {{
                    {{ ?anyImg a :Image }} UNION {{ ?anyImg a :Rec_Training_Image }} .
                    ?anyImg :isTrainedBy ?cnn
                }}
            }}
        """,

    }


def _execute_queries(ontology, query_plan, label):
    print(f"\nExecuting {label} ({len(query_plan)} queries)")
    results = {}
    iterator = tqdm(query_plan, desc="queries", ncols=80) if TQDM_AVAILABLE else query_plan

    for name in iterator:
        try:
            q_str = get_queries.cache[name]  # type: ignore[attr-defined]
        except KeyError:
            print(f"  ⚠ {name}: not found")
            results[name] = []
            continue
        try:
            res = list(default_world.sparql(q_str))
            formatted = [
                [item.name if hasattr(item, "name") else str(item) for item in row]
                for row in res
            ]
            results[name] = formatted
            if not TQDM_AVAILABLE:
                print(f"  ✓ {name}: {len(res)} result(s)")
        except Exception as exc:  # noqa: BLE001
            if not TQDM_AVAILABLE:
                print(f"  ✗ {name}: {exc}")
            results[name] = []

    return results


def run_queries(onto, target_cnn_name, mode):
    # Build and cache query strings for this CNN
    get_queries.cache = get_queries(target_cnn_name)  # type: ignore[attr-defined]

    if mode == "ALL":
        plan = list(get_queries.cache.keys())
        label = "ALL"
    elif mode == "MINIMAL":
        plan = MINIMAL_QUERIES
        label = "MINIMAL"
    elif isinstance(mode, list):
        plan = mode
        label = "CUSTOM"
    else:
        plan = MINIMAL_QUERIES
        label = "MINIMAL (default)"

    return _execute_queries(onto, plan, label), plan


def main(iteration_num=1, seed_suffix="_seed42", query_mode="MINIMAL"):
    print("\n" + "=" * 70)
    print(f"OSCAR - Query Execution (Iteration {iteration_num})")
    print("=" * 70)

    # Iteration 1 uses Base model, later iterations use Rec naming
    if iteration_num == 1:
        target_cnn_name = "CNN_RunwayDetector_Base"
    else:
        target_cnn_name = f"CNN_RunwayDetector_Rec{iteration_num - 1}"
    
    owl_path = os.path.join(OWL_DIR, f"OSCAR{iteration_num}{seed_suffix}_with_rules.owl")

    if not os.path.exists(owl_path):
        print(f"✗ Ontology not found: {owl_path}")
        print("  Run OSCAR_Rule.py first to produce the _with_rules OWL file.")
        sys.exit(1)

    onto = get_ontology(owl_path).load()
    print(f"✓ Loaded: {owl_path}")

    t0 = time.time()
    with onto:
        sync_reasoner_pellet(infer_property_values=True)
    reasoning_time = time.time() - t0
    print(f"Reasoning completed in {reasoning_time:.2f}s")

    results, executed_queries = run_queries(onto, target_cnn_name, query_mode)

    os.makedirs(QUERY_RESULT_DIR, exist_ok=True)
    filepath = os.path.join(
        QUERY_RESULT_DIR, f"querying_result{iteration_num}{seed_suffix}.json"
    )
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "iteration": iteration_num,
                    "target_model": target_cnn_name,
                    "timestamp": dt.now().isoformat(),
                    "reasoning_time_sec": round(reasoning_time, 2),
                    "query_execution_mode": query_mode if isinstance(query_mode, str) else "CUSTOM",
                    "executed_queries": executed_queries,
                    "total_queries_executed": len(executed_queries),
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Saved results to: {filepath}")
    print(f"   Mode: {query_mode if isinstance(query_mode, str) else 'CUSTOM'}")
    print(f"   Executed: {len(executed_queries)} queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OSCAR SPARQL queries")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number")
    parser.add_argument("--seeds", type=str, default='42', help='Comma-separated seed numbers (e.g., 12,42,88)')
    parser.add_argument(
        "--mode",
        type=str,
        default="MINIMAL",
        help='Query execution mode: "ALL", "MINIMAL", or comma-separated names',
    )
    args = parser.parse_args()
    
    seeds = [s.strip() for s in args.seeds.split(',')]

    mode = (
        args.mode
        if args.mode in {"ALL", "MINIMAL"}
        else [name.strip() for name in args.mode.split(",") if name.strip()]
    )
    
    for seed in seeds:
        seed_suffix = f"_seed{seed}"
        print(f"\n{'#'*80}")
        print(f"# Processing seed: {seed}")
        print(f"{'#'*80}")
        main(iteration_num=args.iteration, seed_suffix=seed_suffix, query_mode=mode)