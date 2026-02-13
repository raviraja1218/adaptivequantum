"""
Check Day 1 completion status.
"""
from pathlib import Path
import sys

def check_day1_completion():
    """Check if Day 1 deliverables are complete."""
    print("=" * 60)
    print("DAY 1 COMPLETION CHECK")
    print("=" * 60)
    
    requirements = {
        "Simulator verification": [
            "experiments/thrust2/validation/simulator_verification.txt"
        ],
        "Benchmark circuits": [
            "data/processed/benchmark_circuits/deutsch_jozsa_5q.pkl",
            "data/processed/benchmark_circuits/vqe_h2_10q.pkl", 
            "data/processed/benchmark_circuits/qaoa_maxcut_20q.pkl",
            "data/processed/benchmark_circuits/manifest.json"
        ],
        "Baseline benchmarks": [
            "experiments/thrust2/compilation_benchmarks/all_baselines.csv",
            "experiments/thrust2/compilation_benchmarks/baseline_summary.csv"
        ],
        "Source code": [
            "src/thrust2/photonic_simulator/__init__.py",
            "src/thrust2/photonic_simulator/components.py",
            "src/thrust2/photonic_simulator/circuit.py",
            "src/thrust2/utils/benchmark_generator.py",
            "src/thrust2/gate_fusion/baseline_compilers.py"
        ]
    }
    
    all_passed = True
    
    for category, files in requirements.items():
        print(f"\n{category}:")
        category_passed = True
        
        for file in files:
            path = Path(file)
            if path.exists():
                print(f"  ✅ {path}")
            else:
                print(f"  ❌ {path} - MISSING")
                category_passed = False
                all_passed = False
        
        if category_passed:
            print(f"  ✓ All files present")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ DAY 1 COMPLETE - READY FOR DAY 2 (RL COMPILER)")
        
        # Create completion marker
        completion_file = Path(".day1_complete")
        completion_file.write_text("Day 1 completed successfully\n")
        
        # Print next steps
        print("\nNEXT STEPS FOR DAY 2:")
        print("1. Implement RL environment (state, action, reward)")
        print("2. Create DQN agent")
        print("3. Train on benchmark circuits")
        print("4. Validate RL compilation")
        
    else:
        print("❌ DAY 1 INCOMPLETE - FIX MISSING FILES")
    
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = check_day1_completion()
    sys.exit(0 if success else 1)
