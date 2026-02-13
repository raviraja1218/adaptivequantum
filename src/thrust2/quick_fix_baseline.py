"""
Quick fix for baseline compilers to include photon metrics.
"""
import numpy as np
from src.thrust2.gate_fusion.baseline_compilers import (
    QiskitBaselineCompiler, PercevalBaselineCompiler
)
from src.thrust2.utils.benchmark_generator import load_benchmark

def fix_and_run_quick():
    """Quick fix and run for paper results."""
    print("Fixing baseline compilers and running quick benchmark...")
    
    # Load a circuit
    circuit = load_benchmark("deutsch_jozsa_5q")
    
    # Create compilers
    qiskit = QiskitBaselineCompiler(optimization_level=1)
    perceval = PercevalBaselineCompiler(optimization_level=2)
    
    print(f"\nCircuit: {circuit.component_count} gates")
    
    # Test Qiskit
    print(f"\nQiskit compilation:")
    optimized, metrics = qiskit.compile(circuit)
    
    # Add photon metrics
    original_photons = circuit.calculate_photon_loss()
    optimized_photons = optimized.calculate_photon_loss()
    photon_improvement = 100 * (optimized_photons - original_photons) / original_photons
    
    print(f"  Gates: {metrics['original_gates']} → {metrics['optimized_gates']}")
    print(f"  Reduction: {metrics['gate_reduction']:.1f}%")
    print(f"  Photon survival: {original_photons:.3f} → {optimized_photons:.3f}")
    print(f"  Photon improvement: {photon_improvement:.1f}%")
    
    # Save fixed metrics
    fixed_metrics = metrics.copy()
    fixed_metrics.update({
        'original_photons': original_photons,
        'optimized_photons': optimized_photons,
        'photon_improvement': photon_improvement
    })
    
    # Save to file for final benchmark
    import json
    from pathlib import Path
    
    output_dir = Path("experiments/thrust2/quick_fix")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "fixed_metrics.json", 'w') as f:
        json.dump(fixed_metrics, f, indent=2)
    
    print(f"\nFixed metrics saved to: {output_dir}/fixed_metrics.json")
    
    return True

if __name__ == "__main__":
    success = fix_and_run_quick()
    exit(0 if success else 1)
