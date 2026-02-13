"""
Generate benchmark circuits for photonic compilation.
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Any
from src.thrust2.photonic_simulator.circuit import PhotonicCircuit
from src.thrust2.photonic_simulator.components import BeamSplitter, PhaseShifter

def generate_deutsch_jozsa_5q() -> PhotonicCircuit:
    """Generate 5-qubit Deutsch-Jozsa circuit (simplified)."""
    circuit = PhotonicCircuit(n_modes=10)  # 5 qubits × 2 modes each
    
    # Simplified Deutsch-Jozsa: alternating beam splitters and phase shifters
    for i in range(0, 10, 2):
        # Add beam splitter between consecutive modes
        circuit.add_beam_splitter(theta=np.pi/4, mode_indices=(i, i+1))
        
        # Add some phase shifters
        if i % 4 == 0:
            circuit.add_phase_shifter(phi=np.pi/3, mode_indices=(i, i+1))
    
    # Add some entangling operations (simplified)
    for i in range(0, 8, 2):
        circuit.add_beam_splitter(theta=np.pi/6, mode_indices=(i, i+2))
    
    print(f"Generated Deutsch-Jozsa circuit with {circuit.component_count} components")
    return circuit

def generate_vqe_h2_10q() -> PhotonicCircuit:
    """Generate 10-qubit VQE circuit for H₂ molecule (simplified)."""
    circuit = PhotonicCircuit(n_modes=20)  # 10 qubits
    
    # VQE ansatz: layers of single-qubit rotations and entangling gates
    n_layers = 3
    
    for layer in range(n_layers):
        # Single-qubit rotations (simulated as phase shifters)
        for i in range(0, 20, 2):
            angle = np.pi * (layer + 1) / (n_layers + 1)
            circuit.add_phase_shifter(phi=angle, mode_indices=(i, i+1))
        
        # Entangling gates (simulated as beam splitters)
        for i in range(0, 18, 2):
            if i % 4 == 0:  # Every other pair
                circuit.add_beam_splitter(theta=np.pi/4, mode_indices=(i, i+2))
    
    print(f"Generated VQE-H₂ circuit with {circuit.component_count} components")
    return circuit

def generate_qaoa_maxcut_20q() -> PhotonicCircuit:
    """Generate 20-qubit QAOA circuit for MaxCut (simplified)."""
    circuit = PhotonicCircuit(n_modes=40)  # 20 qubits
    
    # QAOA with p=2 (2 layers)
    p = 2
    
    for layer in range(p):
        # Mixer layer (beam splitters)
        for i in range(0, 40, 4):
            circuit.add_beam_splitter(theta=np.pi/3, mode_indices=(i, i+1))
        
        # Cost layer (phase shifters for edges)
        # Simplified: phase shifters on alternating modes
        for i in range(0, 38, 2):
            if (i // 2) % 3 == 0:  # Simulating graph edges
                circuit.add_phase_shifter(phi=np.pi/2, mode_indices=(i, i+2))
    
    print(f"Generated QAOA-MaxCut circuit with {circuit.component_count} components")
    return circuit

def save_benchmark_circuit(circuit: PhotonicCircuit, name: str):
    """Save benchmark circuit to file."""
    output_dir = Path("data/processed/benchmark_circuits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle
    output_path = output_dir / f"{name}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(circuit, f)
    
    # Also save metadata
    metadata = {
        "name": name,
        "n_modes": circuit.n_modes,
        "component_count": circuit.component_count,
        "gate_counts": circuit.count_gates(),
        "generated": np.datetime64('now')
    }
    
    metadata_path = output_dir / f"{name}_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.datetime64):
                return str(obj)
            return obj
        
        json.dump(metadata, f, default=convert, indent=2)
    
    print(f"Saved {name} to {output_path}")
    return output_path

def generate_all_benchmarks():
    """Generate all benchmark circuits."""
    print("Generating benchmark circuits...")
    print("-" * 40)
    
    benchmarks = {}
    
    # Generate circuits
    dj_circuit = generate_deutsch_jozsa_5q()
    vqe_circuit = generate_vqe_h2_10q()
    qaoa_circuit = generate_qaoa_maxcut_20q()
    
    # Save circuits
    benchmarks["deutsch_jozsa_5q"] = {
        "circuit": dj_circuit,
        "path": save_benchmark_circuit(dj_circuit, "deutsch_jozsa_5q")
    }
    
    benchmarks["vqe_h2_10q"] = {
        "circuit": vqe_circuit,
        "path": save_benchmark_circuit(vqe_circuit, "vqe_h2_10q")
    }
    
    benchmarks["qaoa_maxcut_20q"] = {
        "circuit": qaoa_circuit,
        "path": save_benchmark_circuit(qaoa_circuit, "qaoa_maxcut_20q")
    }
    
    # Create benchmark manifest
    manifest = {
        "benchmarks": list(benchmarks.keys()),
        "total_circuits": len(benchmarks),
        "generation_date": str(np.datetime64('now'))
    }
    
    manifest_path = Path("data/processed/benchmark_circuits/manifest.json")
    with open(manifest_path, 'w') as f:
        import json
        json.dump(manifest, f, indent=2)
    
    print("-" * 40)
    print(f"✅ Generated {len(benchmarks)} benchmark circuits")
    print(f"Manifest saved to: {manifest_path}")
    
    return benchmarks

def load_benchmark(name: str) -> PhotonicCircuit:
    """Load a benchmark circuit by name."""
    circuit_path = Path(f"data/processed/benchmark_circuits/{name}.pkl")
    
    if not circuit_path.exists():
        raise FileNotFoundError(f"Benchmark {name} not found at {circuit_path}")
    
    with open(circuit_path, 'rb') as f:
        circuit = pickle.load(f)
    
    return circuit

if __name__ == "__main__":
    benchmarks = generate_all_benchmarks()
    
    # Verify circuits can be loaded
    print("\nVerifying circuit loading...")
    for name in benchmarks.keys():
        try:
            circuit = load_benchmark(name)
            print(f"✓ {name}: {circuit.component_count} components")
        except Exception as e:
            print(f"✗ {name}: Failed to load - {e}")
