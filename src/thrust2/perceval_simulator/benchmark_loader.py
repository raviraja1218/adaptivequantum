#!/usr/bin/env python3
"""
Load and prepare benchmark circuits for photonic compilation
"""

import pickle
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, QAOAAnsatz
import networkx as nx
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perceval_simulator.circuit_converters.qiskit_to_perceval import QiskitToPercevalConverter

class BenchmarkLoader:
    """Load benchmark circuits for photonic compilation"""
    
    def __init__(self, noise_profile_dir=None):
        """
        Initialize benchmark loader
        
        Args:
            noise_profile_dir: Directory containing Phase 2 noise profiles
        """
        self.noise_profile_dir = noise_profile_dir
        self.converter = QiskitToPercevalConverter()
        
    def create_deutsch_jozsa(self, n_qubits=5):
        """
        Create Deutsch-Jozsa algorithm circuit
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            circuit: Qiskit QuantumCircuit
            metadata: Circuit metadata
        """
        print(f"Creating Deutsch-Jozsa circuit ({n_qubits} qubits)...")
        
        # Create balanced oracle (alternating 0s and 1s)
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Apply Hadamard to all qubits
        for i in range(n_qubits):
            qc.h(i)
        
        # Oracle: CNOTs with alternating pattern
        for i in range(n_qubits - 1):
            if i % 2 == 0:
                qc.cx(i, i + 1)
        
        # Apply Hadamard to all qubits again
        for i in range(n_qubits):
            qc.h(i)
        
        # Measure
        qc.measure_all()
        
        metadata = {
            'name': 'deutsch_jozsa',
            'n_qubits': n_qubits,
            'depth': qc.depth(),
            'size': qc.size(),
            'description': 'Deutsch-Jozsa algorithm with balanced oracle'
        }
        
        return qc, metadata
    
    def create_vqe_h2(self, n_qubits=10):
        """
        Create VQE circuit for H2 molecule
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            circuit: Qiskit QuantumCircuit
            metadata: Circuit metadata
        """
        print(f"Creating VQE-H2 circuit ({n_qubits} qubits)...")
        
        # Use EfficientSU2 ansatz for VQE
        qc = EfficientSU2(n_qubits, reps=2, entanglement='linear')
        
        # Add measurements
        qc.measure_all()
        
        metadata = {
            'name': 'vqe_h2',
            'n_qubits': n_qubits,
            'depth': qc.depth(),
            'size': qc.size(),
            'description': 'VQE circuit for H2 molecule using EfficientSU2 ansatz'
        }
        
        return qc, metadata
    
    def create_qaoa_maxcut(self, n_qubits=20):
        """
        Create QAOA circuit for MaxCut problem
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            circuit: Qiskit QuantumCircuit
            metadata: Circuit metadata
        """
        print(f"Creating QAOA-MaxCut circuit ({n_qubits} qubits)...")
        
        # Create random graph for MaxCut
        graph = nx.erdos_renyi_graph(n_qubits, 0.5)
        
        # Create QAOA ansatz
        qc = QAOAAnsatz(cost_operator=graph, reps=2)
        
        # Add measurements
        qc.measure_all()
        
        metadata = {
            'name': 'qaoa_maxcut',
            'n_qubits': n_qubits,
            'depth': qc.depth(),
            'size': qc.size(),
            'graph_nodes': n_qubits,
            'graph_edges': graph.number_of_edges(),
            'description': 'QAOA circuit for MaxCut on random graph'
        }
        
        return qc, metadata
    
    def load_noise_profile(self, n_qubits):
        """
        Load noise profile from Phase 2
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            noise_profile: Dictionary with noise parameters
        """
        if self.noise_profile_dir is None:
            print("No noise profile directory specified, using defaults")
            return self._default_noise_profile(n_qubits)
        
        noise_file = os.path.join(self.noise_profile_dir, f"{n_qubits}q", 
                                 f"noise_profile_{n_qubits}q_final.csv")
        
        if os.path.exists(noise_file):
            import pandas as pd
            df = pd.read_csv(noise_file)
            
            # Convert to dictionary format
            noise_profile = {
                'T1': df['T1'].tolist(),
                'T2': df['T2'].tolist(),
                'depolarization_prob': df['depolarization_prob'].tolist(),
                'dephasing_prob': df['dephasing_prob'].tolist(),
                'gate_error_rate': df['gate_error_rate'].tolist()
            }
            
            print(f"Loaded noise profile for {n_qubits} qubits")
            return noise_profile
        else:
            print(f"Noise profile not found: {noise_file}")
            print("Using default noise profile")
            return self._default_noise_profile(n_qubits)
    
    def _default_noise_profile(self, n_qubits):
        """Create default noise profile"""
        return {
            'T1': [100.0] * n_qubits,  # microseconds
            'T2': [80.0] * n_qubits,   # microseconds
            'depolarization_prob': [0.001] * n_qubits,
            'dephasing_prob': [0.0005] * n_qubits,
            'gate_error_rate': [0.005] * n_qubits
        }
    
    def load_all_benchmarks(self, circuits=None, qubits=None):
        """
        Load all benchmark circuits
        
        Args:
            circuits: List of circuit names to load
            qubits: List of qubit counts for each circuit
            
        Returns:
            benchmarks: Dictionary with all benchmark circuits
        """
        if circuits is None:
            circuits = ['deutsch_jozsa', 'vqe_h2', 'qaoa_maxcut']
        
        if qubits is None:
            qubits = [5, 10, 20]  # Default qubit counts
        
        if len(circuits) != len(qubits):
            raise ValueError("Number of circuits must match number of qubit counts")
        
        benchmarks = {}
        
        for circuit_name, n_qubits in zip(circuits, qubits):
            if circuit_name == 'deutsch_jozsa':
                qiskit_circuit, metadata = self.create_deutsch_jozsa(n_qubits)
            elif circuit_name == 'vqe_h2':
                qiskit_circuit, metadata = self.create_vqe_h2(n_qubits)
            elif circuit_name == 'qaoa_maxcut':
                qiskit_circuit, metadata = self.create_qaoa_maxcut(n_qubits)
            else:
                print(f"Unknown circuit: {circuit_name}")
                continue
            
            # Convert to Perceval
            perceval_circuit, conversion_metadata = self.converter.convert_circuit(qiskit_circuit)
            
            # Load noise profile
            noise_profile = self.load_noise_profile(n_qubits)
            
            # Store everything
            benchmarks[circuit_name] = {
                'qiskit_circuit': qiskit_circuit,
                'perceval_circuit': perceval_circuit,
                'metadata': metadata,
                'conversion_metadata': conversion_metadata,
                'noise_profile': noise_profile,
                'n_qubits': n_qubits
            }
            
            print(f"✅ Loaded {circuit_name} with {n_qubits} qubits")
            print(f"  Qiskit gates: {qiskit_circuit.size()}")
            print(f"  Perceval components: {conversion_metadata['converted_gates']}")
        
        return benchmarks

def test_benchmark_loader():
    """Test the benchmark loader"""
    print("Testing BenchmarkLoader...")
    
    # Initialize loader with Phase 2 noise profiles
    loader = BenchmarkLoader(
        noise_profile_dir="../../../experiments/thrust1/noise_profiles/"
    )
    
    # Load benchmarks
    benchmarks = loader.load_all_benchmarks()
    
    print("\n" + "="*60)
    print("BENCHMARK LOADER TEST RESULTS")
    print("="*60)
    
    for name, data in benchmarks.items():
        print(f"\n{name.upper()}:")
        print(f"  Qubits: {data['n_qubits']}")
        print(f"  Qiskit gates: {data['qiskit_circuit'].size()}")
        print(f"  Converted gates: {data['conversion_metadata']['converted_gates']}")
        print(f"  Noise profile loaded: {len(data['noise_profile']['T1'])} qubits")
    
    return benchmarks

if __name__ == "__main__":
    benchmarks = test_benchmark_loader()
