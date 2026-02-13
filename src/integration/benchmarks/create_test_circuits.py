#!/usr/bin/env python3
"""
Create test benchmark circuits for Phase 5 integration.
"""

import numpy as np
import pickle
from pathlib import Path

def create_test_circuits():
    """Create sample benchmark circuits for integration testing."""
    
    circuits_dir = Path('data/processed/benchmark_circuits')
    circuits_dir.mkdir(parents=True, exist_ok=True)
    
    # Circuit 1: Deutsch-Jozsa (5 qubits)
    deutsch_jozsa = {
        'name': 'deutsch_jozsa_5q',
        'qubits': 5,
        'depth': 12,
        'gates': ['H', 'CX', 'H', 'M'],
        'params': np.random.randn(5, 3),  # 5 qubits, 3 params each
        'description': '5-qubit Deutsch-Jozsa algorithm'
    }
    
    # Circuit 2: VQE for H2 (10 qubits)
    vqe_h2 = {
        'name': 'vqe_h2_10q',
        'qubits': 10,
        'depth': 45,
        'gates': ['H', 'RX', 'RY', 'RZ', 'CX', 'M'],
        'params': np.random.randn(10, 3),
        'description': '10-qubit VQE for H2 molecule ground state'
    }
    
    # Circuit 3: QAOA for MaxCut (20 qubits)
    qaoa_maxcut = {
        'name': 'qaoa_maxcut_20q',
        'qubits': 20,
        'depth': 34,
        'gates': ['H', 'RZ', 'RX', 'CX', 'M'],
        'params': np.random.randn(20, 2),  # QAOA has beta and gamma params
        'description': '20-qubit QAOA for MaxCut problem'
    }
    
    # Save circuits
    circuits = {
        'deutsch_jozsa': deutsch_jozsa,
        'vqe_h2': vqe_h2,
        'qaoa_maxcut': qaoa_maxcut
    }
    
    with open(circuits_dir / 'integration_benchmarks.pkl', 'wb') as f:
        pickle.dump(circuits, f)
    
    print(f"✅ Created {len(circuits)} benchmark circuits")
    print(f"  - Deutsch-Jozsa: {deutsch_jozsa['qubits']} qubits, {deutsch_jozsa['depth']} depth")
    print(f"  - VQE H2: {vqe_h2['qubits']} qubits, {vqe_h2['depth']} depth")
    print(f"  - QAOA MaxCut: {qaoa_maxcut['qubits']} qubits, {qaoa_maxcut['depth']} depth")
    print(f"📁 Saved to: {circuits_dir / 'integration_benchmarks.pkl'}")

if __name__ == '__main__':
    create_test_circuits()
