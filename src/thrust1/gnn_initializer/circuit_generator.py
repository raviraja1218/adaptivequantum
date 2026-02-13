"""
Generate random quantum circuits for training data - FIXED VERSION
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from pathlib import Path
import pandas as pd

class CircuitGenerator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
    
    def generate_circuit(self, n_qubits, depth=20):
        """Generate random parameterized quantum circuit"""
        # Create circuit with parameters
        params = []
        param_idx = 0
        
        qc = QuantumCircuit(n_qubits)
        
        for layer in range(depth):
            # Single-qubit rotations
            for q in range(n_qubits):
                # Add parameterized rotation
                theta = Parameter(f'θ_{param_idx}')
                params.append(theta)
                param_idx += 1
                
                # Random rotation axis
                axis = self.rng.choice(['x', 'y', 'z'])
                if axis == 'x':
                    qc.rx(theta, q)
                elif axis == 'y':
                    qc.ry(theta, q)
                else:
                    qc.rz(theta, q)
            
            # Entangling layers (alternating pattern)
            if layer % 2 == 0:
                # Even layer: connect even-odd pairs
                for q in range(0, n_qubits-1, 2):
                    if q+1 < n_qubits:
                        qc.cx(q, q+1)
            else:
                # Odd layer: connect odd-even pairs
                for q in range(1, n_qubits-1, 2):
                    if q+1 < n_qubits:
                        qc.cx(q, q+1)
        
        # Create adjacency matrix from circuit connectivity
        adjacency = self._create_adjacency_matrix(n_qubits, qc)
        
        return {
            'circuit': qc,
            'parameters': params,
            'n_qubits': n_qubits,
            'depth': depth,
            'adjacency': adjacency,
            'n_parameters': len(params)
        }
    
    def _create_adjacency_matrix(self, n_qubits, circuit):
        """Create adjacency matrix from circuit connectivity"""
        adjacency = np.zeros((n_qubits, n_qubits))
        
        # Analyze circuit gates to determine connectivity
        for instruction in circuit.data:
            qubits = [qubit.index for qubit in instruction.qubits]
            
            if len(qubits) == 1:
                # Single-qubit gate, self-connection
                q = qubits[0]
                adjacency[q, q] = 1
            elif len(qubits) == 2:
                # Two-qubit gate, bidirectional connection
                q1, q2 = qubits
                adjacency[q1, q2] = 1
                adjacency[q2, q1] = 1
        
        # Ensure at least self-connections
        for q in range(n_qubits):
            adjacency[q, q] = 1
        
        return adjacency
    
    def generate_noise_profile(self, n_qubits):
        """Generate realistic noise profile for qubits"""
        noise_profiles_dir = Path("experiments/thrust1/noise_profiles")
        profile_file = noise_profiles_dir / f"{n_qubits}q" / f"noise_profile_{n_qubits}q_final.csv"
        
        if profile_file.exists():
            # Load existing profile
            df = pd.read_csv(profile_file)
            
            # Ensure we have enough rows
            if len(df) < n_qubits:
                # Repeat and trim if needed
                repeats = (n_qubits // len(df)) + 1
                df = pd.concat([df] * repeats, ignore_index=True).head(n_qubits)
            
            noise_profile = {
                'T1': df['T1'].values[:n_qubits],
                'T2': df['T2'].values[:n_qubits],
                'depolarizing_prob': df['depolarizing_prob'].values[:n_qubits] 
                                   if 'depolarizing_prob' in df.columns 
                                   else np.full(n_qubits, 0.001),
                'dephasing_prob': df['dephasing_prob'].values[:n_qubits] 
                                if 'dephasing_prob' in df.columns 
                                else np.full(n_qubits, 0.0005),
                'gate_error_1q': df['gate_error_1q'].values[:n_qubits] 
                               if 'gate_error_1q' in df.columns 
                               else np.full(n_qubits, 0.005),
                'gate_error_2q': df['gate_error_2q'].values[:n_qubits] 
                               if 'gate_error_2q' in df.columns 
                               else np.full(n_qubits, 0.015)
            }
        else:
            # Generate synthetic noise profile
            noise_profile = {
                'T1': np.random.uniform(80, 120, n_qubits),  # microseconds
                'T2': np.random.uniform(60, 100, n_qubits),
                'depolarizing_prob': np.random.uniform(0.0005, 0.002, n_qubits),
                'dephasing_prob': np.random.uniform(0.0003, 0.0015, n_qubits),
                'gate_error_1q': np.random.uniform(0.004, 0.015, n_qubits),
                'gate_error_2q': np.random.uniform(0.012, 0.045, n_qubits)  # 3x 1-qubit error
            }
            
            # Ensure T2 <= 2*T1 (physical constraint)
            noise_profile['T2'] = np.minimum(noise_profile['T2'], 2 * noise_profile['T1'])
        
        return noise_profile

if __name__ == "__main__":
    # Test the circuit generator
    generator = CircuitGenerator(seed=42)
    
    for n_qubits in [5, 10]:
        print(f"\nTesting with {n_qubits} qubits:")
        
        circuit_data = generator.generate_circuit(n_qubits, depth=10)
        print(f"  Circuit depth: {circuit_data['depth']}")
        print(f"  Parameters: {circuit_data['n_parameters']}")
        print(f"  Adjacency shape: {circuit_data['adjacency'].shape}")
        
        noise_profile = generator.generate_noise_profile(n_qubits)
        print(f"  Noise profile generated")
        print(f"  T1 range: {noise_profile['T1'].min():.1f}-{noise_profile['T1'].max():.1f} μs")
