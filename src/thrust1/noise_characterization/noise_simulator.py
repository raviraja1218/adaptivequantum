"""
Noise simulator for quantum circuits.
"""

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import depolarizing_error, thermal_relaxation_error
import json
from pathlib import Path

class QuantumNoiseSimulator:
    """Simulate quantum noise for characterization."""
    
    def __init__(self, n_qubits=5, shots=1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_noise_model(self, t1=100e-6, t2=100e-6, single_qubit_error=0.001, two_qubit_error=0.01):
        """Create a realistic noise model."""
        noise_model = NoiseModel()
        
        # Single qubit depolarizing error
        error_single = depolarizing_error(single_qubit_error, 1)
        noise_model.add_all_qubit_quantum_error(error_single, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg'])
        
        # Two qubit depolarizing error
        error_two = depolarizing_error(two_qubit_error, 2)
        noise_model.add_all_qubit_quantum_error(error_two, ['cx', 'cz', 'swap'])
        
        # Thermal relaxation
        error_thermal = thermal_relaxation_error(t1, t2, 50e-9)  # 50ns gate time
        noise_model.add_all_qubit_quantum_error(error_thermal, ['id'])
        
        return noise_model
    
    def create_test_circuit(self, depth=10):
        """Create a test circuit with given depth."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        
        # Add layers of random gates
        for layer in range(depth):
            # Add single qubit rotations
            for qubit in range(self.n_qubits):
                qc.rx(np.random.random() * 2*np.pi, qubit)
                qc.ry(np.random.random() * 2*np.pi, qubit)
            
            # Add entangling gates
            if layer % 2 == 0:
                for qubit in range(0, self.n_qubits-1, 2):
                    qc.cx(qubit, qubit+1)
            else:
                for qubit in range(1, self.n_qubits-1, 2):
                    qc.cx(qubit, qubit+1)
        
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc
    
    def simulate(self, circuit, noise_model=None):
        """Simulate circuit with optional noise."""
        if noise_model:
            result = execute(circuit, self.backend, shots=self.shots, noise_model=noise_model).result()
        else:
            result = execute(circuit, self.backend, shots=self.shots).result()
        
        counts = result.get_counts(circuit)
        return counts
    
    def characterize(self, output_dir='experiments/thrust1'):
        """Run noise characterization."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Test different noise levels
        for error_rate in [0.0001, 0.001, 0.01, 0.05]:
            print(f"Testing error rate: {error_rate}")
            
            noise_model = self.create_noise_model(
                single_qubit_error=error_rate,
                two_qubit_error=error_rate * 10
            )
            
            circuit = self.create_test_circuit(depth=5)
            
            # Simulate with noise
            noisy_counts = self.simulate(circuit, noise_model)
            
            # Simulate without noise (ideal)
            ideal_counts = self.simulate(circuit)
            
            result = {
                'error_rate': error_rate,
                'n_qubits': self.n_qubits,
                'noisy_counts': noisy_counts,
                'ideal_counts': ideal_counts,
                'total_shots': self.shots
            }
            
            results.append(result)
        
        # Save results
        output_file = Path(output_dir) / f'noise_characterization_{self.n_qubits}q.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        return results

def main():
    """Main function for testing."""
    print("Quantum Noise Simulator")
    print("=" * 50)
    
    # Test with 5 qubits
    simulator = QuantumNoiseSimulator(n_qubits=5, shots=512)
    results = simulator.characterize()
    
    print(f"\nCharacterization complete for {simulator.n_qubits} qubits")
    print(f"Generated {len(results)} noise profiles")

if __name__ == "__main__":
    main()
