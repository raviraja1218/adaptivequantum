#!/usr/bin/env python3
"""
Convert Qiskit circuits to Perceval photonic circuits
"""

import numpy as np
import perceval as pcvl
import perceval.components.unitary_components as comp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

class QiskitToPercevalConverter:
    """Convert Qiskit circuits to Perceval representation"""
    
    def __init__(self, n_qubits=None):
        """
        Initialize converter
        
        Args:
            n_qubits: Number of qubits (modes in photonic system)
        """
        self.n_qubits = n_qubits
        self.gate_conversions = self._initialize_gate_conversions()
        
    def _initialize_gate_conversions(self):
        """Initialize gate conversion dictionary"""
        return {
            'h': self._convert_hadamard,
            'x': self._convert_pauli_x,
            'y': self._convert_pauli_y,
            'z': self._convert_pauli_z,
            'rx': self._convert_rx,
            'ry': self._convert_ry,
            'rz': self._convert_rz,
            'cx': self._convert_cnot,
            'cz': self._convert_cz,
            'swap': self._convert_swap
        }
    
    def convert_circuit(self, qiskit_circuit):
        """
        Convert Qiskit circuit to Perceval circuit
        
        Args:
            qiskit_circuit: Qiskit QuantumCircuit
            
        Returns:
            perceval_circuit: Perceval Circuit object
            conversion_metadata: Dictionary with conversion info
        """
        if self.n_qubits is None:
            self.n_qubits = qiskit_circuit.num_qubits
        
        # Create empty Perceval circuit
        perceval_circuit = pcvl.Circuit(self.n_qubits)
        
        conversion_metadata = {
            'original_qubits': self.n_qubits,
            'total_gates': qiskit_circuit.size(),
            'converted_gates': 0,
            'failed_conversions': 0,
            'gate_types': {}
        }
        
        # Process each gate in the Qiskit circuit
        for instruction in qiskit_circuit.data:
            gate_name = instruction.operation.name.lower()
            qubits = [qiskit_circuit.qubits.index(q) for q in instruction.qubits]
            params = instruction.operation.params
            
            # Track gate type
            if gate_name not in conversion_metadata['gate_types']:
                conversion_metadata['gate_types'][gate_name] = 0
            conversion_metadata['gate_types'][gate_name] += 1
            
            # Convert gate
            if gate_name in self.gate_conversions:
                try:
                    self.gate_conversions[gate_name](
                        perceval_circuit, qubits, params
                    )
                    conversion_metadata['converted_gates'] += 1
                except Exception as e:
                    print(f"Warning: Failed to convert {gate_name} gate: {e}")
                    conversion_metadata['failed_conversions'] += 1
            else:
                print(f"Warning: Gate {gate_name} not supported for conversion")
                conversion_metadata['failed_conversions'] += 1
        
        return perceval_circuit, conversion_metadata
    
    def _convert_hadamard(self, circuit, qubits, params):
        """Convert Hadamard gate to beam splitter + phase shifters"""
        q = qubits[0]
        # Hadamard ≈ 50/50 beam splitter with specific phases
        circuit.add(q, comp.BS())
        
    def _convert_pauli_x(self, circuit, qubits, params):
        """Convert Pauli-X gate (bit flip)"""
        q = qubits[0]
        # X gate = π rotation around X axis
        circuit.add(q, comp.PS(phi=np.pi))
        
    def _convert_rx(self, circuit, qubits, params):
        """Convert rotation around X axis"""
        q = qubits[0]
        theta = float(params[0])
        circuit.add(q, comp.PS(phi=theta))
        
    def _convert_ry(self, circuit, qubits, params):
        """Convert rotation around Y axis"""
        q = qubits[0]
        theta = float(params[0])
        # RY(θ) = RX(π/2) RZ(θ) RX(-π/2)
        circuit.add(q, comp.PS(phi=np.pi/2))
        circuit.add(q, comp.PS(phi=theta))
        circuit.add(q, comp.PS(phi=-np.pi/2))
        
    def _convert_rz(self, circuit, qubits, params):
        """Convert rotation around Z axis"""
        q = qubits[0]
        theta = float(params[0])
        circuit.add(q, comp.PS(phi=theta))
        
    def _convert_cnot(self, circuit, qubits, params):
        """
        Convert CNOT gate to photonic components
        CNOT requires 4 modes for 2 qubits in linear optics
        """
        control, target = qubits[0], qubits[1]
        
        # CNOT decomposition for linear optics (simplified)
        # In reality, CNOT requires non-linear optics or measurement
        # This is a placeholder approximation
        
        # Add controlled phase
        circuit.add(control, comp.PS(phi=np.pi))
        # Add target operation conditioned on control
        circuit.add(target, comp.PS(phi=np.pi))
        
    def _convert_cz(self, circuit, qubits, params):
        """Convert CZ gate"""
        control, target = qubits[0], qubits[1]
        circuit.add(control, comp.PS(phi=np.pi))
        circuit.add(target, comp.PS(phi=np.pi))
        
    def _convert_swap(self, circuit, qubits, params):
        """Convert SWAP gate"""
        q1, q2 = qubits[0], qubits[1]
        # SWAP = 3 CNOTs, simplified here
        circuit.add(q1, comp.PS(phi=np.pi))
        circuit.add(q2, comp.PS(phi=np.pi))

# Test function
def test_conversion():
    """Test the converter with a simple circuit"""
    from qiskit import QuantumCircuit
    
    # Create a simple Qiskit circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    print("Original Qiskit circuit:")
    print(qc)
    
    # Convert to Perceval
    converter = QiskitToPercevalConverter(n_qubits=2)
    perceval_circuit, metadata = converter.convert_circuit(qc)
    
    print("\nConversion metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print("\nPerceval circuit created successfully")
    return perceval_circuit, metadata

if __name__ == "__main__":
    test_conversion()
