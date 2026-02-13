"""
Surface code implementation for distance d=3 (9 physical qubits).
"""
import numpy as np
import torch
from typing import List, Tuple, Dict
import yaml

class SurfaceCode:
    """Surface code implementation for quantum error correction."""
    
    def __init__(self, distance: int = 3):
        self.distance = distance
        self.n_physical = distance ** 2
        self.n_logical = 1
        self.stabilizers = self._generate_stabilizers()
        
        # Load configuration
        with open('config/phase4_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
    
    def _generate_stabilizers(self):
        """Generate stabilizer generators for surface code."""
        stabilizers = []
        d = self.distance
        
        # Z-stabilizers (plaquettes)
        for i in range(d-1):
            for j in range(d-1):
                if (i + j) % 2 == 0:  # Only even plaquettes for Z
                    qubits = []
                    if i > 0: qubits.append((i-1)*d + j)
                    if i < d-1: qubits.append((i+1)*d + j)
                    if j > 0: qubits.append(i*d + (j-1))
                    if j < d-1: qubits.append(i*d + (j+1))
                    stabilizers.append(('Z', qubits))
        
        # X-stabilizers (stars)
        for i in range(d-1):
            for j in range(d-1):
                if (i + j) % 2 == 1:  # Only odd stars for X
                    qubits = []
                    if i > 0: qubits.append((i-1)*d + j)
                    if i < d-1: qubits.append((i+1)*d + j)
                    if j > 0: qubits.append(i*d + (j-1))
                    if j < d-1: qubits.append(i*d + (j+1))
                    stabilizers.append(('X', qubits))
        
        return stabilizers
    
    def generate_syndrome(self, error: torch.Tensor) -> torch.Tensor:
        """Generate syndrome from error vector."""
        syndrome = torch.zeros(len(self.stabilizers))
        
        for idx, (stabilizer_type, qubits) in enumerate(self.stabilizers):
            # Check parity of errors on stabilizer support
            if stabilizer_type == 'Z':
                # Z stabilizer anti-commutes with X errors
                syndrome[idx] = sum(error[q].item() for q in qubits) % 2
            else:  # 'X' stabilizer
                # X stabilizer anti-commutes with Z errors
                # For simplicity, we'll use the same for now (can extend later)
                syndrome[idx] = sum(error[q].item() for q in qubits) % 2
        
        return syndrome
    
    def generate_error(self, noise_type: str = 'depolarizing', 
                       error_rate: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random error and corresponding syndrome."""
        error = torch.zeros(self.n_physical)
        
        if noise_type == 'depolarizing':
            # Depolarizing noise: X, Y, or Z errors with equal probability
            for i in range(self.n_physical):
                if torch.rand(1).item() < error_rate:
                    error[i] = 1  # Simple bit-flip model
        
        elif noise_type == 'amplitude_damping':
            # Amplitude damping: |1> -> |0> transitions
            for i in range(self.n_physical):
                if torch.rand(1).item() < error_rate:
                    error[i] = 1
        
        elif noise_type == 'phase_damping':
            # Phase damping: |+> -> |->, |-> -> |+>
            for i in range(self.n_physical):
                if torch.rand(1).item() < error_rate:
                    error[i] = 1
        
        elif noise_type == 'combined':
            # Combined noise model
            for i in range(self.n_physical):
                if torch.rand(1).item() < error_rate/2:
                    error[i] = 1  # Bit-flip
                if torch.rand(1).item() < error_rate/2:
                    # Could add phase-flip component here
                    pass
        
        syndrome = self.generate_syndrome(error)
        return error, syndrome
    
    def generate_dataset(self, n_samples: int = 1000, 
                        noise_type: str = 'depolarizing') -> Dict:
        """Generate dataset of (error, syndrome) pairs."""
        errors = []
        syndromes = []
        
        for _ in range(n_samples):
            error, syndrome = self.generate_error(noise_type)
            errors.append(error)
            syndromes.append(syndrome)
        
        return {
            'errors': torch.stack(errors),
            'syndromes': torch.stack(syndromes),
            'noise_type': noise_type,
            'n_samples': n_samples
        }

if __name__ == "__main__":
    # Test the surface code implementation
    sc = SurfaceCode(distance=3)
    print(f"Surface Code with distance {sc.distance}")
    print(f"Physical qubits: {sc.n_physical}")
    print(f"Stabilizers: {len(sc.stabilizers)}")
    
    # Generate a test error
    error, syndrome = sc.generate_error('depolarizing', 0.01)
    print(f"\nTest error: {error}")
    print(f"Syndrome: {syndrome}")
