"""
Photonic circuit representation and simulation.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from .components import BeamSplitter, PhaseShifter, IdentityGate

class PhotonicCircuit:
    """Simple photonic circuit simulator."""
    
    def __init__(self, n_modes: int = 2):
        """
        Initialize photonic circuit.
        
        Parameters:
        -----------
        n_modes : int
            Number of optical modes (default: 2)
        """
        self.n_modes = n_modes
        self.components = []  # List of (component, mode_indices)
        self.component_count = 0
        
    def add_component(self, component, mode_indices: Tuple[int, int]):
        """
        Add component to circuit.
        
        Parameters:
        -----------
        component : object
            Component (BeamSplitter, PhaseShifter, etc.)
        mode_indices : tuple
            Which modes the component acts on
        """
        if len(mode_indices) != 2:
            raise ValueError("Components must act on exactly 2 modes")
        
        self.components.append((component, mode_indices))
        self.component_count += 1
        
    def add_beam_splitter(self, theta: float = np.pi/4, phi: float = 0.0, 
                          mode_indices: Tuple[int, int] = (0, 1)):
        """Convenience method to add beam splitter."""
        bs = BeamSplitter(theta, phi)
        self.add_component(bs, mode_indices)
        return bs
    
    def add_phase_shifter(self, phi: float = 0.0, 
                          mode_indices: Tuple[int, int] = (0, 1)):
        """Convenience method to add phase shifter."""
        ps = PhaseShifter(phi)
        self.add_component(ps, mode_indices)
        return ps
    
    def get_unitary(self) -> np.ndarray:
        """
        Calculate overall unitary matrix for the circuit.
        
        Returns:
        --------
        np.ndarray
            n_modes x n_modes unitary matrix
        """
        # Start with identity matrix
        U = np.eye(self.n_modes, dtype=complex)
        
        for component, (i, j) in self.components:
            # Create matrix for this component in full Hilbert space
            comp_matrix = np.eye(self.n_modes, dtype=complex)
            
            # Extract 2x2 submatrix from component
            comp_2x2 = component.matrix
            
            # Place in correct positions
            comp_matrix[i, i] = comp_2x2[0, 0]
            comp_matrix[i, j] = comp_2x2[0, 1]
            comp_matrix[j, i] = comp_2x2[1, 0]
            comp_matrix[j, j] = comp_2x2[1, 1]
            
            # Multiply with overall unitary
            U = comp_matrix @ U
        
        return U
    
    def count_gates(self) -> Dict[str, int]:
        """Count different types of gates in circuit."""
        counts = {"beam_splitter": 0, "phase_shifter": 0, "identity": 0}
        
        for component, _ in self.components:
            if isinstance(component, BeamSplitter):
                counts["beam_splitter"] += 1
            elif isinstance(component, PhaseShifter):
                counts["phase_shifter"] += 1
            elif isinstance(component, IdentityGate):
                counts["identity"] += 1
        
        return counts
    
    def optimize(self, max_iterations: int = 100):
        """
        Simple optimization: remove identities and fuse consecutive gates.
        Returns optimized circuit and gate reduction percentage.
        """
        original_count = self.component_count
        
        # Remove identity gates
        self.components = [(comp, modes) for comp, modes in self.components 
                          if not (isinstance(comp, IdentityGate) or 
                                 (isinstance(comp, BeamSplitter) and np.abs(comp.theta) < 1e-10) or
                                 (isinstance(comp, PhaseShifter) and np.abs(comp.phi) < 1e-10))]
        
        # Simple fusion of consecutive same-type gates on same modes
        optimized_components = []
        i = 0
        
        while i < len(self.components):
            if i == len(self.components) - 1:
                optimized_components.append(self.components[i])
                i += 1
                continue
            
            comp1, modes1 = self.components[i]
            comp2, modes2 = self.components[i + 1]
            
            # Check if same type and same modes
            if modes1 == modes2:
                if isinstance(comp1, BeamSplitter) and isinstance(comp2, BeamSplitter):
                    # Fuse beam splitters
                    from .components import fuse_beam_splitters
                    fused = fuse_beam_splitters(comp1, comp2)
                    optimized_components.append((fused, modes1))
                    i += 2
                    continue
                elif isinstance(comp1, PhaseShifter) and isinstance(comp2, PhaseShifter):
                    # Fuse phase shifters
                    from .components import fuse_phase_shifters
                    fused = fuse_phase_shifters(comp1, comp2)
                    optimized_components.append((fused, modes1))
                    i += 2
                    continue
            
            optimized_components.append(self.components[i])
            i += 1
        
        self.components = optimized_components
        self.component_count = len(self.components)
        
        # Calculate reduction
        if original_count > 0:
            reduction = 100 * (original_count - self.component_count) / original_count
        else:
            reduction = 0.0
        
        return reduction
    
    def calculate_photon_loss(self, loss_per_gate: float = 0.1) -> float:
        """
        Calculate photon survival probability.
        
        Parameters:
        -----------
        loss_per_gate : float
            Probability of photon loss per gate (default: 0.1)
        
        Returns:
        --------
        float
            Probability that photon survives entire circuit
        """
        survival_per_gate = 1 - loss_per_gate
        return survival_per_gate ** self.component_count
    
    def __str__(self):
        counts = self.count_gates()
        return (f"PhotonicCircuit({self.n_modes} modes, {self.component_count} components)\n"
                f"  Beam splitters: {counts['beam_splitter']}\n"
                f"  Phase shifters: {counts['phase_shifter']}\n"
                f"  Identity gates: {counts['identity']}")
