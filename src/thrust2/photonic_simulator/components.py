"""
Basic photonic components: Beam splitters and phase shifters.
"""
import numpy as np
from typing import Tuple

class BeamSplitter:
    """Beam splitter component with tunable reflectivity."""
    
    def __init__(self, theta: float = np.pi/4, phi: float = 0.0):
        """
        Initialize beam splitter.
        
        Parameters:
        -----------
        theta : float
            Reflectivity parameter (default: 50/50 beam splitter)
        phi : float
            Phase difference between transmission and reflection
        """
        self.theta = theta
        self.phi = phi
        
    @property
    def matrix(self) -> np.ndarray:
        """Get 2x2 unitary matrix for beam splitter."""
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        exp_i_phi = np.exp(1j * self.phi)
        
        return np.array([
            [cos_t, -exp_i_phi * sin_t],
            [exp_i_phi * sin_t, cos_t]
        ])
    
    def __str__(self):
        return f"BeamSplitter(θ={self.theta:.3f}, φ={self.phi:.3f})"


class PhaseShifter:
    """Phase shifter component."""
    
    def __init__(self, phi: float = 0.0):
        """
        Initialize phase shifter.
        
        Parameters:
        -----------
        phi : float
            Phase shift angle
        """
        self.phi = phi
        
    @property
    def matrix(self) -> np.ndarray:
        """Get 2x2 diagonal matrix for phase shifter."""
        return np.array([
            [np.exp(1j * self.phi), 0],
            [0, 1]
        ])
    
    def __str__(self):
        return f"PhaseShifter(φ={self.phi:.3f})"


class IdentityGate:
    """Identity gate (no operation)."""
    
    @property
    def matrix(self) -> np.ndarray:
        """Get 2x2 identity matrix."""
        return np.eye(2, dtype=complex)
    
    def __str__(self):
        return "Identity()"


def fuse_beam_splitters(bs1: BeamSplitter, bs2: BeamSplitter) -> BeamSplitter:
    """
    Fuse two consecutive beam splitters into one.
    
    For small circuits: BS(θ1)·BS(θ2) ≈ BS(θ1 + θ2)
    """
    # Simplified fusion: add angles
    new_theta = (bs1.theta + bs2.theta) % (2 * np.pi)
    new_phi = (bs1.phi + bs2.phi) % (2 * np.pi)
    
    return BeamSplitter(new_theta, new_phi)


def fuse_phase_shifters(ps1: PhaseShifter, ps2: PhaseShifter) -> PhaseShifter:
    """
    Fuse two consecutive phase shifters into one.
    PS(φ1)·PS(φ2) = PS(φ1 + φ2)
    """
    new_phi = (ps1.phi + ps2.phi) % (2 * np.pi)
    return PhaseShifter(new_phi)


def is_identity_component(component) -> bool:
    """Check if component is approximately identity."""
    if isinstance(component, IdentityGate):
        return True
    if isinstance(component, BeamSplitter):
        return np.abs(component.theta) < 1e-10
    if isinstance(component, PhaseShifter):
        return np.abs(component.phi) < 1e-10
    return False
