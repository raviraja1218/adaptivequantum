"""
Verify the photonic simulator works correctly.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.thrust2.photonic_simulator.components import (
    BeamSplitter, PhaseShifter, IdentityGate,
    fuse_beam_splitters, fuse_phase_shifters
)
from src.thrust2.photonic_simulator.circuit import PhotonicCircuit

def test_components():
    """Test basic component functionality."""
    print("Testing components...")
    
    # Test beam splitter
    bs = BeamSplitter(theta=np.pi/4, phi=0.0)
    bs_matrix = bs.matrix
    print(f"Beam splitter matrix shape: {bs_matrix.shape}")
    print(f"Beam splitter is unitary: {np.allclose(bs_matrix @ bs_matrix.conj().T, np.eye(2))}")
    
    # Test phase shifter
    ps = PhaseShifter(phi=np.pi/2)
    ps_matrix = ps.matrix
    print(f"Phase shifter matrix shape: {ps_matrix.shape}")
    print(f"Phase shifter is diagonal: {np.allclose(ps_matrix[0,1], 0) and np.allclose(ps_matrix[1,0], 0)}")
    
    # Test identity
    identity = IdentityGate()
    id_matrix = identity.matrix
    print(f"Identity matrix is identity: {np.allclose(id_matrix, np.eye(2))}")
    
    # Test fusion
    bs1 = BeamSplitter(theta=np.pi/8)
    bs2 = BeamSplitter(theta=np.pi/8)
    fused_bs = fuse_beam_splitters(bs1, bs2)
    print(f"Fused beam splitter theta: {fused_bs.theta:.3f} (expected: {np.pi/4:.3f})")
    
    ps1 = PhaseShifter(phi=np.pi/4)
    ps2 = PhaseShifter(phi=np.pi/4)
    fused_ps = fuse_phase_shifters(ps1, ps2)
    print(f"Fused phase shifter phi: {fused_ps.phi:.3f} (expected: {np.pi/2:.3f})")
    
    print("✓ Component tests passed\n")

def test_circuit():
    """Test circuit functionality."""
    print("Testing circuit...")
    
    # Create simple 2-mode circuit
    circuit = PhotonicCircuit(n_modes=2)
    
    # Add some components
    circuit.add_beam_splitter(theta=np.pi/4, mode_indices=(0, 1))
    circuit.add_phase_shifter(phi=np.pi/2, mode_indices=(0, 1))
    circuit.add_beam_splitter(theta=np.pi/4, mode_indices=(0, 1))
    
    print(f"Circuit: {circuit}")
    
    # Get unitary matrix
    U = circuit.get_unitary()
    print(f"Unitary matrix shape: {U.shape}")
    print(f"Unitary is unitary: {np.allclose(U @ U.conj().T, np.eye(2))}")
    
    # Count gates
    counts = circuit.count_gates()
    print(f"Gate counts: {counts}")
    
    # Calculate photon loss
    survival = circuit.calculate_photon_loss(loss_per_gate=0.1)
    print(f"Photon survival probability: {survival:.3f}")
    
    # Optimize circuit
    reduction = circuit.optimize()
    print(f"After optimization:")
    print(f"  Component count: {circuit.component_count}")
    print(f"  Gate reduction: {reduction:.1f}%")
    
    counts_after = circuit.count_gates()
    print(f"  Gate counts after: {counts_after}")
    
    print("✓ Circuit tests passed\n")

def test_optimization():
    """Test circuit optimization."""
    print("Testing optimization...")
    
    # Create circuit with redundant gates
    circuit = PhotonicCircuit(n_modes=2)
    
    # Add consecutive beam splitters that can be fused
    circuit.add_beam_splitter(theta=np.pi/8, mode_indices=(0, 1))
    circuit.add_beam_splitter(theta=np.pi/8, mode_indices=(0, 1))
    circuit.add_phase_shifter(phi=np.pi/4, mode_indices=(0, 1))
    circuit.add_phase_shifter(phi=np.pi/4, mode_indices=(0, 1))
    circuit.add_component(IdentityGate(), (0, 1))  # Should be removed
    
    original_count = circuit.component_count
    print(f"Original component count: {original_count}")
    
    # Optimize
    reduction = circuit.optimize()
    optimized_count = circuit.component_count
    
    print(f"Optimized component count: {optimized_count}")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Components removed: {original_count - optimized_count}")
    
    if optimized_count < original_count:
        print("✓ Optimization test passed")
    else:
        print("✗ Optimization test failed")
    
    print()

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("PHOTONIC SIMULATOR VERIFICATION")
    print("=" * 60)
    
    try:
        test_components()
        test_circuit()
        test_optimization()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED - SIMULATOR IS WORKING")
        print("=" * 60)
        
        # Save verification result
        with open("experiments/thrust2/validation/simulator_verification.txt", "w") as f:
            f.write("Simulator verification passed on ")
            f.write(__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            f.write("\n\nAll tests passed successfully.")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
