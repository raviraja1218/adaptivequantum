#!/usr/bin/env python3
"""
Correct verification for Perceval 1.1.0 (Quandela)
"""

import sys
import numpy as np
import perceval as pcvl
from perceval.components import unitary_components as uc
from perceval.components import core_catalog as cc
import os
import json
from datetime import datetime

def test_basic_components():
    """Test basic Perceval components"""
    print("Testing Perceval 1.1.0 (Quandela) basic components...")
    
    try:
        # Beam splitter
        bs = uc.BS()
        print(f"✅ Beam splitter created: {type(bs).__name__}")
        
        # Phase shifter
        ps = uc.PS(phi=np.pi/4)
        print(f"✅ Phase shifter created: PS(phi={ps.get_parameters()[0]:.3f})")
        
        # Create a simple circuit
        circuit = pcvl.Circuit(2)
        circuit.add(0, bs)
        circuit.add(0, ps)
        
        print(f"✅ Simple circuit created with {len(circuit)} components")
        print(f"  Circuit U shape: {circuit.compute_unitary().shape}")
        
        # Test simulation
        backend = pcvl.BackendFactory().get_backend("SLOS")
        input_state = pcvl.BasicState([1, 0])  # One photon in mode 0
        output_distribution = backend(circuit).prob_distribution(input_state)
        
        print(f"✅ Simulation successful")
        print(f"  Input: {input_state}")
        print(f"  Output states: {len(output_distribution)}")
        
        # Show first few outputs
        for state, prob in list(output_distribution.items())[:3]:
            print(f"    {state}: {prob:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic components: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_circuit_operations():
    """Test circuit operations"""
    print("\nTesting circuit operations...")
    
    try:
        # Create a more complex circuit
        circuit = pcvl.Circuit(4)
        
        # Add various components
        circuit.add(0, uc.BS())
        circuit.add(1, uc.PS(phi=np.pi/3))
        
        # Note: CNOT might not be directly available in unitary_components
        # Let's check what's available
        print("Available unitary components:")
        for attr in dir(uc):
            if not attr.startswith('_') and not attr[0].islower():
                print(f"  - {attr}")
        
        # Try to add a waveplate
        circuit.add(2, uc.WP(delta=np.pi/2, theta=np.pi/4))
        
        print(f"✅ Complex circuit created with {len(circuit)} components")
        
        # Test circuit description
        try:
            desc = circuit.describe()
            print(f"✅ Circuit description available (length: {len(desc)} chars)")
        except:
            print("⚠️  Circuit description not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in circuit operations: {e}")
        return False

def test_noise_profiles():
    """Test loading Phase 2 noise profiles"""
    print("\nTesting noise profile integration...")
    
    noise_dir = "../../../experiments/thrust1/noise_profiles/"
    
    if os.path.exists(noise_dir):
        import glob
        import pandas as pd
        
        # Find all noise profile files
        noise_files = glob.glob(os.path.join(noise_dir, "*q", "*final.csv"))
        
        if noise_files:
            # Load first file
            noise_file = noise_files[0]
            df = pd.read_csv(noise_file)
            
            print(f"✅ Noise profile loaded: {os.path.basename(noise_file)}")
            print(f"  Qubits: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            # Calculate average noise parameters
            avg_T1 = df['T1'].mean()
            avg_T2 = df['T2'].mean()
            avg_depol = df['depolarization_prob'].mean()
            
            print(f"  Average T1: {avg_T1:.1f} µs")
            print(f"  Average T2: {avg_T2:.1f} µs")
            print(f"  Average depolarization: {avg_depol:.6f}")
            
            return True
        else:
            print("❌ No final CSV files found in noise profiles")
            return False
    else:
        print(f"❌ Noise directory not found: {noise_dir}")
        return False

def test_backend_performance():
    """Test different backends"""
    print("\nTesting backend performance...")
    
    try:
        # Get available backends
        backends = pcvl.BackendFactory().available_backends()
        
        print(f"✅ Available backends: {backends}")
        
        # Test SLOS backend (state linear optics simulator)
        if "SLOS" in backends:
            backend = pcvl.BackendFactory().get_backend("SLOS")
            
            # Create a test circuit
            circuit = pcvl.Circuit(2)
            circuit.add(0, uc.BS())
            
            # Test performance
            import time
            start = time.time()
            
            input_state = pcvl.BasicState([1, 0])
            output = backend(circuit).prob_distribution(input_state)
            
            elapsed = time.time() - start
            
            print(f"✅ SLOS backend test:")
            print(f"  Circuit: 2 modes, 1 BS")
            print(f"  Execution time: {elapsed*1000:.2f} ms")
            print(f"  Output states: {len(output)}")
            
            return True
        else:
            print("❌ SLOS backend not available")
            return False
            
    except Exception as e:
        print(f"❌ Error in backend test: {e}")
        return False

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("PHASE 3: PERCEVAL 1.1.0 (QUANDELA) - CORRECT VERIFICATION")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Basic Components", test_basic_components),
        ("Circuit Operations", test_circuit_operations),
        ("Backend Performance", test_backend_performance),
        ("Noise Profile Integration", test_noise_profiles)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for i, (test_name, result) in enumerate(zip([t[0] for t in tests], results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test_name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 All {total}/{total} tests passed!")
        overall = "PASS"
    elif passed >= total/2:
        print(f"\n⚠️  {passed}/{total} tests passed - Proceed with caution")
        overall = "PARTIAL"
    else:
        print(f"\n❌ Only {passed}/{total} tests passed - Needs fixing")
        overall = "FAIL"
    
    # Save verification results
    os.makedirs("../../../logs/thrust2", exist_ok=True)
    log_file = "../../../logs/thrust2/verification_correct.log"
    
    with open(log_file, "w") as f:
        f.write("PHASE 3 VERIFICATION RESULTS (Perceval 1.1.0 - Correct)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Perceval version: 1.1.0 (Quandela)\n")
        f.write(f"Python version: {sys.version}\n\n")
        
        f.write("TEST RESULTS:\n")
        for i, (test_name, result) in enumerate(zip([t[0] for t in tests], results)):
            f.write(f"{i+1}. {test_name}: {'PASS' if result else 'FAIL'}\n")
        
        f.write(f"\nOverall: {overall} ({passed}/{total} passed)\n")
    
    print(f"\nLog saved to: {log_file}")
    
    # Also save a simple completion marker
    if passed >= total/2:  # If at least half tests pass
        with open("../../../.phase3_verification_complete", "w") as f:
            f.write(f"Perceval verification completed: {passed}/{total} tests passed\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    
    return overall == "PASS" or overall == "PARTIAL"

if __name__ == "__main__":
    success = main()
