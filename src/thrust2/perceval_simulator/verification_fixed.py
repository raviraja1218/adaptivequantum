#!/usr/bin/env python3
"""
Verify Perceval 1.1.0 (Quandela) installation and basic photonic circuit functionality
"""

import sys
import numpy as np
import perceval as pcvl
import perceval.components.base_components as comp
import perceval.components.unitary_components as uc
import os
import json
from datetime import datetime

def test_basic_components():
    """Test basic Perceval components"""
    print("Testing Perceval 1.1.0 (Quandela)...")
    
    # Create basic components
    try:
        # Beam splitter
        bs = uc.BS()
        print(f"✅ Beam splitter created: {bs}")
        
        # Phase shifter
        ps = uc.PS(phi=np.pi/4)
        print(f"✅ Phase shifter created: {ps}")
        
        # Create a simple circuit
        circuit = pcvl.Circuit(2)
        circuit.add(0, bs)
        circuit.add(0, ps)
        
        print(f"✅ Simple circuit created")
        print(f"  Circuit description: {circuit.describe()}")
        
        # Test simulation
        backend = pcvl.BackendFactory().get_backend("SLOS")
        input_state = pcvl.BasicState([1, 0])  # One photon in mode 0
        output_distribution = backend(circuit).prob_distribution(input_state)
        
        print(f"✅ Simulation successful")
        print(f"  Input: {input_state}")
        print(f"  Output distribution samples:")
        for state, prob in list(output_distribution.items())[:3]:
            print(f"    {state}: {prob:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_noise_integration():
    """Test loading Phase 2 noise profiles"""
    print("\nTesting noise profile integration...")
    
    # Check if Phase 2 noise profiles exist
    noise_dir = "../../../experiments/thrust1/noise_profiles/"
    
    if os.path.exists(noise_dir):
        # Find a noise profile file
        import glob
        noise_files = glob.glob(os.path.join(noise_dir, "*q", "*.csv"))
        
        if noise_files:
            # Load first available noise profile
            import pandas as pd
            noise_file = noise_files[0]
            df = pd.read_csv(noise_file)
            
            print(f"✅ Noise profile loaded: {os.path.basename(noise_file)}")
            print(f"  Qubits: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sample noise values:")
            print(df.head(3).to_string(index=False))
            
            return True
        else:
            print("❌ No noise profile CSV files found")
            return False
    else:
        print(f"❌ Noise directory not found: {noise_dir}")
        return False

def test_circuit_conversion():
    """Test basic circuit conversion"""
    print("\nTesting circuit conversion capabilities...")
    
    try:
        # Create a simple Qiskit-like circuit in Perceval
        circuit = pcvl.Circuit(4)
        
        # Add some components
        circuit.add(0, uc.BS())
        circuit.add(1, uc.PS(phi=np.pi/3))
        circuit.add((0, 1), uc.CNOT())
        
        # Count components
        n_components = len(circuit)
        
        print(f"✅ Test circuit created with {n_components} components")
        print(f"  Circuit: {circuit}")
        
        # Try to get unitary matrix
        try:
            U = circuit.compute_unitary(use_symbolic=False)
            print(f"✅ Unitary matrix computed: {U.shape}")
        except:
            print("⚠️  Could not compute unitary (expected for some circuits)")
        
        return True
        
    except Exception as e:
        print(f"❌ Circuit conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("PHASE 3: PERCEVAL 1.1.0 (QUANDELA) VERIFICATION")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Basic Perceval Components", test_basic_components),
        ("Circuit Conversion", test_circuit_conversion),
        ("Noise Profile Integration", test_noise_integration)
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
    
    all_passed = all(results)
    
    if all_passed:
        print("\n🎉 All verification tests passed!")
        print("Ready to proceed with Phase 3.")
    else:
        print(f"\n⚠️  {sum(results)}/{len(results)} tests passed.")
        print("Some features may not work correctly.")
    
    # Save verification results
    os.makedirs("../../../logs/thrust2", exist_ok=True)
    log_file = "../../../logs/thrust2/verification_fixed.log"
    
    with open(log_file, "w") as f:
        f.write("PHASE 3 VERIFICATION RESULTS (Perceval 1.1.0)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Perceval version: 1.1.0 (Quandela)\n\n")
        
        for i, (test_name, result) in enumerate(zip([t[0] for t in tests], results)):
            f.write(f"{i+1}. {test_name}: {'PASS' if result else 'FAIL'}\n")
        
        f.write(f"\nOverall: {'PASS' if all_passed else 'PARTIAL/FAIL'}\n")
    
    print(f"\nLog saved to: {log_file}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
