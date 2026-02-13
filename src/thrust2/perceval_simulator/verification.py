#!/usr/bin/env python3
"""
Verify Perceval installation and basic photonic circuit functionality
"""

import sys
import numpy as np
import perceval as pcvl
import perceval.components.unitary_components as comp

def test_basic_components():
    """Test basic Perceval components"""
    print("Testing Perceval installation...")
    
    # Create basic components
    try:
        # Beam splitter
        bs = comp.BS()
        print(f"✅ Beam splitter created: {bs}")
        
        # Phase shifter
        ps = comp.PS(phi=np.pi/4)
        print(f"✅ Phase shifter created: {ps}")
        
        # Create a simple circuit
        circuit = pcvl.Circuit(2)
        circuit.add(0, bs)
        circuit.add(0, ps)
        
        print(f"✅ Simple circuit created: {circuit}")
        
        # Test simulation
        backend = pcvl.BackendFactory().get_backend("SLOS")
        input_state = pcvl.BasicState([1, 0])  # One photon in mode 0
        output_distribution = backend(circuit).probampli(input_state)
        
        print(f"✅ Simulation successful")
        print(f"  Input: {input_state}")
        print(f"  Output probability amplitude: {output_distribution}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_noise_integration():
    """Test loading Phase 2 noise profiles"""
    import os
    import pandas as pd
    
    print("\nTesting noise profile integration...")
    
    # Check if Phase 2 noise profiles exist
    noise_dir = "../../../experiments/thrust1/noise_profiles/"
    
    if os.path.exists(noise_dir):
        # Find a noise profile file
        noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.csv')]
        
        if noise_files:
            # Load first available noise profile
            noise_file = os.path.join(noise_dir, noise_files[0])
            df = pd.read_csv(noise_file)
            
            print(f"✅ Noise profile loaded: {noise_file}")
            print(f"  Qubits: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sample noise values:")
            print(df.head(3).to_string())
            
            return True
        else:
            print("❌ No noise profile files found")
            return False
    else:
        print(f"❌ Noise directory not found: {noise_dir}")
        return False

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("PHASE 3: PERCEVAL VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Basic Perceval Components", test_basic_components),
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
        
        # Save verification results
        with open("../../../logs/thrust2/verification.log", "w") as f:
            f.write("PHASE 3 VERIFICATION RESULTS\n")
            f.write("=" * 40 + "\n")
            for i, (test_name, result) in enumerate(zip([t[0] for t in tests], results)):
                f.write(f"{i+1}. {test_name}: {'PASS' if result else 'FAIL'}\n")
            f.write("\nAll tests passed: YES\n")
    else:
        print("\n⚠️  Some verification tests failed.")
        print("Please fix the issues before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    if not success:
        print("Verification failed. Exiting.")
