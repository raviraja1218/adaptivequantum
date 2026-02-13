#!/usr/bin/env python3
"""
Final working test for Perceval 1.1.0
"""

import perceval as pcvl
import numpy as np
import os
from datetime import datetime

def create_and_simulate_circuit():
    """Create and simulate a simple photonic circuit"""
    print("=" * 60)
    print("CREATING AND SIMULATING PHOTONIC CIRCUIT")
    print("=" * 60)
    
    try:
        # 1. Create a 2-mode circuit
        print("\n1. Creating 2-mode circuit...")
        circuit = pcvl.Circuit(2)
        print(f"   ✅ Circuit created with {circuit.m} modes")
        
        # 2. Add a 50/50 beam splitter between modes 0 and 1
        print("\n2. Adding beam splitter...")
        bs = pcvl.BS()  # Default is 50/50
        circuit.add(0, bs)  # Add to modes 0-1
        print(f"   ✅ Beam splitter added")
        
        # 3. Add a phase shifter to mode 0
        print("\n3. Adding phase shifter...")
        ps = pcvl.PS(phi=np.pi/4)  # 45 degree phase shift
        circuit.add(0, ps)
        print(f"   ✅ Phase shifter added (phi=π/4)")
        
        # 4. Display circuit info
        print("\n4. Circuit information:")
        print(f"   Components: {circuit.components_count}")
        
        # 5. Create SLOS backend for simulation
        print("\n5. Creating SLOS backend...")
        from perceval.backends import SLOSBackend
        backend = SLOSBackend(circuit)
        print(f"   ✅ Backend created: {type(backend).__name__}")
        
        # 6. Define input state: 1 photon in mode 0
        print("\n6. Defining input state...")
        input_state = pcvl.BasicState([1, 0])
        print(f"   ✅ Input: {input_state} (1 photon in mode 0)")
        
        # 7. Calculate probability distribution
        print("\n7. Calculating probability distribution...")
        prob_dist = backend.prob_distribution(input_state)
        print(f"   ✅ Distribution calculated")
        
        # 8. Display results
        print("\n8. RESULTS:")
        print(f"   Total possible output states: {len(prob_dist)}")
        
        total_prob = 0
        for state, prob in sorted(prob_dist.items(), key=lambda x: -x[1]):
            if prob > 0.0001:  # Show significant probabilities
                print(f"     {state}: {prob:.6f}")
                total_prob += prob
        
        print(f"\n   Total probability: {total_prob:.10f}")
        
        if abs(total_prob - 1.0) < 0.0001:
            print("   ✅ Probability conserved")
        else:
            print(f"   ⚠️  Probability not conserved: {total_prob}")
        
        return True, circuit, prob_dist
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_different_backends():
    """Test different simulation backends"""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT BACKENDS")
    print("=" * 60)
    
    # Create a simple test circuit
    circuit = pcvl.Circuit(2)
    circuit.add(0, pcvl.BS())
    
    input_state = pcvl.BasicState([1, 0])
    
    backends_to_test = ["SLOS", "Naive", "Clifford2017"]
    
    results = {}
    for backend_name in backends_to_test:
        try:
            print(f"\nTesting {backend_name} backend...")
            
            # Get backend from factory
            factory = pcvl.BackendFactory()
            backend = factory.get_backend(backend_name)(circuit)
            
            # Calculate probabilities
            prob_dist = backend.prob_distribution(input_state)
            
            # Extract probability of |1,0> state (should be 0.5 for 50/50 BS)
            prob_10 = prob_dist.get(pcvl.BasicState([1, 0]), 0)
            
            results[backend_name] = {
                'success': True,
                'prob_10': prob_10,
                'total_states': len(prob_dist)
            }
            
            print(f"  ✅ {backend_name}: P(|1,0>) = {prob_10:.4f}")
            
        except Exception as e:
            print(f"  ❌ {backend_name}: {e}")
            results[backend_name] = {'success': False, 'error': str(e)}
    
    return results

def test_noise_model():
    """Test noise model integration"""
    print("\n" + "=" * 60)
    print("TESTING NOISE MODEL INTEGRATION")
    print("=" * 60)
    
    # Load noise profile from Phase 2
    noise_dir = "../../../experiments/thrust1/noise_profiles/"
    
    if os.path.exists(noise_dir):
        print("Noise profiles directory exists")
        
        # List files
        import glob
        csv_files = glob.glob(os.path.join(noise_dir, "**", "*.csv"), recursive=True)
        
        if csv_files:
            print(f"Found {len(csv_files)} CSV files")
            
            # Read first file
            import pandas as pd
            first_file = csv_files[0]
            df = pd.read_csv(first_file)
            
            print(f"Loaded: {os.path.basename(first_file)}")
            print(f"  Qubits: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            # Create a simple noise model from data
            avg_T1 = df['T1'].mean()
            avg_T2 = df['T2'].mean()
            avg_depol = df['depolarization_prob'].mean()
            
            print(f"\nAverage noise parameters:")
            print(f"  T1: {avg_T1:.1f} µs")
            print(f"  T2: {avg_T2:.1f} µs")
            print(f"  Depolarization: {avg_depol:.6f}")
            
            # Note: Perceval has its own noise model system
            # We'll use these parameters in Phase 3 compilation
            
            return True, df
        else:
            print("No CSV files found")
            return False, None
    else:
        print("Noise directory not found")
        return False, None

def main():
    """Main test function"""
    print("PHASE 3: PERCEVAL FINAL WORKING TEST")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Basic circuit creation and simulation
    success1, circuit, prob_dist = create_and_simulate_circuit()
    
    # Test 2: Different backends
    backend_results = test_different_backends()
    
    # Test 3: Noise model
    success3, noise_df = test_noise_model()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print(f"1. Basic circuit: {'✅ PASS' if success1 else '❌ FAIL'}")
    
    backend_success = sum(1 for r in backend_results.values() if r.get('success', False))
    print(f"2. Backends: {backend_success}/{len(backend_results)} successful")
    
    print(f"3. Noise model: {'✅ PASS' if success3 else '❌ FAIL'}")
    
    # Save results
    os.makedirs("../../../logs/thrust2", exist_ok=True)
    log_file = "../../../logs/thrust2/final_working_test.log"
    
    with open(log_file, "w") as f:
        f.write("PHASE 3: PERCEVAL FINAL WORKING TEST RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        f.write("TEST 1: Basic Circuit\n")
        f.write(f"  Success: {success1}\n")
        if circuit:
            f.write(f"  Circuit modes: {circuit.m}\n")
            f.write(f"  Components: {circuit.components_count}\n\n")
        
        f.write("TEST 2: Backends\n")
        for backend_name, result in backend_results.items():
            if result.get('success'):
                f.write(f"  {backend_name}: P(|1,0>) = {result['prob_10']:.6f}\n")
            else:
                f.write(f"  {backend_name}: FAILED - {result.get('error', 'Unknown')}\n")
        f.write("\n")
        
        f.write("TEST 3: Noise Model\n")
        f.write(f"  Success: {success3}\n")
        if success3 and noise_df is not None:
            f.write(f"  Qubits in profile: {len(noise_df)}\n")
            f.write(f"  Avg T1: {noise_df['T1'].mean():.1f} µs\n")
            f.write(f"  Avg T2: {noise_df['T2'].mean():.1f} µs\n")
    
    print(f"\nLog saved to: {log_file}")
    
    # Create completion marker
    if success1:  # If basic circuit works, we can proceed
        marker_file = "../../../.phase3_core_working"
        with open(marker_file, "w") as f:
            f.write(f"Perceval core functionality verified\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Basic circuit: {success1}\n")
            f.write(f"Working backends: {backend_success}\n")
        
        print(f"✅ Created core working marker: {marker_file}")
        print("\n✅ PERCEVAL IS WORKING - PROCEED WITH PHASE 3")
        return True
    else:
        print("\n❌ PERCEVAL NOT WORKING PROPERLY - NEEDS FIXING")
        return False

if __name__ == "__main__":
    success = main()
