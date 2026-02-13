#!/usr/bin/env python3
"""
Main script to run all noise characterization steps
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.thrust1.noise_characterization.randomized_benchmarking import RandomizedBenchmarking
from src.thrust1.noise_characterization.state_tomography import StateTomographyCharacterization
from src.thrust1.noise_characterization.noise_profile_extractor import NoiseProfileExtractor

def run_characterization(n_qubits, output_dir, skip_rb=False, skip_tomo=False):
    """Run complete noise characterization pipeline"""
    print(f"=== Noise Characterization for {n_qubits} qubits ===")
    
    # Create output directory
    output_path = Path(output_dir) / f"{n_qubits}q"
    output_path.mkdir(parents=True, exist_ok=True)
    
    rb_data = None
    tomo_data = None
    
    # Run Randomized Benchmarking
    if not skip_rb:
        print("\n1. Running Randomized Benchmarking...")
        try:
            rb = RandomizedBenchmarking(n_qubits=n_qubits)
            rb_data = rb.characterize_all_qubits(str(output_path))
            print("✓ Randomized Benchmarking complete")
        except Exception as e:
            print(f"✗ Randomized Benchmarking failed: {e}")
    
    # Run State Tomography
    if not skip_tomo:
        print("\n2. Running State Tomography...")
        try:
            tomo = StateTomographyCharacterization(n_qubits=n_qubits)
            tomo_data = tomo.characterize_qubits(str(output_path))
            print("✓ State Tomography complete")
        except Exception as e:
            print(f"✗ State Tomography failed: {e}")
    
    # Create unified profile
    print("\n3. Creating unified noise profile...")
    extractor = NoiseProfileExtractor(str(output_path))
    unified_profile = extractor.create_unified_profile(n_qubits, rb_data, tomo_data)
    
    # Save unified profile
    profile_file = output_path / f"noise_profile_{n_qubits}q_final.csv"
    unified_profile.to_csv(profile_file, index=False)
    print(f"✓ Unified profile saved to {profile_file}")
    
    # Print summary
    print("\n=== Characterization Summary ===")
    print(f"Qubits: {n_qubits}")
    print(f"Average T1: {unified_profile['T1'].mean():.1f} μs")
    print(f"Average T2: {unified_profile['T2'].mean():.1f} μs")
    print(f"Average 1-qubit gate error: {unified_profile['gate_error_1q'].mean()*100:.3f}%")
    print(f"Average 2-qubit gate error: {unified_profile['gate_error_2q'].mean()*100:.3f}%")
    
    return unified_profile

def main():
    parser = argparse.ArgumentParser(description="Run quantum noise characterization")
    parser.add_argument("--n_qubits", type=int, default=5, 
                       help="Number of qubits to characterize")
    parser.add_argument("--output_dir", type=str, 
                       default="experiments/thrust1/noise_profiles",
                       help="Output directory for noise profiles")
    parser.add_argument("--skip_rb", action="store_true",
                       help="Skip Randomized Benchmarking")
    parser.add_argument("--skip_tomo", action="store_true",
                       help="Skip State Tomography")
    parser.add_argument("--all_sizes", action="store_true",
                       help="Run for all qubit sizes (5-100)")
    
    args = parser.parse_args()
    
    if args.all_sizes:
        # Run for all qubit sizes from our paper
        qubit_sizes = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
        print(f"Running characterization for {len(qubit_sizes)} qubit sizes...")
        
        for n_q in qubit_sizes:
            print(f"\n{'='*60}")
            run_characterization(n_q, args.output_dir, args.skip_rb, args.skip_tomo)
            
        print(f"\n{'='*60}")
        print("All characterizations complete!")
        
        # Also generate unified extractor profiles
        extractor = NoiseProfileExtractor(args.output_dir)
        extractor.generate_profiles_for_sizes(qubit_sizes)
        
    else:
        # Run for single qubit size
        run_characterization(args.n_qubits, args.output_dir, args.skip_rb, args.skip_tomo)

if __name__ == "__main__":
    main()
