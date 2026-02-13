"""
Subalgebra Identification Module - FIXED VERSION
Correctly handles realistic noise profiles with high-noise qubits
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

class SubalgebraIdentifier:
    """
    Identifies adaptive subalgebra based on REALISTIC noise profiles
    Correctly counts low-noise and high-noise qubits
    """
    
    def __init__(self, noise_threshold=0.001):
        self.noise_threshold = noise_threshold
        self.noise_profiles = {}
        self.load_noise_profiles()
        
    def load_noise_profiles(self):
        """Load realistic noise profiles from Phase 2"""
        noise_path = "experiments/thrust1/noise_profiles/"
        qubits_list = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
        
        for n in qubits_list:
            try:
                df = pd.read_csv(f"{noise_path}/noise_profile_{n}q.csv")
                self.noise_profiles[n] = df['depolarizing_rate'].values
                low = np.sum(df['depolarizing_rate'].values <= self.noise_threshold)
                high = n - low
                print(f"✅ Loaded noise profile for n={n}: {low} low, {high} high")
            except Exception as e:
                print(f"❌ Error loading n={n}: {e}")
                raise
    
    def identify_low_noise_qubits(self, n_qubits):
        """Return indices of qubits with noise below threshold"""
        if n_qubits not in self.noise_profiles:
            self.load_noise_profiles()
        
        profile = self.noise_profiles[n_qubits]
        low_noise_indices = np.where(profile <= self.noise_threshold)[0]
        high_noise_indices = np.where(profile > self.noise_threshold)[0]
        
        k = len(low_noise_indices)
        h = len(high_noise_indices)
        
        return low_noise_indices, k, h
    
    def estimate_adaptive_subalgebra_dimension(self, n_qubits):
        """
        Compute dimension of adaptive subalgebra
        Only qubits with noise ≤ λ* can participate
        """
        low_noise_qubits, k, h = self.identify_low_noise_qubits(n_qubits)
        
        print(f"   n={n_qubits:3d}: k={k:3d} low-noise, h={h:3d} high-noise", end='')
        
        if k <= 1:
            dim = 3 * k
            print(f" → dim={dim:4d} (only single-qubit)")
            return dim, k, h
        
        # Nearest-neighbor connectivity
        # Count connected pairs among low-noise qubits
        connected_pairs = 0
        for i in range(len(low_noise_qubits)):
            for j in range(i+1, len(low_noise_qubits)):
                q1 = low_noise_qubits[i]
                q2 = low_noise_qubits[j]
                if abs(q1 - q2) == 1:  # Nearest neighbor
                    connected_pairs += 1
        
        # Each connected pair can generate SU(4) algebra (dim=15)
        # Each qubit has 3 single-qubit Paulis
        dim = 15 * connected_pairs + 3 * k
        
        print(f" → dim={dim:4d} (pairs={connected_pairs})")
        
        return dim, k, h
    
    def compute_adaptive_scaling(self):
        """
        Compute adaptive subalgebra dimensions for all n
        THIS IS THE KEY RESULT FOR THEOREM 1
        """
        qubits_list = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
        results = []
        
        print("\n" + "="*70)
        print("📊 COMPUTING ADAPTIVE SUBALGEBRA SCALING")
        print("="*70)
        
        # Load DLA dimensions from B1
        try:
            df_dla = pd.read_csv('experiments/physics_analysis/lie_algebra/dla_computation/dla_dimension_scaling_PAPER.csv')
            dla_dict = dict(zip(df_dla['qubits'], df_dla['hardware_efficient_dla']))
        except:
            print("⚠️  Using theoretical bounds for full DLA")
            dla_dict = {}
        
        for n in qubits_list:
            dim, k, h = self.estimate_adaptive_subalgebra_dimension(n)
            
            # Get full DLA dimension
            if n in dla_dict:
                full_dla = dla_dict[n]
            else:
                full_dla = 4**n - 1
            
            reduction_factor = full_dla / dim
            log_reduction = np.log10(reduction_factor)
            
            results.append({
                'qubits': n,
                'low_noise_qubits': k,
                'high_noise_qubits': h,
                'adaptive_dim': dim,
                'full_dla_dim': full_dla,
                'reduction_factor': f"{reduction_factor:.2e}",
                'log10_reduction': round(log_reduction, 1)
            })
        
        # Save results
        df = pd.DataFrame(results)
        output_file = 'experiments/physics_analysis/lie_algebra/subalgebra_identification/adaptive_subalgebra_scaling_fixed.csv'
        df.to_csv(output_file, index=False)
        
        print("\n" + "="*70)
        print("✅ ADAPTIVE SUBALGEBRA SCALING COMPLETE")
        print("="*70)
        print(df[['qubits', 'low_noise_qubits', 'high_noise_qubits', 'adaptive_dim', 'log10_reduction']].to_string(index=False))
        
        return df

if __name__ == "__main__":
    print("="*70)
    print("  🚀 STEP B2: SUBALGEBRA IDENTIFICATION (FIXED)")
    print("="*70)
    
    # Initialize with IBM Eagle threshold (0.1%)
    identifier = SubalgebraIdentifier(noise_threshold=0.001)
    
    # Compute adaptive scaling for all n
    df_scaling = identifier.compute_adaptive_scaling()
    
    print("\n✅ STEP B2 COMPLETE (FIXED)")
    print("📁 Results saved to:")
    print("   experiments/physics_analysis/lie_algebra/subalgebra_identification/")
    print("   └── adaptive_subalgebra_scaling_fixed.csv")
    
    # Print key result for paper
    n100 = df_scaling[df_scaling['qubits'] == 100].iloc[0]
    print("\n🔬 KEY RESULT FOR PAPER:")
    print(f"   At n=100 qubits:")
    print(f"   - Low-noise qubits: {n100['low_noise_qubits']}")
    print(f"   - High-noise qubits: {n100['high_noise_qubits']}")
    print(f"   - Adaptive subalgebra dimension: {n100['adaptive_dim']}")
    print(f"   - Hilbert space reduction: 10^{n100['log10_reduction']}×")
