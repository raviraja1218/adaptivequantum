#!/usr/bin/env python3
"""
Simple script to generate noise profiles without complex dependencies
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def generate_realistic_noise_profile(n_qubits):
    """Generate realistic quantum noise profile"""
    print(f"Generating noise profile for {n_qubits} qubits...")
    
    profile = []
    
    # Base parameters (realistic for superconducting qubits)
    base_T1 = 100.0  # microseconds
    base_T2 = 80.0   # microseconds
    base_depol = 0.001
    base_dephase = 0.0005
    base_gate_error_1q = 0.005  # 0.5%
    base_gate_error_2q = 0.015  # 1.5%
    
    for q in range(n_qubits):
        # Add realistic variations
        # 1. Chip position effect: edges are worse
        is_edge = (q == 0 or q == n_qubits - 1)
        
        # 2. Periodic variations (simulating fabrication variations)
        period = 5
        is_worst_in_group = (q % period == 2)
        is_best_in_group = (q % period == 0)
        
        # Calculate parameters with variations
        T1 = base_T1 * np.random.uniform(0.8, 1.2)
        T2 = base_T2 * np.random.uniform(0.7, 1.1)
        
        if is_edge:
            T1 *= 0.9
            T2 *= 0.85
        
        if is_worst_in_group:
            T1 *= 0.85
            T2 *= 0.8
        
        if is_best_in_group:
            T1 *= 1.1
            T2 *= 1.05
        
        # Ensure T2 <= 2*T1 (physical constraint)
        T2 = min(T2, 2 * T1)
        
        # Gate errors (correlated with coherence times)
        gate_error_1q = base_gate_error_1q * (100 / T1) * np.random.uniform(0.8, 1.2)
        gate_error_2q = base_gate_error_2q * (100 / T1) * np.random.uniform(0.8, 1.2)
        
        # Depolarizing and dephasing (correlated)
        depolarizing_prob = base_depol * (100 / T1) * np.random.uniform(0.7, 1.3)
        dephasing_prob = base_dephase * (80 / T2) * np.random.uniform(0.7, 1.3)
        
        # Readout error (typically higher)
        readout_error = np.random.uniform(0.01, 0.03)
        
        profile.append({
            'qubit': q,
            'T1': float(T1),
            'T2': float(T2),
            'depolarizing_prob': float(depolarizing_prob),
            'dephasing_prob': float(dephasing_prob),
            'gate_error_1q': float(gate_error_1q),
            'gate_error_2q': float(gate_error_2q),
            'readout_error': float(readout_error),
            'is_edge': is_edge,
            'coherence_limited': T2 < T1 * 0.7
        })
    
    return pd.DataFrame(profile)

def save_profile(df, n_qubits, output_dir):
    """Save noise profile with metadata"""
    output_dir = Path(output_dir) / f"{n_qubits}q"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / f"noise_profile_{n_qubits}q_final.csv"
    df.to_csv(csv_path, index=False)
    
    # Save metadata
    metadata = {
        'n_qubits': n_qubits,
        'timestamp': datetime.now().isoformat(),
        'avg_T1': float(df['T1'].mean()),
        'avg_T2': float(df['T2'].mean()),
        'avg_gate_error_1q': float(df['gate_error_1q'].mean()),
        'avg_gate_error_2q': float(df['gate_error_2q'].mean()),
        'min_T1': float(df['T1'].min()),
        'max_T1': float(df['T1'].max()),
        'min_T2': float(df['T2'].min()),
        'max_T2': float(df['T2'].max()),
        'n_edge_qubits': int(df['is_edge'].sum()),
        'n_coherence_limited': int(df['coherence_limited'].sum())
    }
    
    metadata_path = output_dir / f"metadata_{n_qubits}q.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {n_qubits}-qubit profile to {csv_path}")
    print(f"  Avg T1: {metadata['avg_T1']:.1f} μs, Avg T2: {metadata['avg_T2']:.1f} μs")
    print(f"  Avg 1q error: {metadata['avg_gate_error_1q']*100:.3f}%, Avg 2q error: {metadata['avg_gate_error_2q']*100:.3f}%")
    
    return csv_path

def main():
    """Generate noise profiles for all qubit sizes"""
    output_dir = Path("experiments/thrust1/noise_profiles")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    qubit_sizes = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    
    print("=" * 60)
    print("Generating quantum noise profiles for AdaptiveQuantum")
    print("=" * 60)
    
    all_profiles = {}
    
    for n_qubits in qubit_sizes:
        print(f"\n{'='*40}")
        print(f"Processing {n_qubits} qubits")
        print('='*40)
        
        # Generate profile
        df = generate_realistic_noise_profile(n_qubits)
        
        # Save profile
        csv_path = save_profile(df, n_qubits, output_dir)
        
        all_profiles[n_qubits] = {
            'dataframe': df,
            'filepath': str(csv_path)
        }
    
    # Create summary file
    summary = {
        'generated_profiles': len(all_profiles),
        'qubit_sizes': qubit_sizes,
        'output_directory': str(output_dir.absolute()),
        'generation_time': datetime.now().isoformat()
    }
    
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Successfully generated {len(all_profiles)} noise profiles")
    print(f"Summary saved to: {summary_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
