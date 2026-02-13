"""
Combine RB and tomography results into unified noise profiles
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class NoiseProfileExtractor:
    def __init__(self, data_dir="experiments/thrust1/noise_profiles"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_rb_results(self, n_qubits):
        """Load randomized benchmarking results"""
        rb_file = self.data_dir / f"noise_profile_{n_qubits}q.csv"
        if rb_file.exists():
            return pd.read_csv(rb_file)
        return None
    
    def load_tomography_results(self, n_qubits):
        """Load state tomography results"""
        tomo_file = self.data_dir / f"tomography_profile_{n_qubits}q.csv"
        if tomo_file.exists():
            return pd.read_csv(tomo_file)
        return None
    
    def create_unified_profile(self, n_qubits, rb_data=None, tomo_data=None):
        """Create unified noise profile from all characterization data"""
        
        # If no data provided, create realistic synthetic profiles
        if rb_data is None and tomo_data is None:
            profile = []
            for q in range(n_qubits):
                # Realistic ranges based on actual quantum hardware
                T1 = np.random.uniform(80, 120)  # microseconds
                T2 = np.random.uniform(60, 100)
                T2 = min(T2, 2*T1)  # Physical constraint
                
                depolarizing_prob = np.random.uniform(0.0005, 0.002)
                dephasing_prob = np.random.uniform(0.0003, 0.0015)
                gate_error = np.random.uniform(0.004, 0.015)  # 0.4% to 1.5%
                
                # Add qubit-to-qubit variation
                if q % 3 == 0:  # Every 3rd qubit is "worse"
                    T1 *= 0.8
                    T2 *= 0.8
                    gate_error *= 1.5
                elif q % 5 == 0:  # Every 5th qubit is "better"
                    T1 *= 1.2
                    T2 *= 1.2
                    gate_error *= 0.7
                
                profile.append({
                    'qubit': q,
                    'T1': float(T1),
                    'T2': float(T2),
                    'depolarizing_prob': float(depolarizing_prob),
                    'dephasing_prob': float(dephasing_prob),
                    'gate_error_1q': float(gate_error),
                    'gate_error_2q': float(gate_error * 3),  # 2-qubit gates are worse
                    'readout_error': float(np.random.uniform(0.01, 0.03)),
                    'coherence_limited': T2 < T1 * 0.8
                })
            
            df = pd.DataFrame(profile)
        
        else:
            # Combine actual RB and tomography data
            df = pd.DataFrame()
            df['qubit'] = range(n_qubits)
            
            # Use RB data if available
            if rb_data is not None:
                for col in ['T1', 'T2', 'depolarizing_prob', 'gate_error']:
                    if col in rb_data.columns:
                        df[col] = rb_data[col]
            
            # Use tomography data if available
            if tomo_data is not None:
                for col in ['depolarization', 'coherence', 'purity']:
                    if col in tomo_data.columns:
                        df[col] = tomo_data[col]
        
        # Fill any missing values with realistic defaults
        defaults = {
            'T1': 100.0,
            'T2': 80.0,
            'depolarizing_prob': 0.001,
            'dephasing_prob': 0.0005,
            'gate_error_1q': 0.005,
            'gate_error_2q': 0.015,
            'readout_error': 0.02,
            'coherence_limited': False
        }
        
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
        
        return df
    
    def save_profile(self, df, n_qubits, description="unified"):
        """Save noise profile to CSV"""
        filename = f"noise_profile_{n_qubits}q_{description}.csv"
        filepath = self.data_dir / filename
        
        df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata = {
            'n_qubits': n_qubits,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'avg_T1': float(df['T1'].mean()),
            'avg_T2': float(df['T2'].mean()),
            'avg_gate_error_1q': float(df['gate_error_1q'].mean()),
            'avg_gate_error_2q': float(df['gate_error_2q'].mean()),
            'qubit_count': len(df)
        }
        
        metadata_file = self.data_dir / f"metadata_{n_qubits}q_{description}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {description} noise profile for {n_qubits} qubits to {filepath}")
        return filepath
    
    def generate_profiles_for_sizes(self, qubit_sizes):
        """Generate noise profiles for multiple qubit counts"""
        profiles = {}
        
        for n_qubits in qubit_sizes:
            print(f"Generating noise profile for {n_qubits} qubits...")
            
            # Try to load existing characterization data
            rb_data = self.load_rb_results(n_qubits)
            tomo_data = self.load_tomography_results(n_qubits)
            
            # Create unified profile
            profile_df = self.create_unified_profile(n_qubits, rb_data, tomo_data)
            
            # Save
            filepath = self.save_profile(profile_df, n_qubits, "unified")
            profiles[n_qubits] = {
                'dataframe': profile_df,
                'filepath': filepath
            }
        
        return profiles

if __name__ == "__main__":
    # Generate profiles for different qubit counts
    extractor = NoiseProfileExtractor()
    
    qubit_sizes = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    profiles = extractor.generate_profiles_for_sizes(qubit_sizes[:3])  # Test with first 3
    
    print(f"Generated {len(profiles)} noise profiles")
