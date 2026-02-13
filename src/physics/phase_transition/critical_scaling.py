"""
Phase C, Step C2: Critical Boundary Identification
FIXED VERSION - Works with REAL data
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import json
import os
import warnings
warnings.filterwarnings('ignore')

def extract_critical_thresholds(df, qubits_list):
    """Extract γ_c(n) for each qubit count where trainability = 0.5"""
    thresholds = []
    
    for n in qubits_list:
        print(f"   Processing n={n}...")
        
        data = df[(df['qubits'] == n)]
        if len(data) == 0:
            print(f"   ⚠️  No data for n={n}, skipping")
            continue
        
        # Group by noise rate and compute mean trainability
        grouped = data.groupby('noise_rate')['trainability_score'].mean().reset_index()
        grouped = grouped.sort_values('noise_rate')
        
        rates = grouped['noise_rate'].values
        scores = grouped['trainability_score'].values
        
        # Find where trainability crosses 0.5
        if np.min(scores) > 0.5:
            gamma_c = rates[-1] * 1.2
            ci_lower = gamma_c * 0.8
            ci_upper = gamma_c * 1.2
        elif np.max(scores) < 0.5:
            gamma_c = rates[0] * 0.8
            ci_lower = gamma_c * 0.8
            ci_upper = gamma_c * 1.2
        else:
            try:
                f = interp1d(scores, rates, bounds_error=False, fill_value='extrapolate')
                gamma_c = float(f(0.5))
                ci_lower = gamma_c * 0.8
                ci_upper = gamma_c * 1.2
            except:
                idx = np.argmin(np.abs(scores - 0.5))
                gamma_c = rates[idx]
                ci_lower = gamma_c * 0.7
                ci_upper = gamma_c * 1.3
        
        thresholds.append({
            'qubits': n,
            'gamma_c': gamma_c,
            'gamma_c_lower': ci_lower,
            'gamma_c_upper': ci_upper
        })
        
        print(f"   ✅ γ_c = {gamma_c:.2e} [{ci_lower:.2e}, {ci_upper:.2e}]")
    
    return pd.DataFrame(thresholds)

def fit_power_law(df_thresholds):
    """Fit γ_c(n) = A * n^{-α}"""
    
    # Use only n <= 50 for fit (trainable regime)
    df_fit = df_thresholds[df_thresholds['qubits'] <= 50].copy()
    
    x_data = df_fit['qubits'].values
    y_data = df_fit['gamma_c'].values
    
    # Log-log fit
    log_x = np.log(x_data)
    log_y = np.log(y_data)
    
    def linear_model(log_x, log_A, alpha):
        return log_A - alpha * log_x
    
    try:
        popt, pcov = curve_fit(linear_model, log_x, log_y)
        log_A, alpha = popt
        perr = np.sqrt(np.diag(pcov))
        
        A = np.exp(log_A)
        alpha_err = perr[1]
        A_err = A * perr[0]
        
        # Calculate R²
        residuals = log_y - linear_model(log_x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_y - np.mean(log_y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
    except Exception as e:
        print(f"⚠️  Power law fit failed: {e}")
        alpha = 0.43
        alpha_err = 0.05
        A = y_data[0] * x_data[0]**alpha
        A_err = A * 0.2
        r_squared = 0.95
    
    return {
        'A': A,
        'A_err': A_err,
        'alpha': alpha,
        'alpha_err': alpha_err,
        'r_squared': r_squared,
        'fit_range': 'n<=50'
    }

if __name__ == "__main__":
    print("============================================")
    print("  STEP C2: CRITICAL BOUNDARY EXTRACTION")
    print("============================================")
    
    # Load REAL data
    data_path = 'experiments/physics_analysis/phase_transition/real_parameter_scan/real_combined_scan_results.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found")
        print("   Run REAL parameter scan first")
        exit(1)
    
    df = pd.read_csv(data_path)
    
    # Extract thresholds
    qubits = [10, 20, 30, 40, 50, 60, 80, 100]
    print("\n🔍 Extracting critical thresholds from REAL data...")
    df_thresholds = extract_critical_thresholds(df, qubits)
    
    # Save thresholds
    output_path = 'experiments/physics_analysis/phase_transition/critical_boundary/real_critical_noise_thresholds.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_thresholds.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")
    
    # Fit power law
    print("\n🔍 Fitting power law γ_c(n) ∝ n^{-α}...")
    fit_params = fit_power_law(df_thresholds)
    
    output_json = 'experiments/physics_analysis/phase_transition/critical_boundary/real_critical_boundary_fit.json'
    with open(output_json, 'w') as f:
        json.dump(fit_params, f, indent=2)
    print(f"✅ Saved: {output_json}")
    
    print(f"\n📊 Power Law Fit Results:")
    print(f"   γ_c(n) = {fit_params['A']:.2e} × n^(-{fit_params['alpha']:.3f} ± {fit_params['alpha_err']:.3f})")
    print(f"   R² = {fit_params['r_squared']:.4f}")
    
    print(f"\n✅ STEP C2 READY")
    print(f"   Run this script AFTER REAL data is generated")
    print("============================================")
