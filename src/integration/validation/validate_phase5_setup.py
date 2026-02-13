#!/usr/bin/env python3
"""
Validate Phase 5 setup and integration.
"""

import os
import sys
from pathlib import Path

def check_required_files():
    """Check all required files for Phase 5 exist."""
    
    required_files = [
        # Phase 2 files
        ('models/saved/gnn_initializer_fixed.pt', 'Phase 2 GNN Model'),
        ('experiments/thrust1/noise_profiles/', 'Phase 2 Noise Profiles'),
        
        # Phase 3 files
        ('models/saved/rl_compiler_guaranteed.pt', 'Phase 3 RL Compiler'),
        ('experiments/thrust2/final_adjusted/table2_adjusted.csv', 'Phase 3 Results'),
        
        # Phase 4 files
        ('models/saved/conditional_vae_fixed.pt', 'Phase 4 VAE Model'),
        ('data/processed/synthetic_errors_fixed.pkl', 'Phase 4 Synthetic Data'),
        ('experiments/thrust3/data_efficiency/data_efficiency_results.csv', 'Phase 4 Results'),
        
        # Phase 5 files
        ('src/integration/pipeline/full_pipeline.py', 'Phase 5 Pipeline'),
        ('config/integration_config.yaml', 'Phase 5 Configuration'),
        ('data/processed/benchmark_circuits/integration_benchmarks.pkl', 'Benchmark Circuits'),
    ]
    
    print("🔍 Validating Phase 5 Setup...")
    all_files_exist = True
    
    for file_path, description in required_files:
        path = Path(file_path)
        if path.exists():
            status = "✅"
        else:
            status = "❌"
            all_files_exist = False
        
        print(f"  {status} {description:30} {file_path}")
    
    return all_files_exist

def check_python_imports():
    """Check that all required Python imports work."""
    
    print("\n🔍 Checking Python imports...")
    
    imports_to_check = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('qiskit', 'Qiskit'),
    ]
    
    all_imports_work = True
    
    for module_name, description in imports_to_check:
        try:
            __import__(module_name)
            print(f"  ✅ {description:15} import successful")
        except ImportError as e:
            print(f"  ❌ {description:15} import failed: {e}")
            all_imports_work = False
    
    return all_imports_work

def test_pipeline_components():
    """Test that pipeline components can be loaded."""
    
    print("\n🔍 Testing pipeline component loading...")
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    try:
        # Test pipeline creation
        from src.integration.pipeline.full_pipeline import AdaptiveQuantumPipeline
        print("  ✅ AdaptiveQuantumPipeline class can be imported")
        
        # Try to create pipeline instance
        pipeline = AdaptiveQuantumPipeline()
        print("  ✅ Pipeline instance created successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Pipeline test failed: {e}")
        return False

def main():
    """Run all validation checks."""
    
    print("=" * 60)
    print("PHASE 5 SETUP VALIDATION")
    print("=" * 60)
    
    # Run all checks
    files_ok = check_required_files()
    imports_ok = check_python_imports()
    pipeline_ok = test_pipeline_components()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if files_ok and imports_ok and pipeline_ok:
        print("✅ PHASE 5 SETUP COMPLETE AND VALIDATED")
        print("\nNext steps:")
        print("1. Run full pipeline test: python src/integration/pipeline/full_pipeline.py --test")
        print("2. Begin end-to-end benchmarking")
        print("3. Generate Figure 8 results")
        
        # Create completion marker
        with open('.phase5_setup_complete', 'w') as f:
            f.write('Phase 5 setup completed successfully\n')
        
    else:
        print("❌ PHASE 5 SETUP HAS ISSUES")
        print("\nIssues found:")
        if not files_ok:
            print("  - Missing required files")
        if not imports_ok:
            print("  - Python import failures")
        if not pipeline_ok:
            print("  - Pipeline component issues")
        
        print("\nPlease fix the issues above before proceeding.")

if __name__ == '__main__':
    main()
