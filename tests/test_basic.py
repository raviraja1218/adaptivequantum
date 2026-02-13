"""Basic tests for AdaptiveQuantum."""

def test_numpy_import():
    """Test NumPy import."""
    import numpy as np
    assert np.__version__ == '1.26.0'
    return True

def test_qiskit_import():
    """Test Qiskit import."""
    import qiskit
    assert hasattr(qiskit, '__version__')
    return True

def test_torch_import():
    """Test PyTorch import."""
    import torch
    assert hasattr(torch, '__version__')
    return True

def test_tensorflow_import():
    """Test TensorFlow import."""
    import tensorflow as tf
    assert hasattr(tf, '__version__')
    return True

def test_quantum_circuit():
    """Test basic quantum circuit creation."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    assert qc.num_qubits == 2
    assert qc.depth() == 2
    return True

def test_tensor_operations():
    """Test basic tensor operations."""
    import torch
    import numpy as np
    
    # Test PyTorch
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.shape == torch.Size([3])
    
    # Test NumPy
    y = np.array([1, 2, 3])
    assert y.shape == (3,)
    
    return True

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("NumPy Import", test_numpy_import),
        ("Qiskit Import", test_qiskit_import),
        ("PyTorch Import", test_torch_import),
        ("TensorFlow Import", test_tensorflow_import),
        ("Quantum Circuit", test_quantum_circuit),
        ("Tensor Operations", test_tensor_operations),
    ]
    
    print("=" * 50)
    print("Running AdaptiveQuantum Basic Tests")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✓ {name}: PASSED")
                passed += 1
            else:
                print(f"✗ {name}: FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {name}: ERROR - {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
