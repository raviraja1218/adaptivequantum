"""Test that all required packages can be imported."""

def test_import_qiskit():
    import qiskit
    assert qiskit.__version__ is not None

def test_import_torch():
    import torch
    assert torch.__version__ is not None

def test_import_tensorflow():
    import tensorflow as tf
    assert tf.__version__ is not None

def test_import_numpy():
    import numpy as np
    assert np.__version__ is not None

def test_import_pandas():
    import pandas as pd
    assert pd.__version__ is not None
