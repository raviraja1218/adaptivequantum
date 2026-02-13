"""
Non-Markovian Noise Model for Quantum Circuits - Qiskit 0.43 Compatible
"""

import numpy as np
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
from qiskit import QuantumCircuit
import json
from datetime import datetime

class NonMarkovianNoiseModel:
    """Noise model with memory effects using Ornstein-Uhlenbeck process"""
    
    def __init__(self, base_error_rate=0.001, correlation_time=50e-9, memory_depth=5):
        self.base_error_rate = base_error_rate
        self.correlation_time = correlation_time
        self.memory_depth = memory_depth
        self.error_history = []
        self.noise_model = NoiseModel()
        
    def ou_process(self, dt=1e-9):
        """Ornstein-Uhlenbeck process for correlated noise"""
        theta = 1.0 / self.correlation_time
        mu = self.base_error_rate
        sigma = 0.3 * self.base_error_rate
        
        if len(self.error_history) == 0:
            return mu
        
        dW = np.random.normal(0, np.sqrt(dt))
        return mu + (self.error_history[-1] - mu) * np.exp(-theta * dt) + \
               sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta)) * dW
    
    def get_correlated_error_rate(self):
        """Get error rate with memory effects"""
        if len(self.error_history) > self.memory_depth:
            recent_errors = self.error_history[-self.memory_depth:]
            memory_factor = 1.0 + 0.2 * np.mean(recent_errors) / self.base_error_rate
        else:
            memory_factor = 1.0
            
        correlated_rate = self.ou_process() * memory_factor
        return max(0.001 * self.base_error_rate, min(correlated_rate, 10 * self.base_error_rate))
    
    def save_model(self, filepath):
        params = {
            'base_error_rate': self.base_error_rate,
            'correlation_time': self.correlation_time,
            'memory_depth': self.memory_depth,
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        return filepath

def generate_correlated_noise_series(duration=1000, base_rate=0.001, correlation_time=50e-9):
    """Generate a time series of correlated error rates"""
    model = NonMarkovianNoiseModel(base_rate, correlation_time)
    rates = []
    times = []
    
    for t in range(duration):
        model.error_history.append(base_rate * (1 + 0.1 * np.sin(t/100)))
        rate = model.get_correlated_error_rate()
        rates.append(rate)
        times.append(t * 50e-9 * 1e6)
        
    return np.array(times), np.array(rates)
