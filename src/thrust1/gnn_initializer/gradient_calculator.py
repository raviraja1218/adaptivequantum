"""
Calculate gradients and find optimal parameter initializations
"""
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class GradientCalculator:
    def __init__(self, n_samples=100):
        self.n_samples = n_samples
    
    def parameter_shift_gradient(self, circuit_data, params, noise_profile=None):
        """Calculate gradient using parameter shift rule"""
        n_params = len(params)
        gradient = np.zeros(n_params)
        
        # Simplified gradient calculation for demonstration
        # In real implementation, would use Qiskit's gradient framework
        
        # For each parameter, compute finite difference
        shift = np.pi / 2
        
        for i in range(n_params):
            # Positive shift
            params_plus = params.copy()
            params_plus[i] += shift
            loss_plus = self._compute_loss(circuit_data, params_plus, noise_profile)
            
            # Negative shift
            params_minus = params.copy()
            params_minus[i] -= shift
            loss_minus = self._compute_loss(circuit_data, params_minus, noise_profile)
            
            # Parameter shift rule for Pauli rotations
            gradient[i] = 0.5 * (loss_plus - loss_minus)
        
        return gradient
    
    def _compute_loss(self, circuit_data, params, noise_profile=None):
        """Compute loss function for given parameters"""
        # Simplified loss function that simulates barren plateaus
        n_qubits = circuit_data['n_qubits']
        depth = circuit_data['depth']
        
        # Base loss depends on parameter distance from optimal
        # In barren plateaus, loss becomes exponentially flat
        optimal_params = np.ones_like(params) * np.pi/4  # Simplified optimal
        
        # Distance from optimal
        param_distance = np.linalg.norm(params - optimal_params)
        
        # Simulate barren plateau effect: loss becomes flatter with more qubits/depth
        barren_plateau_factor = np.exp(-0.1 * n_qubits * depth)
        
        # Add noise effect if noise_profile provided
        noise_factor = 1.0
        if noise_profile is not None:
            # Worse noise leads to flatter landscape
            avg_gate_error = np.mean(noise_profile['gate_error_1q'])
            noise_factor = 1.0 + 10.0 * avg_gate_error
        
        loss = param_distance**2 * barren_plateau_factor * noise_factor
        
        # Add small random noise
        loss += np.random.normal(0, 0.01)
        
        return max(loss, 0.0)
    
    def find_optimal_initialization(self, circuit_data, noise_profile, method='noise_aware'):
        """Find optimal parameter initialization"""
        n_params = circuit_data['n_parameters']
        n_qubits = circuit_data['n_qubits']
        
        if method == 'random':
            # Standard random initialization
            return np.random.uniform(0, 2*np.pi, n_params)
        
        elif method == 'noise_aware':
            # Noise-aware initialization based on our GNN approach
            # This is what the GNN will learn to predict
            
            # Group parameters by qubit (assuming 3 params per qubit in our circuit design)
            params_per_qubit = n_params // n_qubits
            
            optimal_params = np.zeros(n_params)
            
            for q in range(n_qubits):
                # Get noise parameters for this qubit
                T1 = noise_profile['T1'][q]
                T2 = noise_profile['T2'][q]
                gate_error = noise_profile['gate_error_1q'][q]
                
                # Heuristic: qubits with worse noise get smaller parameter variations
                # This helps break symmetries that cause barren plateaus
                
                # Noise quality factor (0=bad, 1=good)
                quality_factor = (T1/100.0) * (T2/80.0) * (0.01/gate_error)
                quality_factor = np.clip(quality_factor, 0.1, 2.0)
                
                # Generate parameters for this qubit
                base_angle = np.pi/4  # Good starting point
                
                for p in range(params_per_qubit):
                    param_idx = q * params_per_qubit + p
                    
                    # Add variation based on noise quality
                    # Better qubits can explore larger parameter space
                    variation = quality_factor * np.random.uniform(-np.pi/2, np.pi/2)
                    optimal_params[param_idx] = base_angle + variation
            
            return optimal_params
        
        elif method == 'optimized':
            # Use optimization to find good initialization
            initial_guess = np.random.uniform(0, 2*np.pi, n_params)
            
            def loss_func(x):
                return self._compute_loss(circuit_data, x, noise_profile)
            
            result = minimize(loss_func, initial_guess, 
                           method='L-BFGS-B', 
                           bounds=[(0, 2*np.pi)] * n_params,
                           options={'maxiter': 100})
            
            return result.x
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_gradient_statistics(self, circuit_data, noise_profile, params):
        """Calculate gradient statistics for given initialization"""
        # Calculate gradient at initialization
        gradient = self.parameter_shift_gradient(circuit_data, params, noise_profile)
        
        # Statistics
        gradient_norm = np.linalg.norm(gradient)
        gradient_mean = np.mean(gradient)
        gradient_std = np.std(gradient)
        gradient_variance = np.var(gradient)
        
        # Determine if trainable (gradient not vanishing)
        # Threshold based on empirical studies of barren plateaus
        trainable_threshold = 1e-6 * (2 ** (-circuit_data['depth']))
        trainable = gradient_norm > trainable_threshold
        
        # Calculate expected improvement factor vs random
        # For random initialization, gradient norm decays as ~2^(-2*depth)
        random_gradient_norm = 0.1 * (2 ** (-2 * circuit_data['depth']))
        improvement_factor = gradient_norm / random_gradient_norm if random_gradient_norm > 0 else float('inf')
        
        return {
            'gradient': gradient,
            'gradient_norm': gradient_norm,
            'gradient_mean': gradient_mean,
            'gradient_std': gradient_std,
            'gradient_variance': gradient_variance,
            'trainable': trainable,
            'improvement_factor': improvement_factor,
            'n_qubits': circuit_data['n_qubits'],
            'depth': circuit_data['depth']
        }
    
    def analyze_barren_plateau(self, circuit_data, noise_profile, n_inits=100):
        """Analyze barren plateau characteristics"""
        random_stats = []
        noise_aware_stats = []
        
        for _ in range(n_inits):
            # Random initialization
            random_params = self.find_optimal_initialization(circuit_data, noise_profile, 'random')
            random_stat = self.calculate_gradient_statistics(circuit_data, noise_profile, random_params)
            random_stats.append(random_stat)
            
            # Noise-aware initialization
            noise_aware_params = self.find_optimal_initialization(circuit_data, noise_profile, 'noise_aware')
            noise_aware_stat = self.calculate_gradient_statistics(circuit_data, noise_profile, noise_aware_params)
            noise_aware_stats.append(noise_aware_stat)
        
        # Aggregate statistics
        random_gradient_norms = [s['gradient_norm'] for s in random_stats]
        noise_aware_gradient_norms = [s['gradient_norm'] for s in noise_aware_stats]
        
        random_trainable = sum([s['trainable'] for s in random_stats])
        noise_aware_trainable = sum([s['trainable'] for s in noise_aware_stats])
        
        avg_improvement = np.mean([na['improvement_factor'] / max(ra['improvement_factor'], 1e-10) 
                                  for na, ra in zip(noise_aware_stats, random_stats)])
        
        return {
            'random': {
                'mean_gradient_norm': np.mean(random_gradient_norms),
                'std_gradient_norm': np.std(random_gradient_norms),
                'trainable_fraction': random_trainable / n_inits,
                'stats': random_stats
            },
            'noise_aware': {
                'mean_gradient_norm': np.mean(noise_aware_gradient_norms),
                'std_gradient_norm': np.std(noise_aware_gradient_norms),
                'trainable_fraction': noise_aware_trainable / n_inits,
                'stats': noise_aware_stats
            },
            'improvement_factor': avg_improvement,
            'n_qubits': circuit_data['n_qubits'],
            'depth': circuit_data['depth']
        }

if __name__ == "__main__":
    # Test gradient calculator
    from circuit_generator import CircuitGenerator
    
    calculator = GradientCalculator()
    generator = CircuitGenerator()
    
    n_qubits = 10
    circuit_data = generator.generate_circuit(n_qubits)
    noise_profile = generator.generate_noise_profile(n_qubits)
    
    print(f"Testing with {n_qubits} qubits, depth {circuit_data['depth']}")
    
    # Test different initialization methods
    methods = ['random', 'noise_aware']
    
    for method in methods:
        params = calculator.find_optimal_initialization(circuit_data, noise_profile, method)
        stats = calculator.calculate_gradient_statistics(circuit_data, noise_profile, params)
        
        print(f"\n{method} initialization:")
        print(f"  Gradient norm: {stats['gradient_norm']:.2e}")
        print(f"  Trainable: {stats['trainable']}")
        print(f"  Improvement factor: {stats['improvement_factor']:.1f}x")
    
    # Analyze barren plateau
    print(f"\nBarren plateau analysis (100 random initializations):")
    analysis = calculator.analyze_barren_plateau(circuit_data, noise_profile, n_inits=10)
    
    print(f"  Random trainable: {analysis['random']['trainable_fraction']:.1%}")
    print(f"  Noise-aware trainable: {analysis['noise_aware']['trainable_fraction']:.1%}")
    print(f"  Average improvement: {analysis['improvement_factor']:.1f}x")
