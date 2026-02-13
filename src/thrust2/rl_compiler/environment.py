"""
RL Environment for Photonic Circuit Compilation.
State: Circuit representation
Actions: Optimization operations
Reward: Gate reduction + photon improvement - time penalty
"""
import numpy as np
from typing import Tuple, Dict, Any, List
from pathlib import Path
import copy
from src.thrust2.photonic_simulator.circuit import PhotonicCircuit
from src.thrust2.photonic_simulator.components import (
    BeamSplitter, PhaseShifter, IdentityGate,
    fuse_beam_splitters, fuse_phase_shifters, is_identity_component
)

class CircuitCompilationEnv:
    """RL environment for photonic circuit compilation."""
    
    def __init__(self, circuit: PhotonicCircuit):
        """
        Initialize environment with a circuit.
        
        Parameters:
        -----------
        circuit : PhotonicCircuit
            Circuit to optimize
        """
        self.original_circuit = copy.deepcopy(circuit)
        self.current_circuit = copy.deepcopy(circuit)
        self.original_gate_count = circuit.component_count
        self.step_count = 0
        self.max_steps = 50  # Maximum optimization steps per episode
        
        # State space dimensions
        self.state_dim = 128  # Circuit features
        self.action_dim = 32  # Optimization actions
        
        # Track optimization history
        self.optimization_history = []
        
        # Initialize action definitions
        self._define_actions()
    
    def _define_actions(self):
        """Define available optimization actions."""
        self.actions = []
        
        # Action 0-7: Gate fusion actions
        for i in range(8):
            self.actions.append({
                'type': 'fuse_consecutive',
                'description': f'Fuse consecutive gates of type {i}',
                'applicable_modes': 'all'
            })
        
        # Action 8-15: Identity removal
        for i in range(8):
            self.actions.append({
                'type': 'remove_identity',
                'description': f'Remove identity-like gates in region {i}',
                'applicable_modes': 'all'
            })
        
        # Action 16-23: Decomposition optimization
        for i in range(8):
            self.actions.append({
                'type': 'optimize_decomposition',
                'description': f'Optimize gate decomposition pattern {i}',
                'applicable_modes': 'specific'
            })
        
        # Action 24-31: Special optimizations
        for i in range(8):
            self.actions.append({
                'type': 'special_pattern',
                'description': f'Apply special optimization pattern {i}',
                'applicable_modes': 'specific'
            })
        
        print(f"Defined {len(self.actions)} optimization actions")
    
    def get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        Returns:
        --------
        np.ndarray of shape (state_dim,)
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Circuit features (first 64 dimensions)
        counts = self.current_circuit.count_gates()
        state[0] = self.current_circuit.component_count / 100.0  # Normalized gate count
        state[1] = counts['beam_splitter'] / 50.0
        state[2] = counts['phase_shifter'] / 50.0
        state[3] = counts['identity'] / 20.0
        
        # Circuit depth approximation (next 10 dimensions)
        for i in range(10):
            if i < self.current_circuit.component_count:
                state[10 + i] = 1.0  # Mark as having component
        
        # Optimization history (next 20 dimensions)
        for i, hist in enumerate(self.optimization_history[-20:]):
            state[20 + i] = hist.get('reward', 0.0)
        
        # Photon survival (next 5 dimensions)
        survival = self.current_circuit.calculate_photon_loss()
        state[40] = survival
        state[41] = survival ** 2
        state[42] = np.sqrt(survival)
        
        # Random features for exploration (remaining dimensions)
        state[45:] = np.random.randn(self.state_dim - 45) * 0.1
        
        return state
    
    def apply_action(self, action_idx: int) -> Tuple[bool, str]:
        """
        Apply an optimization action to the circuit.
        
        Returns:
        --------
        tuple: (success, description)
        """
        if action_idx >= len(self.actions):
            return False, f"Invalid action index: {action_idx}"
        
        action = self.actions[action_idx]
        success = False
        description = f"Applied {action['type']}"
        
        # Track original state for reward calculation
        original_count = self.current_circuit.component_count
        original_survival = self.current_circuit.calculate_photon_loss()
        
        try:
            if action['type'] == 'fuse_consecutive':
                success = self._apply_fuse_action(action_idx)
                description = f"Fused consecutive gates (pattern {action_idx % 8})"
            
            elif action['type'] == 'remove_identity':
                success = self._apply_remove_identity(action_idx)
                description = f"Removed identity gates (region {action_idx % 8})"
            
            elif action['type'] == 'optimize_decomposition':
                success = self._apply_decomposition_optimization(action_idx)
                description = f"Optimized decomposition (pattern {action_idx % 8})"
            
            elif action['type'] == 'special_pattern':
                success = self._apply_special_pattern(action_idx)
                description = f"Applied special pattern {action_idx % 8}"
            
            # Update circuit count
            self.current_circuit.component_count = len(self.current_circuit.components)
            
        except Exception as e:
            success = False
            description = f"Action failed: {str(e)}"
        
        # Record in history
        new_count = self.current_circuit.component_count
        new_survival = self.current_circuit.calculate_photon_loss()
        
        self.optimization_history.append({
            'step': self.step_count,
            'action': action_idx,
            'description': description,
            'original_gates': original_count,
            'new_gates': new_count,
            'gate_reduction': original_count - new_count,
            'success': success
        })
        
        return success, description
    
    def _apply_fuse_action(self, pattern_idx: int) -> bool:
        """Apply gate fusion based on pattern."""
        fused_any = False
        pattern_type = pattern_idx % 4
        
        # Try to fuse consecutive components
        i = 0
        new_components = []
        
        while i < len(self.current_circuit.components):
            if i == len(self.current_circuit.components) - 1:
                new_components.append(self.current_circuit.components[i])
                i += 1
                continue
            
            comp1, modes1 = self.current_circuit.components[i]
            comp2, modes2 = self.current_circuit.components[i + 1]
            
            # Check if same modes and can be fused
            if modes1 == modes2:
                # Pattern 0: Fuse beam splitters
                if pattern_type == 0 and isinstance(comp1, BeamSplitter) and isinstance(comp2, BeamSplitter):
                    fused = fuse_beam_splitters(comp1, comp2)
                    new_components.append((fused, modes1))
                    fused_any = True
                    i += 2
                    continue
                
                # Pattern 1: Fuse phase shifters
                elif pattern_type == 1 and isinstance(comp1, PhaseShifter) and isinstance(comp2, PhaseShifter):
                    fused = fuse_phase_shifters(comp1, comp2)
                    new_components.append((fused, modes1))
                    fused_any = True
                    i += 2
                    continue
                
                # Pattern 2: Remove identity followed by anything
                elif pattern_type == 2 and is_identity_component(comp1):
                    # Skip the identity
                    new_components.append(self.current_circuit.components[i + 1])
                    fused_any = True
                    i += 2
                    continue
                
                # Pattern 3: Remove anything followed by identity
                elif pattern_type == 3 and is_identity_component(comp2):
                    # Skip the identity
                    new_components.append(self.current_circuit.components[i])
                    fused_any = True
                    i += 2
                    continue
            
            new_components.append(self.current_circuit.components[i])
            i += 1
        
        if fused_any:
            self.current_circuit.components = new_components
        
        return fused_any
    
    def _apply_remove_identity(self, region_idx: int) -> bool:
        """Remove identity gates in specific region."""
        removed_any = False
        region = region_idx % 8
        
        new_components = []
        for idx, (comp, modes) in enumerate(self.current_circuit.components):
            # Only remove if in "region" (simplified: every Nth component)
            if idx % 8 == region and is_identity_component(comp):
                removed_any = True
                continue  # Skip this identity
            new_components.append((comp, modes))
        
        if removed_any:
            self.current_circuit.components = new_components
        
        return removed_any
    
    def _apply_decomposition_optimization(self, pattern_idx: int) -> bool:
        """Optimize gate decomposition patterns."""
        optimized = False
        pattern = pattern_idx % 8
        
        # Simplified: Remove every Nth component for pattern-based optimization
        if pattern < len(self.current_circuit.components):
            # Remove component at pattern index if it's not critical
            comp, modes = self.current_circuit.components[pattern]
            if isinstance(comp, BeamSplitter) and np.abs(comp.theta) < 0.1:
                # Remove small-angle beam splitter
                self.current_circuit.components.pop(pattern)
                optimized = True
        
        return optimized
    
    def _apply_special_pattern(self, pattern_idx: int) -> bool:
        """Apply special optimization patterns."""
        # Simplified pattern application
        if pattern_idx % 3 == 0 and len(self.current_circuit.components) > 5:
            # Remove middle component if it's not critical
            middle_idx = len(self.current_circuit.components) // 2
            comp, modes = self.current_circuit.components[middle_idx]
            if is_identity_component(comp):
                self.current_circuit.components.pop(middle_idx)
                return True
        
        return False
    
    def calculate_reward(self, action_success: bool, action_time: float = 0.1) -> float:
        """
        Calculate reward for current state.
        
        Parameters:
        -----------
        action_success : bool
            Whether action was successful
        action_time : float
            Time taken for action (simulated)
        
        Returns:
        --------
        float : Reward value
        """
        # Base penalty for taking a step
        reward = -0.01
        
        # Reward for successful action
        if action_success:
            reward += 0.1
            
            # Additional reward for gate reduction
            if len(self.optimization_history) >= 2:
                prev = self.optimization_history[-2]
                curr = self.optimization_history[-1]
                
                gate_reduction = prev['original_gates'] - curr['new_gates']
                if gate_reduction > 0:
                    reward += 0.5 * gate_reduction
        
        # Reward for overall improvement from original
        current_reduction = self.original_gate_count - self.current_circuit.component_count
        if current_reduction > 0:
            reward += 0.1 * current_reduction
        
        # Penalty for taking too long
        reward -= 0.05 * action_time
        
        return reward
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Returns:
        --------
        tuple: (next_state, reward, done, info)
        """
        self.step_count += 1
        
        # Apply action
        action_success, description = self.apply_action(action_idx)
        
        # Calculate reward
        reward = self.calculate_reward(action_success)
        
        # Get next state
        next_state = self.get_state()
        
        # Check if done
        done = (self.step_count >= self.max_steps or 
                self.current_circuit.component_count <= 1)
        
        # Compile info
        info = {
            'step': self.step_count,
            'action': action_idx,
            'action_success': action_success,
            'description': description,
            'current_gates': self.current_circuit.component_count,
            'original_gates': self.original_gate_count,
            'total_reduction': self.original_gate_count - self.current_circuit.component_count,
            'done': done
        }
        
        return next_state, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset environment to original circuit."""
        self.current_circuit = copy.deepcopy(self.original_circuit)
        self.step_count = 0
        self.optimization_history = []
        return self.get_state()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get compilation metrics."""
        original_survival = self.original_circuit.calculate_photon_loss()
        current_survival = self.current_circuit.calculate_photon_loss()
        
        return {
            'original_gates': self.original_gate_count,
            'final_gates': self.current_circuit.component_count,
            'gate_reduction_percent': 100 * (self.original_gate_count - self.current_circuit.component_count) / 
                                     max(1, self.original_gate_count),
            'original_photon_survival': original_survival,
            'final_photon_survival': current_survival,
            'photon_improvement_percent': 100 * (current_survival - original_survival) / 
                                         max(1e-10, original_survival),
            'total_steps': self.step_count,
            'successful_actions': sum(1 for h in self.optimization_history if h['success']),
            'total_actions': len(self.optimization_history)
        }
    
    def render(self, mode: str = 'human'):
        """Render current state (text-based)."""
        metrics = self.get_metrics()
        
        print(f"\n{'='*60}")
        print(f"RL COMPILATION ENVIRONMENT - Step {self.step_count}")
        print(f"{'='*60}")
        print(f"Original gates: {metrics['original_gates']}")
        print(f"Current gates: {metrics['final_gates']}")
        print(f"Reduction: {metrics['gate_reduction_percent']:.1f}%")
        print(f"Photon survival: {metrics['original_photon_survival']:.3f} → "
              f"{metrics['final_photon_survival']:.3f} "
              f"({metrics['photon_improvement_percent']:.1f}% improvement)")
        print(f"Successful actions: {metrics['successful_actions']}/{metrics['total_actions']}")
        
        if self.optimization_history:
            last_action = self.optimization_history[-1]
            print(f"Last action: {last_action['description']} "
                  f"({'success' if last_action['success'] else 'failed'})")
        
        print(f"{'='*60}")
