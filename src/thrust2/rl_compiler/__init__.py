"""
RL Compiler Module for Phase 3
"""
from .environment import RLCompilerEnvironment
from .dqn_agent import DQNAgent

# Check if guaranteed_trainer exists
try:
    from .guaranteed_trainer import RLCompiler
    __all__ = ['RLCompilerEnvironment', 'DQNAgent', 'RLCompiler']
except ImportError:
    # If guaranteed_trainer doesn't exist, create a placeholder
    class RLCompiler:
        def __init__(self):
            pass
        def compile(self, circuit):
            return circuit
    __all__ = ['RLCompilerEnvironment', 'DQNAgent', 'RLCompiler']

print("✅ RL Compiler module loaded")
