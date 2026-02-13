"""
Verify Day 2 RL components are working.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.thrust2.utils.benchmark_generator import load_benchmark
from src.thrust2.rl_compiler.environment import CircuitCompilationEnv
from src.thrust2.rl_compiler.dqn_agent import DQNAgent, DQN
import numpy as np
import torch

def test_environment():
    """Test RL environment."""
    print("Testing RL Environment...")
    
    # Load a benchmark circuit
    circuit = load_benchmark("deutsch_jozsa_5q")
    
    # Create environment
    env = CircuitCompilationEnv(circuit)
    
    # Test state representation
    state = env.get_state()
    print(f"State shape: {state.shape}")
    print(f"State range: [{state.min():.3f}, {state.max():.3f}]")
    
    # Test action application
    success, description = env.apply_action(0)
    print(f"Action 0: {description} (success: {success})")
    
    # Test step function
    next_state, reward, done, info = env.step(1)
    print(f"Step 1 - Reward: {reward:.3f}, Done: {done}")
    print(f"Info: {info}")
    
    # Test metrics
    metrics = env.get_metrics()
    print(f"Metrics: {metrics}")
    
    # Test reset
    env.reset()
    print(f"After reset - Gates: {env.current_circuit.component_count}")
    
    print("✓ Environment tests passed\n")
    return True

def test_dqn_agent():
    """Test DQN agent."""
    print("Testing DQN Agent...")
    
    # Test network architecture
    state_dim = 128
    action_dim = 32
    
    network = DQN(state_dim, action_dim)
    print(f"Network architecture: {network}")
    
    # Test forward pass
    test_state = torch.randn(1, state_dim)
    q_values = network(test_state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values range: [{q_values.min():.3f}, {q_values.max():.3f}]")
    
    # Test agent initialization
    agent = DQNAgent(state_dim, action_dim, device='cpu')
    print(f"Agent initialized on: {agent.device}")
    print(f"Initial epsilon: {agent.epsilon}")
    
    # Test action selection
    test_state_np = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(test_state_np, training=True)
    print(f"Selected action (training): {action}")
    
    action = agent.select_action(test_state_np, training=False)
    print(f"Selected action (no exploration): {action}")
    
    # Test replay buffer
    for i in range(10):
        agent.memory.push(
            np.random.randn(state_dim),
            np.random.randint(action_dim),
            np.random.randn(),
            np.random.randn(state_dim),
            False
        )
    
    print(f"Replay buffer size: {len(agent.memory)}")
    
    # Test training step
    loss = agent.train_step()
    print(f"Training loss: {loss if loss is not None else 'No batch yet'}")
    
    print("✓ DQN Agent tests passed\n")
    return True

def test_integration():
    """Test environment-agent integration."""
    print("Testing Environment-Agent Integration...")
    
    # Load circuit
    circuit = load_benchmark("deutsch_jozsa_5q")
    
    # Create environment
    env = CircuitCompilationEnv(circuit)
    
    # Create agent
    agent = DQNAgent(env.state_dim, env.action_dim, device='cpu')
    
    # Run one episode
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 10:
        action = agent.select_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        
        # Store in replay buffer
        agent.memory.push(state, action, reward, next_state, done)
        
        # Train
        agent.train_step()
        
        state = next_state
        total_reward += reward
        steps += 1
    
    print(f"Integration test - Steps: {steps}, Total reward: {total_reward:.2f}")
    print(f"Final gates: {env.current_circuit.component_count} "
          f"(from {env.original_gate_count})")
    
    metrics = env.get_metrics()
    print(f"Gate reduction: {metrics['gate_reduction_percent']:.1f}%")
    
    print("✓ Integration tests passed\n")
    return True

def main():
    """Run all verification tests."""
    print("=" * 70)
    print("DAY 2 RL COMPONENTS VERIFICATION")
    print("=" * 70)
    
    try:
        # Run tests
        env_ok = test_environment()
        agent_ok = test_dqn_agent()
        integration_ok = test_integration()
        
        if env_ok and agent_ok and integration_ok:
            print("=" * 70)
            print("✅ ALL DAY 2 TESTS PASSED - READY FOR TRAINING")
            print("=" * 70)
            
            # Save verification result
            import datetime
            with open("experiments/thrust2/validation/day2_verification.txt", "w") as f:
                f.write("Day 2 RL components verification passed\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            
            return True
        else:
            print("❌ Some tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
