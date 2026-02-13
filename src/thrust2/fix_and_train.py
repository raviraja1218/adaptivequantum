"""
Fix configuration and start training.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.thrust2.rl_compiler.trainer import RLCompilerTrainer

def fix_and_train():
    """Fix configuration and start training."""
    print("Fixing configuration and starting training...")
    
    # Create proper configuration
    config = {
        'total_episodes': 100,  # Increased for better learning
        'eval_frequency': 20,
        'checkpoint_frequency': 50,
        'max_steps_per_episode': 50,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.1,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'target_update_frequency': 10,
        'replay_buffer_capacity': 10000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'circuit_rotation_frequency': 10,
        'save_best_model': True
    }
    
    print(f"Using device: {config['device']}")
    print(f"Total episodes: {config['total_episodes']}")
    
    # Initialize trainer with fixed config
    trainer = RLCompilerTrainer(config)
    
    try:
        # Train for 100 episodes
        final_metrics = trainer.train()
        
        return final_metrics
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_test():
    """Quick test to verify everything works."""
    print("\nQuick test of RL components...")
    
    from src.thrust2.utils.benchmark_generator import load_benchmark
    from src.thrust2.rl_compiler.environment import CircuitCompilationEnv
    from src.thrust2.rl_compiler.dqn_agent import DQNAgent
    
    # Load circuit
    circuit = load_benchmark("deutsch_jozsa_5q")
    
    # Create environment
    env = CircuitCompilationEnv(circuit)
    
    # Create agent
    agent = DQNAgent(env.state_dim, env.action_dim, device='cpu')
    
    # Run a few steps
    state = env.reset()
    total_reward = 0
    
    for step in range(5):
        action = agent.select_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        
        # Store experience
        agent.memory.push(state, action, reward, next_state, done)
        
        # Train
        loss = agent.train_step()
        
        state = next_state
        total_reward += reward
        
        print(f"Step {step}: Action {action}, Reward {reward:.3f}, "
              f"Gates {info['current_gates']}")
    
    print(f"Quick test complete - Total reward: {total_reward:.3f}")
    print(f"Agent epsilon: {agent.epsilon:.3f}")
    print(f"Memory size: {len(agent.memory)}")
    
    return True

if __name__ == "__main__":
    print("="*70)
    print("FIXED RL TRAINING STARTUP")
    print("="*70)
    
    # First run quick test
    test_ok = quick_test()
    
    if test_ok:
        print("\n" + "="*70)
        print("STARTING MAIN TRAINING")
        print("="*70)
        
        results = fix_and_train()
        
        if results:
            avg_reduction = results.get('avg_gate_reduction', 0)
            print(f"\nTraining complete!")
            print(f"Average gate reduction: {avg_reduction:.1f}%")
            
            if avg_reduction >= 10.0:
                print("✅ Excellent results! Paper target (12-25%) within reach.")
            elif avg_reduction >= 5.0:
                print("⚠️  Good start - Continue training for better results.")
            else:
                print("❌ Results need improvement - Check RL parameters.")
        else:
            print("❌ Training failed")
    else:
        print("❌ Quick test failed - Check RL components")
