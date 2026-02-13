"""
Start RL training for photonic compiler.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.thrust2.rl_compiler.trainer import RLCompilerTrainer

def main():
    """Start training with reduced episodes for testing."""
    print("Starting RL Training (Test Mode - 50 episodes)")
    
    # Create trainer with reduced episodes for testing
    trainer = RLCompilerTrainer({
        'total_episodes': 50,  # Reduced for testing
        'eval_frequency': 10,
        'checkpoint_frequency': 25,
        'max_steps_per_episode': 30,
        'circuit_rotation_frequency': 5
    })
    
    try:
        final_metrics = trainer.train()
        
        # Quick evaluation
        avg_reduction = final_metrics['avg_gate_reduction']
        print(f"\nTest training complete!")
        print(f"Average gate reduction: {avg_reduction:.1f}%")
        
        if avg_reduction > 5.0:
            print("✅ Test successful - RL is learning!")
            print("\nNext: Run full training with 500 episodes")
            return True
        else:
            print("⚠️  RL learning is slow - may need tuning")
            return False
            
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
