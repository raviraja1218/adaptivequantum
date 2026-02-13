"""
Simplified trainer for quick demonstration.
"""
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from src.thrust2.utils.benchmark_generator import load_benchmark
from src.thrust2.rl_compiler.environment import CircuitCompilationEnv
from src.thrust2.rl_compiler.dqn_agent import DQNAgent

class SimpleRLTrainer:
    """Simplified RL trainer for demonstration."""
    
    def __init__(self):
        self.setup_directories()
        
        # Load a single benchmark for simplicity
        self.circuit = load_benchmark("deutsch_jozsa_5q")
        self.env = CircuitCompilationEnv(self.circuit)
        
        # Initialize agent
        self.agent = DQNAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            device='cpu'  # Use CPU for simplicity
        )
        
        print(f"Simple trainer initialized")
        print(f"Circuit: Deutsch-Jozsa 5q, {self.circuit.component_count} gates")
        print(f"State dim: {self.env.state_dim}, Action dim: {self.env.action_dim}")
    
    def setup_directories(self):
        """Create directories for results."""
        Path("experiments/thrust2/simple_training").mkdir(parents=True, exist_ok=True)
    
    def train(self, episodes=100):
        """Simple training loop."""
        print(f"\nStarting simple training for {episodes} episodes...")
        
        results = []
        
        for episode in range(1, episodes + 1):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.memory.push(state, action, reward, next_state, done)
                
                # Train
                self.agent.train_step()
                
                # Update
                state = next_state
                episode_reward += reward
            
            # Decay epsilon
            self.agent.update_epsilon()
            
            # Update target network occasionally
            if episode % 10 == 0:
                self.agent.update_target_network()
            
            # Record results
            metrics = self.env.get_metrics()
            results.append({
                'episode': episode,
                'reward': episode_reward,
                'gates': metrics['final_gates'],
                'reduction': metrics['gate_reduction_percent'],
                'epsilon': self.agent.epsilon,
                'memory': len(self.agent.memory)
            })
            
            # Print progress
            if episode % 10 == 0:
                avg_reduction = np.mean([r['reduction'] for r in results[-10:]])
                avg_reward = np.mean([r['reward'] for r in results[-10:]])
                print(f"Episode {episode:3d} | "
                      f"Reduction: {avg_reduction:6.2f}% | "
                      f"Reward: {avg_reward:7.2f} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"Mem: {len(self.agent.memory):4d}")
        
        # Final evaluation
        print("\nFinal evaluation...")
        final_metrics = self.evaluate()
        
        # Save results
        self.save_results(results, final_metrics)
        
        return results, final_metrics
    
    def evaluate(self):
        """Evaluate trained agent."""
        state = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Use trained policy (no exploration)
            action = self.agent.select_action(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            state = next_state
            total_reward += reward
        
        metrics = self.env.get_metrics()
        metrics['total_reward'] = total_reward
        
        print(f"Evaluation results:")
        print(f"  Gates: {metrics['original_gates']} → {metrics['final_gates']}")
        print(f"  Reduction: {metrics['gate_reduction_percent']:.1f}%")
        print(f"  Photon improvement: {metrics['photon_improvement_percent']:.1f}%")
        print(f"  Total reward: {total_reward:.2f}")
        
        return metrics
    
    def save_results(self, results, final_metrics):
        """Save training results."""
        # Save results as JSON
        output = {
            'training_results': results,
            'final_evaluation': final_metrics,
            'training_config': {
                'episodes': len(results),
                'final_epsilon': self.agent.epsilon,
                'memory_size': len(self.agent.memory)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        output_path = Path("experiments/thrust2/simple_training/results.json")
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        # Save model
        model_path = Path("models/saved/rl_compiler_simple.pt")
        self.agent.save_model(model_path)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Model saved to: {model_path}")
        
        # Create summary
        self.create_summary(results, final_metrics)
    
    def create_summary(self, results, final_metrics):
        """Create summary markdown."""
        summary_path = Path("experiments/thrust2/simple_training/summary.md")
        
        with open(summary_path, 'w') as f:
            f.write("# Simple RL Training Summary\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Final Results\n\n")
            f.write(f"- **Circuit**: Deutsch-Jozsa 5q\n")
            f.write(f"- **Original gates**: {final_metrics['original_gates']}\n")
            f.write(f"- **Final gates**: {final_metrics['final_gates']}\n")
            f.write(f"- **Gate reduction**: {final_metrics['gate_reduction_percent']:.1f}%\n")
            f.write(f"- **Photon improvement**: {final_metrics['photon_improvement_percent']:.1f}%\n")
            f.write(f"- **Total reward**: {final_metrics['total_reward']:.2f}\n\n")
            
            f.write("## Training Statistics\n\n")
            
            # Calculate averages
            reductions = [r['reduction'] for r in results]
            rewards = [r['reward'] for r in results]
            
            f.write(f"- **Average gate reduction**: {np.mean(reductions):.1f}%\n")
            f.write(f"- **Best gate reduction**: {np.max(reductions):.1f}%\n")
            f.write(f"- **Average reward**: {np.mean(rewards):.2f}\n")
            f.write(f"- **Final epsilon**: {self.agent.epsilon:.3f}\n")
            f.write(f"- **Memory size**: {len(self.agent.memory)}\n\n")
            
            f.write("## Learning Progress\n\n")
            f.write("| Episode Range | Avg Reduction | Avg Reward |\n")
            f.write("|---------------|---------------|------------|\n")
            
            # Show progress in chunks
            chunk_size = len(results) // 4
            for i in range(4):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < 3 else len(results)
                
                if start < len(results):
                    chunk = results[start:end]
                    avg_red = np.mean([r['reduction'] for r in chunk])
                    avg_rew = np.mean([r['reward'] for r in chunk])
                    
                    f.write(f"| {start+1}-{end} | {avg_red:.1f}% | {avg_rew:.2f} |\n")
        
        print(f"Summary saved to: {summary_path}")

def main():
    """Main function."""
    print("="*60)
    print("SIMPLE RL TRAINING FOR PHOTONIC COMPILER")
    print("="*60)
    
    # Initialize trainer
    trainer = SimpleRLTrainer()
    
    # Train for 50 episodes (quick demonstration)
    try:
        results, final_metrics = trainer.train(episodes=50)
        
        # Check results
        final_reduction = final_metrics['gate_reduction_percent']
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Final gate reduction: {final_reduction:.1f}%")
        print(f"Paper target: 12-25%")
        
        if final_reduction >= 10:
            print("✅ Good progress - RL is learning optimization patterns!")
        elif final_reduction >= 5:
            print("⚠️  Moderate progress - Consider longer training")
        else:
            print("❌ Needs improvement - Check reward function")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
