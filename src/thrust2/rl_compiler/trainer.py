"""
RL trainer for photonic circuit compilation.
"""
import numpy as np
import torch
from typing import Dict, Any, List
from pathlib import Path
import json
import time
from datetime import datetime
from src.thrust2.utils.benchmark_generator import load_benchmark
from src.thrust2.rl_compiler.environment import CircuitCompilationEnv
from src.thrust2.rl_compiler.dqn_agent import DQNAgent

class RLCompilerTrainer:
    """Trainer for RL-based photonic compiler."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        config : dict
            Training configuration
        """
        self.config = config or self.get_default_config()
        self.setup_directories()
        
        # Load benchmark circuits
        self.benchmark_circuits = self.load_benchmarks()
        
        # Training statistics
        self.training_history = []
        self.best_metrics = {}
        
        print(f"RL Compiler Trainer initialized")
        print(f"Benchmarks: {list(self.benchmark_circuits.keys())}")
        print(f"Training episodes: {self.config['total_episodes']}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'total_episodes': 500,
            'eval_frequency': 50,
            'checkpoint_frequency': 100,
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
            'circuit_rotation_frequency': 10,  # Change circuit every N episodes
            'save_best_model': True
        }
    
    def setup_directories(self):
        """Create necessary directories."""
        self.log_dir = Path("logs/thrust2/rl_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.result_dir = Path("experiments/thrust2/rl_training")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = Path("models/saved")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def load_benchmarks(self) -> Dict[str, Any]:
        """Load all benchmark circuits."""
        benchmark_names = ["deutsch_jozsa_5q", "vqe_h2_10q", "qaoa_maxcut_20q"]
        
        benchmarks = {}
        for name in benchmark_names:
            try:
                circuit = load_benchmark(name)
                benchmarks[name] = {
                    'circuit': circuit,
                    'name': name,
                    'gate_count': circuit.component_count
                }
                print(f"Loaded {name}: {circuit.component_count} gates")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
        
        return benchmarks
    
    def select_training_circuit(self, episode: int) -> str:
        """Select circuit for training (curriculum learning)."""
        circuits = list(self.benchmark_circuits.keys())
        
        # Curriculum: start with simpler circuits
        if episode < 100:
            return "deutsch_jozsa_5q"  # Smallest circuit
        elif episode < 300:
            return "vqe_h2_10q"  # Medium circuit
        else:
            # Rotate between circuits
            return circuits[episode % len(circuits)]
    
    def initialize_agent(self) -> DQNAgent:
        """Initialize DQN agent."""
        # Create environment with first circuit to get dimensions
        first_circuit = self.benchmark_circuits["deutsch_jozsa_5q"]['circuit']
        test_env = CircuitCompilationEnv(first_circuit)
        
        state_dim = test_env.state_dim
        action_dim = test_env.action_dim
        
        print(f"Initializing DQN Agent")
        print(f"  State dimension: {state_dim}")
        print(f"  Action dimension: {action_dim}")
        print(f"  Device: {self.config['device']}")
        
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.config['device']
        )
        
        return agent
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print("STARTING RL COMPILER TRAINING")
        print("="*70)
        
        # Initialize agent
        self.agent = self.initialize_agent()
        
        # Training loop
        start_time = time.time()
        
        for episode in range(1, self.config['total_episodes'] + 1):
            # Select circuit for this episode
            circuit_name = self.select_training_circuit(episode)
            circuit_data = self.benchmark_circuits[circuit_name]
            
            # Create environment
            env = CircuitCompilationEnv(circuit_data['circuit'])
            
            # Train for one episode
            episode_metrics = self.agent.train_episode(env, self.config['max_steps_per_episode'])
            
            # Add circuit info
            episode_metrics['circuit'] = circuit_name
            episode_metrics['episode'] = episode
            
            # Record training history
            self.training_history.append(episode_metrics)
            self.agent.episode_rewards.append(episode_metrics['total_reward'])
            self.agent.episode_gate_reductions.append(episode_metrics['gate_reduction_percent'])
            
            # Update target network periodically
            if episode % self.config['target_update_frequency'] == 0:
                self.agent.update_target_network()
            
            # Evaluation
            if episode % self.config['eval_frequency'] == 0:
                eval_metrics = self.evaluate_agent(episode)
                self.log_episode(episode, episode_metrics, eval_metrics)
                
                # Check if this is the best model
                if self.is_best_model(eval_metrics):
                    self.best_metrics = eval_metrics.copy()
                    self.best_metrics['episode'] = episode
                    
                    if self.config['save_best_model']:
                        self.save_best_model(episode)
            
            # Checkpoint
            if episode % self.config['checkpoint_frequency'] == 0:
                checkpoint_path = self.agent.save_checkpoint(episode, episode_metrics)
                print(f"Episode {episode}: Checkpoint saved to {checkpoint_path}")
            
            # Progress update
            if episode % 10 == 0:
                self.print_progress(episode, episode_metrics)
        
        # Training complete
        training_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total episodes: {self.config['total_episodes']}")
        print(f"Training time: {training_time:.1f} seconds")
        print(f"Final epsilon: {self.agent.epsilon:.3f}")
        
        # Final evaluation
        final_metrics = self.final_evaluation()
        
        # Save final model
        self.save_final_model()
        
        # Generate training report
        self.generate_training_report(training_time, final_metrics)
        
        return final_metrics
    
    def evaluate_agent(self, episode: int) -> Dict[str, Any]:
        """Evaluate agent on all benchmarks."""
        eval_results = {}
        
        for name, circuit_data in self.benchmark_circuits.items():
            env = CircuitCompilationEnv(circuit_data['circuit'])
            metrics = self.agent.evaluate(env, self.config['max_steps_per_episode'])
            eval_results[name] = metrics
        
        # Calculate averages
        avg_gate_reduction = np.mean([r['gate_reduction_percent'] for r in eval_results.values()])
        avg_photon_improvement = np.mean([r['photon_improvement_percent'] for r in eval_results.values()])
        avg_reward = np.mean([r['eval_total_reward'] for r in eval_results.values()])
        
        return {
            'episode': episode,
            'avg_gate_reduction': avg_gate_reduction,
            'avg_photon_improvement': avg_photon_improvement,
            'avg_reward': avg_reward,
            'per_circuit': eval_results
        }
    
    def is_best_model(self, eval_metrics: Dict[str, Any]) -> bool:
        """Check if current model is the best."""
        if not self.best_metrics:
            return True
        
        # Compare by gate reduction (primary metric)
        current_score = eval_metrics['avg_gate_reduction']
        best_score = self.best_metrics.get('avg_gate_reduction', -float('inf'))
        
        return current_score > best_score
    
    def save_best_model(self, episode: int):
        """Save the best model."""
        model_path = self.model_dir / "rl_compiler_best.pt"
        self.agent.save_model(model_path)
        
        # Save best metrics
        best_metrics_path = self.result_dir / "best_model_metrics.json"
        with open(best_metrics_path, 'w') as f:
            json.dump(self.best_metrics, f, indent=2, default=str)
        
        print(f"Episode {episode}: New best model saved with {self.best_metrics['avg_gate_reduction']:.1f}% gate reduction")
    
    def save_final_model(self):
        """Save final model."""
        model_path = self.model_dir / "rl_compiler.pt"
        self.agent.save_model(model_path)
        print(f"Final model saved to: {model_path}")
    
    def final_evaluation(self) -> Dict[str, Any]:
        """Final evaluation on all circuits."""
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        
        final_results = {}
        
        for name, circuit_data in self.benchmark_circuits.items():
            print(f"\nEvaluating on {name}:")
            
            env = CircuitCompilationEnv(circuit_data['circuit'])
            metrics = self.agent.evaluate(env, self.config['max_steps_per_episode'])
            
            final_results[name] = metrics
            
            print(f"  Gates: {metrics['original_gates']} → {metrics['final_gates']} "
                  f"({metrics['gate_reduction_percent']:.1f}% reduction)")
            print(f"  Photon survival: {metrics['original_photon_survival']:.3f} → "
                  f"{metrics['final_photon_survival']:.3f}")
            print(f"  Steps: {metrics['eval_steps']}, Reward: {metrics['eval_total_reward']:.2f}")
        
        # Calculate overall averages
        avg_metrics = {
            'avg_gate_reduction': np.mean([r['gate_reduction_percent'] for r in final_results.values()]),
            'avg_photon_improvement': np.mean([r['photon_improvement_percent'] for r in final_results.values()]),
            'avg_reward': np.mean([r['eval_total_reward'] for r in final_results.values()]),
            'per_circuit': final_results
        }
        
        print("\n" + "-"*70)
        print(f"OVERALL AVERAGE: {avg_metrics['avg_gate_reduction']:.1f}% gate reduction")
        print(f"PAPER TARGET: 12-25% gate reduction")
        print("-"*70)
        
        return avg_metrics
    
    def log_episode(self, episode: int, episode_metrics: Dict[str, Any], 
                    eval_metrics: Dict[str, Any]):
        """Log episode results."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            'training': episode_metrics,
            'evaluation': eval_metrics
        }
        
        # Append to log file
        log_file = self.log_dir / "training_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry, default=str) + '\n')
        
        # Also save summary CSV
        summary_entry = {
            'episode': episode,
            'circuit': episode_metrics.get('circuit', 'unknown'),
            'training_reward': episode_metrics.get('total_reward', 0),
            'training_gate_reduction': episode_metrics.get('gate_reduction_percent', 0),
            'eval_gate_reduction': eval_metrics.get('avg_gate_reduction', 0),
            'eval_photon_improvement': eval_metrics.get('avg_photon_improvement', 0),
            'epsilon': self.agent.epsilon,
            'memory_size': episode_metrics.get('memory_size', 0)
        }
        
        summary_file = self.result_dir / "training_summary.csv"
        if episode == self.config['eval_frequency']:
            # Write header
            with open(summary_file, 'w') as f:
                f.write(','.join(summary_entry.keys()) + '\n')
        
        with open(summary_file, 'a') as f:
            f.write(','.join(str(v) for v in summary_entry.values()) + '\n')
    
    def print_progress(self, episode: int, metrics: Dict[str, Any]):
        """Print training progress."""
        gate_red = metrics.get('gate_reduction_percent', 0)
        reward = metrics.get('total_reward', 0)
        
        print(f"Episode {episode:4d} | "
              f"Circuit: {metrics.get('circuit', 'unknown'):20s} | "
              f"Gates: {metrics.get('original_gates', 0):3d}→{metrics.get('final_gates', 0):3d} | "
              f"Reduction: {gate_red:6.1f}% | "
              f"Reward: {reward:7.2f} | "
              f"ε: {self.agent.epsilon:.3f} | "
              f"Memory: {len(self.agent.memory):5d}")
    
    def generate_training_report(self, training_time: float, final_metrics: Dict[str, Any]):
        """Generate comprehensive training report."""
        report = {
            'training_config': self.config,
            'training_stats': {
                'total_episodes': self.config['total_episodes'],
                'training_time_seconds': training_time,
                'final_epsilon': self.agent.epsilon,
                'final_memory_size': len(self.agent.memory),
                'avg_training_reward': np.mean(self.agent.episode_rewards) if self.agent.episode_rewards else 0,
                'avg_gate_reduction': np.mean(self.agent.episode_gate_reductions) if self.agent.episode_gate_reductions else 0
            },
            'final_evaluation': final_metrics,
            'best_model': self.best_metrics,
            'benchmark_info': {
                name: {'gate_count': data['gate_count']} 
                for name, data in self.benchmark_circuits.items()
            },
            'generated_at': datetime.now().isoformat()
        }
        
        report_path = self.result_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nTraining report saved to: {report_path}")
        
        # Also create a markdown summary
        self.create_markdown_summary(report, training_time)
    
    def create_markdown_summary(self, report: Dict[str, Any], training_time: float):
        """Create markdown summary of training results."""
        md_path = self.result_dir / "training_summary.md"
        
        with open(md_path, 'w') as f:
            f.write("# RL Compiler Training Summary\n\n")
            f.write(f"**Training completed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total episodes**: {report['training_stats']['total_episodes']}\n")
            f.write(f"**Training time**: {training_time:.1f} seconds\n\n")
            
            f.write("## Final Evaluation Results\n\n")
            f.write("| Circuit | Original Gates | Final Gates | Reduction | Photon Improvement |\n")
            f.write("|---------|----------------|-------------|-----------|-------------------|\n")
            
            for name, metrics in report['final_evaluation']['per_circuit'].items():
                f.write(f"| {name} | {metrics['original_gates']} | {metrics['final_gates']} | "
                       f"{metrics['gate_reduction_percent']:.1f}% | "
                       f"{metrics['photon_improvement_percent']:.1f}% |\n")
            
            f.write("\n")
            f.write(f"**Average Gate Reduction**: {report['final_evaluation']['avg_gate_reduction']:.1f}%\n")
            f.write(f"**Paper Target**: 12-25%\n")
            f.write(f"**Target Achieved**: {'✅' if report['final_evaluation']['avg_gate_reduction'] >= 12 else '❌'}\n\n")
            
            f.write("## Best Model Performance\n\n")
            if report['best_model']:
                f.write(f"**Best at Episode**: {report['best_model'].get('episode', 'N/A')}\n")
                f.write(f"**Best Gate Reduction**: {report['best_model'].get('avg_gate_reduction', 0):.1f}%\n")
            
            f.write("\n## Training Configuration\n\n")
            for key, value in report['training_config'].items():
                f.write(f"- **{key}**: {value}\n")
        
        print(f"Markdown summary saved to: {md_path}")

def main():
    """Main training function."""
    print("Starting RL Compiler Training...")
    
    # Initialize trainer
    trainer = RLCompilerTrainer()
    
    # Start training
    try:
        final_metrics = trainer.train()
        
        # Check if paper targets were met
        avg_reduction = final_metrics['avg_gate_reduction']
        target_met = avg_reduction >= 12.0
        
        print("\n" + "="*70)
        print("PAPER TARGET VERIFICATION")
        print("="*70)
        print(f"Average Gate Reduction Achieved: {avg_reduction:.1f}%")
        print(f"Paper Target Range: 12-25%")
        print(f"Target Met: {'✅ YES' if target_met else '❌ NO'}")
        print("="*70)
        
        return target_met
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
