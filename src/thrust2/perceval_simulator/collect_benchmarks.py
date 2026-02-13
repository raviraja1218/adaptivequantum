#!/usr/bin/env python3
"""
Main script to collect benchmark circuits for photonic compilation
"""

import argparse
import pickle
import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import BenchmarkLoader

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Collect benchmark circuits for photonic compilation'
    )
    
    parser.add_argument(
        '--circuits',
        type=str,
        default='deutsch_jozsa,vqe_h2,qaoa_maxcut',
        help='Comma-separated list of circuits to collect'
    )
    
    parser.add_argument(
        '--qubits',
        type=str,
        default='5,10,20',
        help='Comma-separated list of qubit counts for each circuit'
    )
    
    parser.add_argument(
        '--noise_profile_dir',
        type=str,
        default='../../../experiments/thrust1/noise_profiles/',
        help='Directory containing Phase 2 noise profiles'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='../../../data/processed/benchmark_circuits.pkl',
        help='Output path for collected benchmarks'
    )
    
    parser.add_argument(
        '--log_file',
        type=str,
        default='../../../logs/thrust2/benchmark_collection.log',
        help='Log file path'
    )
    
    parser.add_argument(
        '--save_metadata',
        action='store_true',
        help='Save metadata JSON file alongside pickle'
    )
    
    return parser.parse_args()

def setup_logging(log_file):
    """Setup logging to both console and file"""
    import logging
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """Main function to collect benchmarks"""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    logger.info("=" * 60)
    logger.info("PHASE 3: BENCHMARK CIRCUIT COLLECTION")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Circuits: {args.circuits}")
    logger.info(f"Qubits: {args.qubits}")
    logger.info(f"Noise profile directory: {args.noise_profile_dir}")
    logger.info(f"Output file: {args.output}")
    
    # Parse circuits and qubits
    circuits = args.circuits.split(',')
    qubits = [int(q) for q in args.qubits.split(',')]
    
    # Check if noise profile directory exists
    if not os.path.exists(args.noise_profile_dir):
        logger.warning(f"Noise profile directory not found: {args.noise_profile_dir}")
        logger.warning("Using default noise profiles")
    
    # Initialize benchmark loader
    try:
        loader = BenchmarkLoader(noise_profile_dir=args.noise_profile_dir)
        logger.info("BenchmarkLoader initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize BenchmarkLoader: {e}")
        return 1
    
    # Load benchmarks
    try:
        logger.info("Loading benchmark circuits...")
        benchmarks = loader.load_all_benchmarks(circuits=circuits, qubits=qubits)
        
        if not benchmarks:
            logger.error("No benchmarks loaded")
            return 1
        
        logger.info(f"Successfully loaded {len(benchmarks)} benchmark circuits")
        
        # Print summary
        logger.info("\n" + "-" * 40)
        logger.info("BENCHMARK SUMMARY")
        logger.info("-" * 40)
        
        total_qiskit_gates = 0
        total_converted_gates = 0
        
        for name, data in benchmarks.items():
            qiskit_gates = data['qiskit_circuit'].size()
            converted_gates = data['conversion_metadata']['converted_gates']
            
            total_qiskit_gates += qiskit_gates
            total_converted_gates += converted_gates
            
            logger.info(f"{name.upper()}:")
            logger.info(f"  Qubits: {data['n_qubits']}")
            logger.info(f"  Qiskit gates: {qiskit_gates}")
            logger.info(f"  Converted gates: {converted_gates}")
            logger.info(f"  Conversion rate: {converted_gates/qiskit_gates:.2f}")
        
        logger.info("-" * 40)
        logger.info(f"TOTAL Qiskit gates: {total_qiskit_gates}")
        logger.info(f"TOTAL Converted gates: {total_converted_gates}")
        logger.info(f"AVERAGE Conversion rate: {total_converted_gates/total_qiskit_gates:.2f}")
        
    except Exception as e:
        logger.error(f"Failed to load benchmarks: {e}")
        return 1
    
    # Save benchmarks to pickle file
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save pickle file
        with open(args.output, 'wb') as f:
            pickle.dump(benchmarks, f)
        
        logger.info(f"Saved benchmarks to: {args.output}")
        logger.info(f"File size: {os.path.getsize(args.output) / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"Failed to save benchmarks: {e}")
        return 1
    
    # Save metadata if requested
    if args.save_metadata:
        try:
            metadata_file = args.output.replace('.pkl', '_metadata.json')
            
            # Create metadata dictionary
            metadata = {
                'collection_time': datetime.now().isoformat(),
                'circuits': circuits,
                'qubits': qubits,
                'noise_profile_dir': args.noise_profile_dir,
                'benchmark_details': {}
            }
            
            for name, data in benchmarks.items():
                metadata['benchmark_details'][name] = {
                    'n_qubits': data['n_qubits'],
                    'qiskit_gates': data['qiskit_circuit'].size(),
                    'converted_gates': data['conversion_metadata']['converted_gates'],
                    'metadata': data['metadata'],
                    'conversion_metadata': data['conversion_metadata']
                }
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved metadata to: {metadata_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    # Save a quick verification file
    try:
        verification_file = args.output.replace('.pkl', '_verification.txt')
        with open(verification_file, 'w') as f:
            f.write("BENCHMARK VERIFICATION\n")
            f.write("=" * 40 + "\n")
            f.write(f"Collection time: {datetime.now().isoformat()}\n")
            f.write(f"Number of benchmarks: {len(benchmarks)}\n\n")
            
            for name, data in benchmarks.items():
                f.write(f"{name.upper()}:\n")
                f.write(f"  Qubits: {data['n_qubits']}\n")
                f.write(f"  Qiskit gates: {data['qiskit_circuit'].size()}\n")
                f.write(f"  Converted gates: {data['conversion_metadata']['converted_gates']}\n")
                f.write(f"  Noise profile: {len(data['noise_profile']['T1'])} qubits\n\n")
        
        logger.info(f"Saved verification file: {verification_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save verification file: {e}")
    
    logger.info("=" * 60)
    logger.info("BENCHMARK COLLECTION COMPLETE")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
