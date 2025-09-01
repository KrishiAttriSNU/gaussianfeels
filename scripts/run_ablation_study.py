#!/usr/bin/env python3
"""
Academic Ablation Study Script for GaussianFeels

Run systematic parameter ablation studies for academic publications.
This script integrates with the main evaluation suite rather than being a separate module.

Usage:
    python scripts/run_ablation_study.py --config config.yaml --output ablation_results/
"""

import sys
import argparse
from pathlib import Path
import json
import time
from typing import Dict, List, Any

# Use proper package imports (require pip install -e .)

from main.config import GaussianFeelsConfig
from main.datasets import DatasetRegistry
from main.evaluation import EvaluationSuite

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run academic ablation study")
    parser.add_argument("--dataset", default="feelsight", help="Dataset name")
    parser.add_argument("--object", default="contactdb_rubber_duck", help="Object name")
    parser.add_argument("--log", default="00", help="Log identifier")
    parser.add_argument("--output", default="ablation_results", help="Output directory")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per parameter")
    parser.add_argument("--max-steps", type=int, default=500, help="Max training steps per run")
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    print("🧪 GaussianFeels Academic Ablation Study")
    print("=" * 50)
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset registry and evaluation suite
    dataset_registry = DatasetRegistry(project_root / "data")
    evaluation_suite = EvaluationSuite(dataset_registry)
    
    # Base configuration
    base_config = GaussianFeelsConfig(
        dataset=args.dataset,
        object=args.object,
        log=args.log,
        training={"max_steps": args.max_steps},
        reproducible=True,
        seed=42
    )
    
    # Define parameter variants for ablation study
    parameter_variants = {
        # Core Gaussian parameters
        'densify_threshold': [0.0001, 0.0002, 0.0005, 0.001],
        'prune_threshold': [0.001, 0.005, 0.01, 0.02],
        'max_sh_degree': [0, 1, 2, 3],  # DC only to full SH
        
        # Learning rates
        'learning_rate_position': [1e-5, 2e-4, 1e-3, 5e-3],
        'learning_rate_scale': [1e-3, 5e-3, 1e-2, 5e-2],
        
        # Multi-modal weights  
        'vision_weight': [0.2, 0.5, 0.8, 1.0],
        'tactile_weight': [0.2, 0.8, 1.5, 2.0]
    }
    
    print(f"📊 Running ablation study:")
    print(f"   Parameters: {list(parameter_variants.keys())}")
    print(f"   Runs per parameter: {args.runs}")
    print(f"   Output directory: {output_dir}")
    
    # Run ablation study
    start_time = time.time()
    
    try:
        results = evaluation_suite.run_ablation_study(
            base_config=base_config,
            parameter_variants=parameter_variants,
            num_runs=args.runs
        )
        
        # Save detailed results
        results_file = output_dir / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        generate_ablation_report(results, output_dir)
        
        # Generate LaTeX tables (if results are good)
        generate_latex_ablation_tables(results, output_dir)
        
        total_time = time.time() - start_time
        print(f"\n✅ Ablation study completed in {total_time:.1f}s")
        print(f"📄 Results saved to: {results_file}")
        
        # Print best parameters
        print_best_parameters(results)
        
    except Exception as e:
        print(f"❌ Ablation study failed: {e}")
        return 1
    
    return 0

def generate_ablation_report(results: Dict[str, Any], output_dir: Path):
    """Generate human-readable ablation report"""
    
    report_file = output_dir / "ablation_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("GaussianFeels Ablation Study Report\n")
        f.write("=" * 50 + "\n\n")
        
        for param_name, param_data in results.items():
            f.write(f"Parameter: {param_name}\n")
            f.write("-" * 30 + "\n")
            
            param_results = param_data['parameter_results']
            best_value = param_data.get('best_value', 'Unknown')
            
            f.write(f"Best value: {best_value}\n\n")
            
            # Results table
            f.write("Value\t\tPSNR\t\tSSIM\t\tTime\n")
            f.write("-" * 50 + "\n")
            
            for param_value, metrics in param_results.items():
                if 'error' not in metrics:
                    psnr = metrics['psnr']['mean']
                    ssim = metrics['ssim']['mean'] 
                    time_val = metrics['training_time']['mean']
                    f.write(f"{param_value}\t\t{psnr:.2f}\t\t{ssim:.3f}\t\t{time_val:.1f}s\n")
                else:
                    f.write(f"{param_value}\t\tFAILED\n")
            
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"📄 Ablation report saved: {report_file}")

def generate_latex_ablation_tables(results: Dict[str, Any], output_dir: Path):
    """Generate LaTeX tables for academic papers"""
    
    latex_file = output_dir / "ablation_tables.tex"
    
    with open(latex_file, 'w') as f:
        f.write("% GaussianFeels Ablation Study Tables\n")
        f.write("% Generated automatically from ablation study results\n\n")
        
        for param_name, param_data in results.items():
            param_results = param_data['parameter_results']
            
            # Skip if no valid results
            valid_results = {k: v for k, v in param_results.items() if 'error' not in v}
            if not valid_results:
                continue
            
            f.write(f"\\begin{{table}}[t]\n")
            f.write(f"\\centering\n")
            f.write(f"\\caption{{Ablation study results for {param_name.replace('_', ' ')}. Best results in \\textbf{{bold}}.}}\n")
            f.write(f"\\label{{tab:ablation_{param_name}}}\n")
            f.write(f"\\begin{{tabular}}{{c|ccc}}\n")
            f.write(f"\\toprule\n")
            f.write(f"{param_name.replace('_', ' ').title()} & PSNR $\\uparrow$ & SSIM $\\uparrow$ & Time (s) $\\downarrow$ \\\\\n")
            f.write(f"\\midrule\n")
            
            # Find best values
            best_psnr = max(v['psnr']['mean'] for v in valid_results.values())
            best_ssim = max(v['ssim']['mean'] for v in valid_results.values())
            best_time = min(v['training_time']['mean'] for v in valid_results.values())
            
            for param_value, metrics in valid_results.items():
                psnr = metrics['psnr']['mean']
                ssim = metrics['ssim']['mean']
                time_val = metrics['training_time']['mean']
                
                # Make best results bold
                psnr_str = f"\\textbf{{{psnr:.2f}}}" if psnr == best_psnr else f"{psnr:.2f}"
                ssim_str = f"\\textbf{{{ssim:.3f}}}" if ssim == best_ssim else f"{ssim:.3f}"  
                time_str = f"\\textbf{{{time_val:.1f}}}" if time_val == best_time else f"{time_val:.1f}"
                
                f.write(f"{param_value} & {psnr_str} & {ssim_str} & {time_str} \\\\\n")
            
            f.write(f"\\bottomrule\n")
            f.write(f"\\end{{tabular}}\n")
            f.write(f"\\end{{table}}\n\n")
    
    print(f"📄 LaTeX tables saved: {latex_file}")

def print_best_parameters(results: Dict[str, Any]):
    """Print summary of best parameters found"""
    
    print("\n🏆 Best Parameters Found:")
    print("-" * 30)
    
    for param_name, param_data in results.items():
        best_value = param_data.get('best_value', 'Unknown')
        
        # Get performance of best value
        param_results = param_data['parameter_results']
        if str(best_value) in param_results and 'error' not in param_results[str(best_value)]:
            best_psnr = param_results[str(best_value)]['psnr']['mean']
            print(f"{param_name}: {best_value} (PSNR: {best_psnr:.2f})")
        else:
            print(f"{param_name}: {best_value} (no valid results)")

if __name__ == "__main__":
    sys.exit(main())