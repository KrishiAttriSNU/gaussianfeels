#!/usr/bin/env python3
"""
GaussianFeels Main Entry Point

This is the main entry point for the GaussianFeels pipeline.
Called from the unified CLI script at scripts/gf.
"""

import sys
sys.setrecursionlimit(5000)
import argparse
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Use proper package imports (require pip install -e .)

console = Console()

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="GaussianFeels: Visuo-Tactile Gaussian Splatting Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["feelsight", "feelsight_real", "feelsight_occlusion"],
                       help="Dataset to use")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["slam", "pose", "map"],
                       help="Operation mode")
    parser.add_argument("--modality", type=str, required=True,
                       choices=["vitac", "vi", "tac"],
                       help="Sensor modality")
    parser.add_argument("--object", type=str, required=True,
                       help="Object name (e.g., contactdb_rubber_duck)")
    parser.add_argument("--log", type=str, required=True,
                       help="Log identifier (e.g., 00)")
    
    # Training parameters
    parser.add_argument("--fps", type=int, required=True,
                       help="Optimization steps per second")
    parser.add_argument("--max-steps", type=int, default=5000,
                       help="Maximum optimization steps")
    parser.add_argument("--lr-position", type=float, default=2e-4,
                       help="Learning rate for Gaussian positions")
    parser.add_argument("--lr-rotation", type=float, default=1e-3,
                       help="Learning rate for Gaussian rotations")
    parser.add_argument("--lr-scale", type=float, default=5e-3,
                       help="Learning rate for Gaussian scales")
    parser.add_argument("--lr-opacity", type=float, default=5e-2,
                       help="Learning rate for Gaussian opacities")
    
    # Gaussian parameters
    parser.add_argument("--max-gaussians", type=int, default=300000,
                       help="Maximum number of Gaussians")
    parser.add_argument("--densify-threshold", type=float, default=0.0002,
                       help="Gradient threshold for densification")
    parser.add_argument("--prune-threshold", type=float, default=0.005,
                       help="Opacity threshold for pruning")
    parser.add_argument("--densify-interval", type=int, default=100,
                       help="Steps between densification")
    
    # Advanced loss functions  
    parser.add_argument("--enable-volumetric-loss", action="store_true",
                       help="Enable advanced volumetric loss functions")
    parser.add_argument("--tactile-weight", type=float, default=1.0,
                       help="Weight for tactile loss component")
    parser.add_argument("--surface-weight", type=float, default=1.0,  
                       help="Weight for surface constraint loss")
    
    # Visualization
    parser.add_argument("--viewer", type=str, required=True,
                       choices=["open3d", "web", "none"],
                       help="Viewer type")
    parser.add_argument("--record", action="store_true",
                       help="Record session artifacts")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory")
    
    # Advanced options
    parser.add_argument("--three-camera-mode", action="store_true",
                       help="Enable three-camera setup")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    # Monitoring options
    parser.add_argument("--monitor-memory", action="store_true",
                       help="Enable memory monitoring and diagnostics")
    parser.add_argument("--monitor-performance", action="store_true",
                       help="Enable performance profiling and diagnostics")
    
    # Evaluation options
    parser.add_argument("--evaluate", action="store_true",
                       help="Run comprehensive evaluation after optimization")
    parser.add_argument("--eval-benchmarks", nargs="*", 
                       choices=["f_score", "add_s", "chamfer", "psnr", "ssim"],
                       help="Specific benchmarks to run (default: all)")
    
    return parser

def setup_environment(args: argparse.Namespace) -> Dict[str, Any]:
    """Set up the environment and validate arguments"""
    
    # Define project root
    project_root = Path(__file__).resolve().parents[1]  # Go up from gaussianfeels/ to repo root
    
    # Setup basic reproducibility (no external dependencies)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Configure PyTorch for deterministic operations
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but not available")
    
    # Validate dataset path
    dataset_path = project_root / "data" / args.dataset / args.object / args.log
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment info
    env_info = {
        "project_root": project_root,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "device": args.device,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    }
    
    return env_info

def print_banner(args: argparse.Namespace, env_info: Dict[str, Any]):
    """Print the startup banner"""
    banner_text = f"""
[bold cyan]🚀 GaussianFeels Pipeline Starting[/bold cyan]

[bold]Configuration:[/bold]
• Dataset: {args.dataset}
• Mode: {args.mode} 
• Modality: {args.modality}
• Object: {args.object}
• Log: {args.log}
• FPS: {args.fps}
• Viewer: {args.viewer}

[bold]Environment:[/bold]
• Device: {env_info['device']}
• GPUs: {env_info['gpu_count']}
• GPU Memory: {env_info['memory_gb']:.1f} GB
• Output: {env_info['output_dir']}

[bold]Gaussian Parameters:[/bold]
• Max Gaussians: {args.max_gaussians:,}
• Position LR: {args.lr_position}
• Densify Threshold: {args.densify_threshold}
"""
    
    console.print(Panel(banner_text, title="GaussianFeels", border_style="cyan"))

def main():
    """Main entry point"""
    try:
        # Parse arguments
        parser = create_parser()
        args = parser.parse_args()
        
        # Setup environment
        env_info = setup_environment(args)
        
        # Print banner
        print_banner(args, env_info)
        
        # Import core modules (delayed to avoid import errors)
        try:
            from .config import GaussianFeelsConfig, LearningRates, ViewerConfig
            from .trainer import GaussianTrainer
            from .datasets import DatasetRegistry
            from .visualization import ViewerManager
            from .evaluation import ComprehensiveEvaluator
            
            # Import monitoring tools if requested
            if args.monitor_memory:
                from shared.memory.memory_monitor import MemoryMonitor
            if args.monitor_performance:
                from shared.utils.performance_diagnostics import PerformanceDiagnostics
                
            # Import advanced loss functions if requested
            if args.enable_volumetric_loss:
                from fusion.loss.volumetric_loss import VolumetricLossFunction
        except ImportError as e:
            console.print(f"❌ Failed to import core modules: {e}", style="red")
            console.print("📝 This is expected during initial implementation", style="yellow")
            console.print("🔧 Core modules will be implemented step by step", style="cyan")
            return 1
        
        # Create configuration
        config = GaussianFeelsConfig(
            dataset=args.dataset,
            mode=args.mode,
            modality=args.modality,
            object=args.object,
            log=args.log,
            fps=args.fps,
            max_steps=args.max_steps,
            max_gaussians=args.max_gaussians,
            viewer=ViewerConfig(type=args.viewer),
            record=args.record,
            output_dir=env_info["output_dir"],
            device=args.device,
            learning_rates=LearningRates(
                position=args.lr_position,
                rotation=args.lr_rotation,
                scale=args.lr_scale,
                opacity=args.lr_opacity,
            ),
            densify_threshold=args.densify_threshold,
            prune_threshold=args.prune_threshold,
            densify_interval=args.densify_interval,
            three_camera_mode=args.three_camera_mode,
            debug=args.debug,
        )
        
        # Initialize dataset registry
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)
            dataset_registry = DatasetRegistry(env_info["project_root"] / "data")
            dataset = dataset_registry.load_dataset(config)
            progress.remove_task(task)
        
        console.print("✅ Dataset loaded successfully", style="green")
        console.print(f"   • Frames: {len(dataset)}")
        console.print(f"   • Modalities: {', '.join(dataset.modalities)}")
        
        # Initialize monitoring if requested
        memory_monitor = None
        performance_monitor = None
        
        if args.monitor_memory:
            memory_monitor = MemoryMonitor()
            memory_monitor.start_monitoring()
            console.print("🔍 Memory monitoring enabled", style="cyan")
            
        if args.monitor_performance:
            performance_monitor = PerformanceDiagnostics()
            performance_monitor.start_profiling()
            console.print("📈 Performance monitoring enabled", style="cyan")
        
        # Initialize trainer
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing trainer...", total=None)
            trainer = GaussianTrainer(config, dataset)
            progress.remove_task(task)
        
        console.print("✅ Trainer initialized", style="green")
        console.print(f"   • Gaussians: {trainer.num_gaussians:,}")
        console.print(f"   • Parameters: {trainer.num_parameters:,}")
        
        # Initialize viewer
        if args.viewer != "none":
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Starting viewer...", total=None)
                viewer = ViewerManager(config, trainer)
                viewer.start()
                progress.remove_task(task)
            
            console.print(f"✅ {args.viewer.title()} viewer started", style="green")
        
        # Main training loop
        console.print("\n🎯 Starting optimization...", style="bold cyan")
        
        try:
            for step in range(args.max_steps):
                start_time = time.time()
                
                # Pose optimization step
                if config.mode in ["slam", "pose"]:
                    pose_loss = trainer.step_pose()
                
                # Map optimization step  
                if config.mode in ["slam", "map"]:
                    map_loss = trainer.step_map()
                
                # Update viewer
                if args.viewer != "none":
                    viewer.update(step)
                
                # Timing control for FPS
                step_time = time.time() - start_time
                target_time = 1.0 / args.fps
                if step_time < target_time:
                    time.sleep(target_time - step_time)
                
                # Progress logging with comprehensive metrics
                if step % 10 == 0:
                    metrics = trainer.get_performance_metrics()
                    
                    # Build progress message with key metrics
                    progress_msg = (f"Step {step:4d} | "
                                  f"Gaussians: {metrics['num_gaussians']:6,} | "
                                  f"FPS: {metrics['recent_fps']:5.1f} | "
                                  f"Map Loss: {metrics['recent_map_loss']:8.6f} | "
                                  f"GPU: {metrics['gpu_utilization']:4.1f}% | "
                                  f"Frame: {metrics['current_frame']}/{len(dataset)} | "
                                  f"Time: {step_time:.3f}s")
                    
                    console.print(progress_msg, style="cyan")
                    
                    # Show loss trends occasionally
                    if step % 100 == 0 and step > 0:
                        trend_msg = (f"  📊 Trends | "
                                   f"Pose: {metrics['pose_trend']:+6.2%} | "
                                   f"Map: {metrics['map_trend']:+6.2%} | "
                                   f"Memory: {metrics['memory_usage_mb']:5.1f}MB | "
                                   f"Efficiency: {metrics['optimization_efficiency']:6.0f} G/s")
                        console.print(trend_msg, style="dim cyan")
                        
                        # Show modality weights
                        modality_msg = f"  🔗 Modalities: {', '.join(metrics['active_modalities'])}"
                        console.print(modality_msg, style="dim green")
                
        except KeyboardInterrupt:
            console.print("\n⏹️  Training interrupted by user", style="yellow")
        
        # Save final results
        console.print("\n💾 Saving results...", style="cyan")
        trainer.save_checkpoint(env_info["output_dir"] / "final_checkpoint.pth")
        
        if args.record:
            trainer.save_artifacts(env_info["output_dir"])
        
        # Save monitoring reports if enabled
        if memory_monitor:
            memory_monitor.stop_monitoring()
            memory_monitor.generate_report(env_info["output_dir"] / "memory_report.json")
            console.print("💾 Memory monitoring report saved", style="dim cyan")
            
        if performance_monitor:
            performance_monitor.stop_profiling() 
            performance_monitor.generate_report(env_info["output_dir"] / "performance_report.json")
            console.print("💾 Performance diagnostics report saved", style="dim cyan")
        
        # Run evaluation if requested
        if args.evaluate:
            console.print("\n📊 Running comprehensive evaluation...", style="bold magenta")
            evaluator = ComprehensiveEvaluator(config, trainer, dataset)
            
            # Configure benchmarks to run
            benchmarks = args.eval_benchmarks if args.eval_benchmarks else ["f_score", "add_s", "chamfer", "psnr", "ssim"]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running evaluation benchmarks...", total=None)
                
                eval_results = evaluator.run_comprehensive_evaluation(
                    benchmarks=benchmarks,
                    output_dir=env_info["output_dir"] / "evaluation"
                )
                
                progress.remove_task(task)
            
            # Display evaluation summary
            console.print("✅ Evaluation completed!", style="bold green")
            console.print(f"   • F-score: {eval_results.get('f_score', 'N/A'):.4f}")
            console.print(f"   • ADD-S: {eval_results.get('add_s', 'N/A'):.4f}")
            console.print(f"   • Chamfer: {eval_results.get('chamfer', 'N/A'):.6f}")
            console.print(f"   • PSNR: {eval_results.get('psnr', 'N/A'):.2f} dB")
            console.print(f"   • SSIM: {eval_results.get('ssim', 'N/A'):.4f}")
            console.print(f"📁 Detailed results saved to: {env_info['output_dir']}/evaluation/")
        
        console.print("✅ Pipeline completed successfully!", style="bold green")
        return 0
        
    except (RuntimeError, ValueError, FileNotFoundError, ImportError) as e:
        console.print(f"\n❌ Error: {e}", style="red")
        if args.debug:
            console.print(traceback.format_exc(), style="red")
        return 1

if __name__ == "__main__":
    sys.exit(main())