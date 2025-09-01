"""
Occlusion plotting and visualization for gaussianfeels evaluation

This module provides comprehensive plotting capabilities for analyzing how
occlusion affects Gaussian splatting reconstruction quality and pose estimation.
Adapted from neuralfeels for gaussianfeels architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pickle
from scipy import stats
from scipy.interpolate import interp1d
import logging

from .gaussian_occlusion_analysis import (
    OcclusionAnalysisResults, ViewpointAnalysis, OcclusionMetrics,
    GaussianOcclusionAnalyzer
)
from .feelsight_init import get_available_objects

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OcclusionPlotter:
    """
    Comprehensive plotting suite for occlusion analysis in Gaussian splatting
    """
    
    def __init__(self, 
                 output_dir: Union[str, Path],
                 style: str = "seaborn-v0_8",
                 color_palette: str = "husl",
                 fig_size: Tuple[int, int] = (12, 8),
                 dpi: int = 300):
        """
        Initialize occlusion plotter.
        
        Args:
            output_dir: Directory for saving plots
            style: Matplotlib style
            color_palette: Seaborn color palette
            fig_size: Default figure size
            dpi: Plot resolution
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fig_size = fig_size
        self.dpi = dpi
        
        # Setup plotting style
        plt.style.use(style)
        sns.set_palette(color_palette)
        
        # Color scheme for different modalities
        self.colors = {
            'vision_only': '#1f77b4',
            'tactile_only': '#ff7f0e', 
            'vision_tactile': '#2ca02c',
            'gaussian_points': '#d62728',
            'confidence_high': '#9467bd',
            'confidence_low': '#8c564b'
        }
        
        # Setup interactive plotting
        self.plotly_template = "plotly_white"
    
    def load_analysis_results(self, results_path: Path) -> OcclusionAnalysisResults:
        """Load analysis results from JSON file."""
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct the dataclass (simplified - could be more robust)
        return OcclusionAnalysisResults(**data)
    
    def pose_error_vs_occlusion_scatter(self, 
                                      results: OcclusionAnalysisResults,
                                      save_path: Optional[Path] = None,
                                      interactive: bool = True) -> None:
        """
        Create scatter plot of pose error vs occlusion level.
        
        Args:
            results: Occlusion analysis results
            save_path: Path to save plot (optional)
            interactive: Whether to create interactive Plotly plot
        """
        # Prepare data
        data_points = []
        for vp_analysis in results.viewpoint_analyses:
            for metrics in vp_analysis.occlusion_metrics:
                data_points.append({
                    'occlusion_level': metrics.occlusion_level * 100,  # Convert to percentage
                    'pose_error': metrics.pose_error_translation,
                    'rotation_error': metrics.pose_error_rotation,
                    'viewpoint': vp_analysis.viewpoint,
                    'fscore': metrics.reconstruction_fscore,
                    'gaussian_count': metrics.gaussian_count,
                    'confidence': metrics.confidence_score,
                    'processing_time': metrics.processing_time
                })
        
        df = pd.DataFrame(data_points)
        
        if interactive:
            # Create interactive Plotly plot
            fig = px.scatter(
                df, 
                x='occlusion_level', 
                y='pose_error',
                color='viewpoint',
                size='confidence',
                hover_data=['rotation_error', 'fscore', 'gaussian_count', 'processing_time'],
                title=f'Pose Error vs Occlusion Level - {results.object_name}',
                labels={
                    'occlusion_level': 'Occlusion Level (%)',
                    'pose_error': 'Translation Error (m)',
                    'viewpoint': 'Camera Viewpoint'
                },
                template=self.plotly_template
            )
            
            # Add correlation line
            if len(df) > 2:
                # Fit linear regression
                z = np.polyfit(df['occlusion_level'], df['pose_error'], 1)
                p = np.poly1d(z)
                
                x_line = np.linspace(df['occlusion_level'].min(), df['occlusion_level'].max(), 100)
                y_line = p(x_line)
                
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name=f'Trend (r={results.overall_correlation:.3f})',
                    line=dict(color='red', dash='dash')
                ))
            
            # Add statistical annotation
            fig.add_annotation(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text=f'Correlation: {results.overall_correlation:.3f}<br>' +
                     f'p-value: {results.statistical_significance:.3f}<br>' +
                     f'n = {len(df)} experiments',
                showarrow=False,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
            
            fig.update_layout(
                width=800,
                height=600,
                showlegend=True
            )
            
            if save_path:
                fig.write_html(str(save_path.with_suffix('.html')))
                fig.write_image(str(save_path.with_suffix('.png')), 
                              width=800, height=600, scale=2)
            
            fig.show()
        
        else:
            # Create static matplotlib plot
            fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
            
            # Scatter plot with viewpoint colors
            viewpoints = df['viewpoint'].unique()
            for i, vp in enumerate(viewpoints):
                vp_data = df[df['viewpoint'] == vp]
                ax.scatter(
                    vp_data['occlusion_level'],
                    vp_data['pose_error'],
                    label=f'Viewpoint {vp}',
                    alpha=0.7,
                    s=60
                )
            
            # Add regression line if enough data
            if len(df) > 2:
                z = np.polyfit(df['occlusion_level'], df['pose_error'], 1)
                p = np.poly1d(z)
                
                x_line = np.linspace(df['occlusion_level'].min(), df['occlusion_level'].max(), 100)
                y_line = p(x_line)
                
                ax.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=2,
                       label=f'Trend (r={results.overall_correlation:.3f})')
            
            ax.set_xlabel('Occlusion Level (%)', fontsize=12)
            ax.set_ylabel('Translation Error (m)', fontsize=12)
            ax.set_title(f'Pose Error vs Occlusion Level - {results.object_name}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = (f'Correlation: {results.overall_correlation:.3f}\n'
                         f'p-value: {results.statistical_significance:.3f}\n'
                         f'n = {len(df)} experiments')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path.with_suffix('.png'), dpi=self.dpi, bbox_inches='tight')
                plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
            
            plt.show()
    
    def reconstruction_quality_vs_occlusion(self, 
                                          results: OcclusionAnalysisResults,
                                          save_path: Optional[Path] = None) -> None:
        """
        Plot reconstruction quality metrics vs occlusion level.
        
        Args:
            results: Occlusion analysis results
            save_path: Path to save plot
        """
        # Prepare data
        data_points = []
        for vp_analysis in results.viewpoint_analyses:
            for metrics in vp_analysis.occlusion_metrics:
                data_points.append({
                    'occlusion_level': metrics.occlusion_level * 100,
                    'fscore': metrics.reconstruction_fscore,
                    'chamfer_distance': metrics.reconstruction_chamfer,
                    'viewpoint': vp_analysis.viewpoint,
                    'gaussian_count': metrics.gaussian_count
                })
        
        df = pd.DataFrame(data_points)
        
        # Create subplot figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=self.dpi)
        
        # F-score vs Occlusion
        viewpoints = df['viewpoint'].unique()
        for i, vp in enumerate(viewpoints):
            vp_data = df[df['viewpoint'] == vp]
            ax1.scatter(vp_data['occlusion_level'], vp_data['fscore'], 
                       label=f'Viewpoint {vp}', alpha=0.7, s=60)
        
        ax1.set_xlabel('Occlusion Level (%)')
        ax1.set_ylabel('F-Score')
        ax1.set_title('Reconstruction F-Score vs Occlusion')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chamfer Distance vs Occlusion (log scale)
        for i, vp in enumerate(viewpoints):
            vp_data = df[df['viewpoint'] == vp]
            # Filter out infinite values for plotting
            finite_mask = np.isfinite(vp_data['chamfer_distance'])
            if finite_mask.any():
                ax2.scatter(vp_data[finite_mask]['occlusion_level'], 
                           vp_data[finite_mask]['chamfer_distance'], 
                           label=f'Viewpoint {vp}', alpha=0.7, s=60)
        
        ax2.set_xlabel('Occlusion Level (%)')
        ax2.set_ylabel('Chamfer Distance (log scale)')
        ax2.set_yscale('log')
        ax2.set_title('Chamfer Distance vs Occlusion')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gaussian Count vs Occlusion
        for i, vp in enumerate(viewpoints):
            vp_data = df[df['viewpoint'] == vp]
            ax3.scatter(vp_data['occlusion_level'], vp_data['gaussian_count'], 
                       label=f'Viewpoint {vp}', alpha=0.7, s=60)
        
        ax3.set_xlabel('Occlusion Level (%)')
        ax3.set_ylabel('Number of Gaussians')
        ax3.set_title('Gaussian Count vs Occlusion')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'Reconstruction Quality Analysis - {results.object_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.with_suffix('.png'), dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        
        plt.show()
    
    def viewpoint_comparison_radar(self, 
                                  results: OcclusionAnalysisResults,
                                  save_path: Optional[Path] = None) -> None:
        """
        Create radar chart comparing viewpoint performance.
        
        Args:
            results: Occlusion analysis results
            save_path: Path to save plot
        """
        if len(results.viewpoint_analyses) < 2:
            logger.warning("Need at least 2 viewpoints for radar comparison")
            return
        
        # Prepare data for radar chart
        metrics_data = []
        for vp_analysis in results.viewpoint_analyses:
            # Aggregate metrics across all experiments for this viewpoint
            all_metrics = vp_analysis.occlusion_metrics
            
            avg_pose_error = np.mean([m.pose_error_translation for m in all_metrics])
            avg_rot_error = np.mean([m.pose_error_rotation for m in all_metrics])
            avg_fscore = np.mean([m.reconstruction_fscore for m in all_metrics])
            avg_processing_time = np.mean([m.processing_time for m in all_metrics])
            avg_gaussian_count = np.mean([m.gaussian_count for m in all_metrics])
            avg_confidence = np.mean([m.confidence_score for m in all_metrics])
            
            metrics_data.append({
                'viewpoint': vp_analysis.viewpoint,
                'pose_accuracy': 1.0 / (1.0 + avg_pose_error),  # Invert so higher is better
                'rotation_accuracy': 1.0 / (1.0 + avg_rot_error / 180.0),  # Normalize and invert
                'reconstruction_quality': avg_fscore,
                'efficiency': 1.0 / (1.0 + avg_processing_time / 60.0),  # Invert time (minutes)
                'gaussian_efficiency': 1.0 / (1.0 + avg_gaussian_count / 10000.0),  # Normalize gaussian count
                'confidence': avg_confidence
            })
        
        df_radar = pd.DataFrame(metrics_data)
        
        # Create interactive radar chart with Plotly
        categories = ['Pose Accuracy', 'Rotation Accuracy', 'Reconstruction Quality', 
                     'Processing Efficiency', 'Gaussian Efficiency', 'Confidence']
        
        fig = go.Figure()
        
        for _, row in df_radar.iterrows():
            values = [
                row['pose_accuracy'],
                row['rotation_accuracy'], 
                row['reconstruction_quality'],
                row['efficiency'],
                row['gaussian_efficiency'],
                row['confidence']
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f'Viewpoint {row["viewpoint"]}',
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=f"Viewpoint Performance Comparison - {results.object_name}",
            template=self.plotly_template
        )
        
        if save_path:
            fig.write_html(str(save_path.with_suffix('.html')))
            fig.write_image(str(save_path.with_suffix('.png')), 
                          width=800, height=600, scale=2)
        
        fig.show()
    
    def occlusion_heatmap(self, 
                         results_list: List[OcclusionAnalysisResults],
                         save_path: Optional[Path] = None) -> None:
        """
        Create heatmap showing occlusion sensitivity across objects and viewpoints.
        
        Args:
            results_list: List of analysis results for different objects
            save_path: Path to save plot
        """
        if not results_list:
            logger.warning("No results provided for heatmap")
            return
        
        # Prepare data matrix
        objects = [r.object_name for r in results_list]
        all_viewpoints = set()
        for results in results_list:
            all_viewpoints.update([vp.viewpoint for vp in results.viewpoint_analyses])
        viewpoints = sorted(list(all_viewpoints))
        
        # Create correlation matrix
        correlation_matrix = np.full((len(objects), len(viewpoints)), np.nan)
        
        for i, results in enumerate(results_list):
            for vp_analysis in results.viewpoint_analyses:
                j = viewpoints.index(vp_analysis.viewpoint)
                correlation_matrix[i, j] = vp_analysis.correlation_occlusion_error
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(viewpoints) * 1.5), 
                                       max(6, len(objects) * 0.8)), dpi=self.dpi)
        
        # Mask NaN values
        mask = np.isnan(correlation_matrix)
        
        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0,
            xticklabels=viewpoints,
            yticklabels=objects,
            cbar_kws={'label': 'Occlusion-Error Correlation'},
            ax=ax
        )
        
        ax.set_title('Occlusion Sensitivity Heatmap\n(Higher values = more sensitive to occlusion)', 
                    fontsize=14, pad=20)
        ax.set_xlabel('Camera Viewpoint', fontsize=12)
        ax.set_ylabel('Object', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.with_suffix('.png'), dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        
        plt.show()
    
    def processing_time_analysis(self, 
                               results: OcclusionAnalysisResults,
                               save_path: Optional[Path] = None) -> None:
        """
        Analyze processing time vs various factors.
        
        Args:
            results: Occlusion analysis results
            save_path: Path to save plot
        """
        # Prepare data
        data_points = []
        for vp_analysis in results.viewpoint_analyses:
            for metrics in vp_analysis.occlusion_metrics:
                data_points.append({
                    'processing_time': metrics.processing_time,
                    'gaussian_count': metrics.gaussian_count,
                    'occlusion_level': metrics.occlusion_level * 100,
                    'convergence_iterations': metrics.convergence_iterations,
                    'viewpoint': vp_analysis.viewpoint
                })
        
        df = pd.DataFrame(data_points)
        
        if df.empty:
            logger.warning("No timing data available")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        
        # Processing time vs Gaussian count
        viewpoints = df['viewpoint'].unique()
        for vp in viewpoints:
            vp_data = df[df['viewpoint'] == vp]
            ax1.scatter(vp_data['gaussian_count'], vp_data['processing_time'], 
                       label=f'Viewpoint {vp}', alpha=0.7, s=60)
        
        ax1.set_xlabel('Number of Gaussians')
        ax1.set_ylabel('Processing Time (s)')
        ax1.set_title('Processing Time vs Gaussian Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Processing time vs Occlusion level
        for vp in viewpoints:
            vp_data = df[df['viewpoint'] == vp]
            ax2.scatter(vp_data['occlusion_level'], vp_data['processing_time'], 
                       label=f'Viewpoint {vp}', alpha=0.7, s=60)
        
        ax2.set_xlabel('Occlusion Level (%)')
        ax2.set_ylabel('Processing Time (s)')
        ax2.set_title('Processing Time vs Occlusion Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Processing time vs Convergence iterations
        for vp in viewpoints:
            vp_data = df[df['viewpoint'] == vp]
            ax3.scatter(vp_data['convergence_iterations'], vp_data['processing_time'], 
                       label=f'Viewpoint {vp}', alpha=0.7, s=60)
        
        ax3.set_xlabel('Convergence Iterations')
        ax3.set_ylabel('Processing Time (s)')
        ax3.set_title('Processing Time vs Convergence Iterations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Processing time distribution
        ax4.hist([df[df['viewpoint'] == vp]['processing_time'] for vp in viewpoints], 
                bins=20, alpha=0.7, label=[f'Viewpoint {vp}' for vp in viewpoints])
        ax4.set_xlabel('Processing Time (s)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Processing Time Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Processing Time Analysis - {results.object_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.with_suffix('.png'), dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        
        plt.show()
    
    def generate_summary_report(self, 
                              results_list: List[OcclusionAnalysisResults],
                              save_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive summary report of occlusion analysis.
        
        Args:
            results_list: List of analysis results for different objects
            save_path: Path to save report
            
        Returns:
            Summary report as markdown string
        """
        report_lines = [
            "# Gaussian Splatting Occlusion Analysis Report",
            "",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Objects analyzed: {len(results_list)}",
            "",
            "## Executive Summary",
            ""
        ]
        
        if not results_list:
            report_lines.extend([
                "No analysis results available.",
                ""
            ])
        else:
            # Overall statistics
            all_correlations = [r.overall_correlation for r in results_list]
            all_p_values = [r.statistical_significance for r in results_list]
            
            mean_correlation = np.mean(all_correlations)
            significant_count = sum(1 for p in all_p_values if p < 0.05)
            
            report_lines.extend([
                f"- Average occlusion-error correlation: {mean_correlation:.3f}",
                f"- Statistically significant results: {significant_count}/{len(results_list)} objects",
                f"- Total experiments analyzed: {sum(len(r.viewpoint_analyses) for r in results_list)}",
                "",
                "## Per-Object Analysis",
                ""
            ])
            
            # Per-object details
            for results in results_list:
                report_lines.extend([
                    f"### {results.object_name}",
                    "",
                    f"- **Viewpoints analyzed**: {len(results.viewpoint_analyses)}",
                    f"- **Overall correlation**: {results.overall_correlation:.3f}",
                    f"- **Statistical significance**: {results.statistical_significance:.3f}",
                    f"- **Status**: {'Significant' if results.statistical_significance < 0.05 else 'Not significant'}",
                    ""
                ])
                
                if results.viewpoint_analyses:
                    best_vp = min(results.viewpoint_analyses, key=lambda x: x.mean_pose_error)
                    worst_vp = max(results.viewpoint_analyses, key=lambda x: x.mean_pose_error)
                    
                    report_lines.extend([
                        f"- **Best viewpoint**: {best_vp.viewpoint} (error: {best_vp.mean_pose_error:.3f}m)",
                        f"- **Worst viewpoint**: {worst_vp.viewpoint} (error: {worst_vp.mean_pose_error:.3f}m)",
                        ""
                    ])
                
                if results.recommendations:
                    report_lines.extend([
                        "**Recommendations**:",
                        ""
                    ])
                    for i, rec in enumerate(results.recommendations, 1):
                        report_lines.append(f"{i}. {rec}")
                    report_lines.append("")
            
            # Global recommendations
            report_lines.extend([
                "## Global Recommendations",
                ""
            ])
            
            if mean_correlation > 0.3:
                report_lines.append("- High occlusion sensitivity detected across objects. Consider improving multi-modal fusion or tactile sensing coverage.")
            elif mean_correlation < -0.1:
                report_lines.append("- Interesting negative correlation patterns suggest the system may be more robust than expected.")
            else:
                report_lines.append("- Moderate occlusion sensitivity. Current approach appears reasonably robust.")
            
            if significant_count / len(results_list) < 0.5:
                report_lines.append("- Many non-significant results suggest need for larger sample sizes or improved data collection.")
            
            report_lines.extend([
                "",
                "## Technical Notes",
                "",
                "- Occlusion levels estimated from depth map coverage and tactile contact regions",
                "- Pose errors computed as Euclidean distance for translation and angular distance for rotation",
                "- F-scores computed using configurable distance thresholds for surface reconstruction",
                "- Statistical significance testing uses Pearson correlation with p < 0.05 threshold",
                ""
            ])
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Summary report saved to {save_path}")
        
        return report_text


def main():
    """Main function for occlusion plotting"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Occlusion Analysis Plotting")
    parser.add_argument("--analysis_results", type=str, required=True,
                       help="Path to analysis results JSON file or directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory for saving plots")
    parser.add_argument("--plot_type", type=str, default="all",
                       choices=["scatter", "quality", "radar", "heatmap", "timing", "all"],
                       help="Type of plot to generate")
    parser.add_argument("--interactive", action="store_true",
                       help="Generate interactive plots where applicable")
    
    args = parser.parse_args()
    
    # Initialize plotter
    plotter = OcclusionPlotter(output_dir=args.output_dir)
    
    # Load results
    results_path = Path(args.analysis_results)
    if results_path.is_file():
        # Single file
        results = plotter.load_analysis_results(results_path)
        results_list = [results]
    else:
        # Directory of files
        results_files = list(results_path.glob("occlusion_analysis_*.json"))
        results_list = [plotter.load_analysis_results(f) for f in results_files]
    
    if not results_list:
        print("No analysis results found!")
        return
    
    print(f"Loaded {len(results_list)} analysis results")
    
    # Generate plots
    for i, results in enumerate(results_list):
        object_name = results.object_name
        
        if args.plot_type in ["scatter", "all"]:
            print(f"Generating scatter plot for {object_name}...")
            save_path = plotter.output_dir / f"occlusion_scatter_{object_name}"
            plotter.pose_error_vs_occlusion_scatter(
                results, save_path=save_path, interactive=args.interactive
            )
        
        if args.plot_type in ["quality", "all"]:
            print(f"Generating quality analysis for {object_name}...")
            save_path = plotter.output_dir / f"reconstruction_quality_{object_name}"
            plotter.reconstruction_quality_vs_occlusion(results, save_path=save_path)
        
        if args.plot_type in ["radar", "all"]:
            print(f"Generating radar chart for {object_name}...")
            save_path = plotter.output_dir / f"viewpoint_radar_{object_name}"
            plotter.viewpoint_comparison_radar(results, save_path=save_path)
        
        if args.plot_type in ["timing", "all"]:
            print(f"Generating timing analysis for {object_name}...")
            save_path = plotter.output_dir / f"processing_timing_{object_name}"
            plotter.processing_time_analysis(results, save_path=save_path)
    
    # Multi-object plots
    if len(results_list) > 1:
        if args.plot_type in ["heatmap", "all"]:
            print("Generating cross-object heatmap...")
            save_path = plotter.output_dir / "occlusion_heatmap_all_objects"
            plotter.occlusion_heatmap(results_list, save_path=save_path)
    
    # Generate summary report
    if args.plot_type in ["all"]:
        print("Generating summary report...")
        save_path = plotter.output_dir / "occlusion_analysis_report.md"
        plotter.generate_summary_report(results_list, save_path=save_path)
    
    print(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()