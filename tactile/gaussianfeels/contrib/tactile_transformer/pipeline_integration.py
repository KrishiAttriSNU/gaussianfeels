"""
Pipeline Integration for Tactile Transformer with GaussianFeels

This module provides integration between the tactile transformer training
workflow and existing gaussianfeels camera/tactile pipelines.
"""

import os
import sys
import logging
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
from PIL import Image

# Use proper package imports (require pip install -e .)

from .touch_vit import TouchVIT
from .dpt_model import DPTModel
from .utils import apply_jet_colormap

logger = logging.getLogger(__name__)


class TactileTransformerPipeline:
    """
    Integration wrapper for using trained tactile transformers in gaussianfeels pipelines
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 device: str = "cuda",
                 preprocess_tactile: bool = True):
        """
        Initialize tactile transformer pipeline
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to training configuration
            device: Device to run inference on
            preprocess_tactile: Whether to apply tactile preprocessing
        """
        from shared.utils.device_utils import get_device
        self.device = get_device(device)
        self.preprocess_tactile = preprocess_tactile
        
        # Load configuration
        from omegaconf import OmegaConf
        self.config = OmegaConf.load(config_path)
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup preprocessing transforms
        self._setup_transforms()
        
        logger.info(f"Tactile transformer pipeline initialized on {self.device}")
    
    def _load_model(self, model_path: str) -> DPTModel:
        """Load trained model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Initialize model architecture
        model = DPTModel(
            image_size=(3, self.config["Dataset"]["transforms"]["resize"][0], 
                       self.config["Dataset"]["transforms"]["resize"][1]),
            emb_dim=self.config["General"]["emb_dim"],
            resample_dim=self.config["General"]["resample_dim"],
            read=self.config["General"]["read"],
            nclasses=len(self.config["Dataset"]["classes"]),
            hooks=self.config["General"]["hooks"],
            model_timm=self.config["General"]["model_timm"],
            type=self.config["General"]["type"],
            patch_size=self.config["General"]["patch_size"],
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def _setup_transforms(self):
        """Setup preprocessing transforms"""
        from torchvision import transforms
        
        resize = self.config["Dataset"]["transforms"]["resize"]
        
        self.transform = transforms.Compose([
            transforms.Resize((resize[0], resize[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess tactile image for inference
        
        Args:
            image: Input tactile image (H, W, 3) uint8 numpy array
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        if self.preprocess_tactile:
            # Apply tactile-specific preprocessing
            image = self._apply_tactile_preprocessing(image)
        
        # Convert to PIL for transforms
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image)
        else:
            # Normalize to 0-255 if needed
            image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_norm)
        
        # Apply transforms and add batch dimension
        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0).to(self.device)
    
    def _apply_tactile_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply tactile-specific preprocessing steps"""
        # Background subtraction (basic version)
        # In a real implementation, this would use template matching
        # For now, we'll apply basic denoising and contrast enhancement
        
        # Convert to float for processing
        image_float = image.astype(np.float32) / 255.0
        
        # Apply Gaussian blur for denoising
        image_blur = cv2.GaussianBlur(image_float, (3, 3), 0.5)
        
        # Enhance contrast using CLAHE
        if len(image_blur.shape) == 3:
            # Convert to LAB, apply CLAHE to L channel
            lab = cv2.cvtColor(image_blur, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply((l_channel * 255).astype(np.uint8)) / 255.0
            lab[:, :, 0] = l_channel
            image_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_enhanced = clahe.apply((image_blur * 255).astype(np.uint8)) / 255.0
        
        # Convert back to uint8
        return (image_enhanced * 255).astype(np.uint8)
    
    def predict(self, 
               tactile_image: np.ndarray,
               return_both: bool = None) -> Dict[str, np.ndarray]:
        """
        Run inference on tactile image
        
        Args:
            tactile_image: Input tactile image (H, W, 3)
            return_both: Whether to return both depth and segmentation (None = auto from config)
            
        Returns:
            Dictionary containing 'depth' and/or 'segmentation' predictions
        """
        if return_both is None:
            return_both = self.config["General"]["type"] == "full"
        
        # Preprocess input
        input_tensor = self.preprocess_image(tactile_image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Handle different return formats
            if isinstance(outputs, tuple):
                if self.config["General"]["type"] == "full":
                    depth_pred, seg_pred = outputs
                elif self.config["General"]["type"] == "depth":
                    depth_pred = outputs[0] if outputs[0] is not None else None
                    seg_pred = None
                else:  # segmentation
                    seg_pred = outputs[1] if len(outputs) > 1 and outputs[1] is not None else outputs[0]
                    depth_pred = None
            else:
                # Single output
                if self.config["General"]["type"] == "depth":
                    depth_pred = outputs
                    seg_pred = None
                else:
                    seg_pred = outputs
                    depth_pred = None
        
        results = {}
        
        # Process depth prediction
        if depth_pred is not None:
            depth_np = depth_pred.squeeze().cpu().numpy()
            # Normalize depth to reasonable range
            if depth_np.max() > depth_np.min():
                depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
            else:
                depth_normalized = depth_np
            
            results['depth'] = depth_normalized
            results['depth_raw'] = depth_np
        
        # Process segmentation prediction
        if seg_pred is not None:
            if seg_pred.dim() > 3:  # Multi-class
                seg_np = torch.argmax(seg_pred.squeeze(), dim=0).cpu().numpy()
            else:
                seg_np = seg_pred.squeeze().cpu().numpy()
            
            results['segmentation'] = seg_np
        
        return results
    
    def predict_batch(self, 
                     tactile_images: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        Run batch inference on multiple tactile images
        
        Args:
            tactile_images: List of input tactile images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in tactile_images:
            result = self.predict(image)
            results.append(result)
        return results
    
    def visualize_prediction(self, 
                           tactile_image: np.ndarray,
                           predictions: Dict[str, np.ndarray],
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Create visualization of tactile image and predictions
        
        Args:
            tactile_image: Original tactile image
            predictions: Prediction results from predict()
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image as numpy array
        """
        import matplotlib.pyplot as plt
        
        # Determine number of subplots
        num_plots = 1  # Original image
        if 'depth' in predictions:
            num_plots += 1
        if 'segmentation' in predictions:
            num_plots += 1
        
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Original tactile image
        axes[plot_idx].imshow(tactile_image)
        axes[plot_idx].set_title("Tactile Image")
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # Depth prediction
        if 'depth' in predictions:
            depth_vis = apply_jet_colormap(Image.fromarray((predictions['depth'] * 255).astype(np.uint8)))
            axes[plot_idx].imshow(np.array(depth_vis))
            axes[plot_idx].set_title("Predicted Depth")
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # Segmentation prediction
        if 'segmentation' in predictions:
            seg_vis = predictions['segmentation']
            axes[plot_idx].imshow(seg_vis, cmap='gray')
            axes[plot_idx].set_title("Predicted Contact")
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Convert to numpy array
        fig.canvas.draw()
        
        # Handle different matplotlib versions
        try:
            # Try newer matplotlib method
            buf = fig.canvas.buffer_rgba()
            vis_array = np.asarray(buf)[:, :, :3]  # Drop alpha channel
        except AttributeError:
            try:
                # Try older method
                vis_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                vis_array = vis_array[:, :, 1:4]  # ARGB to RGB
            except AttributeError:
                # Fallback: save to buffer and read back
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='rgb')
                buf.seek(0)
                vis_array = np.frombuffer(buf.read(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return vis_array


class GaussianFeelsTactileIntegration:
    """
    Integration class for using tactile transformer with gaussianfeels gaussian optimization
    """
    
    def __init__(self, 
                 tactile_pipeline: TactileTransformerPipeline,
                 confidence_threshold: float = 0.1):
        """
        Initialize integration with gaussianfeels
        
        Args:
            tactile_pipeline: Trained tactile transformer pipeline
            confidence_threshold: Minimum confidence for tactile measurements
        """
        self.tactile_pipeline = tactile_pipeline
        self.confidence_threshold = confidence_threshold
        
        logger.info("GaussianFeels tactile integration initialized")
    
    def process_tactile_data(self, 
                           tactile_frames: List[np.ndarray],
                           finger_poses: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process tactile data for integration with gaussian optimization
        
        Args:
            tactile_frames: List of tactile images
            finger_poses: Optional finger pose information
            
        Returns:
            Processed tactile data for gaussian integration
        """
        results = {
            'depths': [],
            'contact_masks': [],
            'contact_points': [],
            'surface_normals': [],
            'confidence_scores': []
        }
        
        for i, frame in enumerate(tactile_frames):
            # Get predictions
            predictions = self.tactile_pipeline.predict(frame)
            
            # Extract depth information
            if 'depth' in predictions:
                depth = predictions['depth']
                results['depths'].append(depth)
                
                # Estimate surface normals from depth
                normals = self._compute_surface_normals(depth)
                results['surface_normals'].append(normals)
            
            # Extract contact information
            if 'segmentation' in predictions:
                contact_mask = predictions['segmentation'] > 0.5
                results['contact_masks'].append(contact_mask)
                
                # Extract contact points
                contact_points = self._extract_contact_points(
                    depth if 'depth' in predictions else None,
                    contact_mask,
                    finger_poses[i] if finger_poses else None
                )
                results['contact_points'].append(contact_points)
            
            # Compute confidence (simplified)
            confidence = self._compute_confidence(predictions)
            results['confidence_scores'].append(confidence)
        
        return results
    
    def _compute_surface_normals(self, depth: np.ndarray) -> np.ndarray:
        """Compute surface normals from depth map"""
        # Simple gradient-based normal computation
        grad_y, grad_x = np.gradient(depth)
        
        # Create 3D gradient vectors
        grad_3d = np.stack([-grad_x, -grad_y, np.ones_like(depth)], axis=-1)
        
        # Normalize to get unit normals
        norms = np.linalg.norm(grad_3d, axis=-1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)  # Avoid division by zero
        normals = grad_3d / norms
        
        return normals
    
    def _extract_contact_points(self, 
                              depth: Optional[np.ndarray],
                              contact_mask: np.ndarray,
                              finger_pose: Optional[np.ndarray] = None) -> List[Tuple[float, float, float]]:
        """Extract 3D contact points from tactile data"""
        contact_points = []
        
        if depth is None:
            return contact_points
        
        # Find contact regions
        contact_coords = np.where(contact_mask)
        
        if len(contact_coords[0]) == 0:
            return contact_points
        
        # Convert pixel coordinates to 3D points
        # This is a simplified version - real implementation would use sensor calibration
        for y, x in zip(contact_coords[0], contact_coords[1]):
            # Convert to normalized coordinates
            norm_x = (x / contact_mask.shape[1] - 0.5) * 2  # [-1, 1]
            norm_y = (y / contact_mask.shape[0] - 0.5) * 2  # [-1, 1]
            z = depth[y, x]
            
            # Simple scaling to get reasonable 3D coordinates
            # In reality, this would use proper camera/sensor calibration
            point_3d = [norm_x * 0.01, norm_y * 0.01, z * 0.005]  # Scale to ~cm
            
            # Transform by finger pose if provided
            if finger_pose is not None:
                point_hom = np.array([*point_3d, 1.0])
                point_world = finger_pose @ point_hom
                contact_points.append(tuple(point_world[:3]))
            else:
                contact_points.append(tuple(point_3d))
        
        return contact_points
    
    def _compute_confidence(self, predictions: Dict[str, np.ndarray]) -> float:
        """Compute confidence score for predictions"""
        confidence = 1.0
        
        if 'depth' in predictions:
            depth = predictions['depth']
            # Higher variance indicates less confidence
            depth_var = np.var(depth)
            depth_confidence = np.exp(-depth_var * 10)  # Exponential decay with variance
            confidence = min(confidence, depth_confidence)
        
        if 'segmentation' in predictions:
            seg = predictions['segmentation']
            # Check segmentation confidence (how close to 0 or 1)
            seg_confidence = np.mean(np.abs(seg - 0.5) * 2)  # Distance from uncertain (0.5)
            confidence = min(confidence, seg_confidence)
        
        return confidence
    
    def filter_low_confidence_data(self, 
                                  tactile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out low-confidence tactile measurements"""
        filtered_data = {key: [] for key in tactile_data.keys()}
        
        for i, confidence in enumerate(tactile_data['confidence_scores']):
            if confidence >= self.confidence_threshold:
                for key, values in tactile_data.items():
                    if i < len(values):
                        filtered_data[key].append(values[i])
        
        return filtered_data


def create_tactile_pipeline_from_checkpoint(checkpoint_dir: str,
                                           device: str = "cuda") -> TactileTransformerPipeline:
    """
    Convenience function to create tactile pipeline from training checkpoint
    
    Args:
        checkpoint_dir: Directory containing model and config files
        device: Device to run inference on
        
    Returns:
        Initialized tactile transformer pipeline
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # Find model file
    model_files = list(checkpoint_path.glob("*.p")) + list(checkpoint_path.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")
    
    model_path = str(model_files[0])  # Use first found model
    
    # Find config file
    config_files = list(checkpoint_path.glob("*.yaml")) + list(checkpoint_path.glob("*.yml"))
    if config_files:
        config_path = str(config_files[0])
    else:
        # Use default config
        config_path = str(Path(__file__).resolve().parents[5] / 
                         "misc/configs/training/tactile_transformer/vit_training.yaml")
    
    return TactileTransformerPipeline(model_path, config_path, device)