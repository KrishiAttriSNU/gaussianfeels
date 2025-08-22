# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tactile dataset for training depth transformer

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging
from glob import glob
from .utils import get_splitted_dataset

logger = logging.getLogger(__name__)


class TactileDataset(Dataset):
    """
    Dataset class for tactile depth estimation and segmentation training
    
    Loads tactile images with corresponding depth maps and contact masks
    for training the TouchVIT/DPTModel architecture.
    """
    
    def __init__(self, config, split="train", transform_image=None, 
                 transform_depth=None, transform_seg=None):
        """
        Initialize tactile dataset
        
        Args:
            config: Hydra configuration object
            split: "train", "val", or "test"
            transform_image: Transform for input tactile images
            transform_depth: Transform for depth maps
            transform_seg: Transform for segmentation masks
        """
        self.config = config
        self.split = split
        self.transform_image = transform_image
        self.transform_depth = transform_depth
        self.transform_seg = transform_seg
        self.type = config["General"]["type"]
        
        # Get dataset paths
        self.dataset_path = config["Dataset"]["paths"]["path_dataset"]
        self.image_dir = config["Dataset"]["paths"]["path_images"]
        self.depth_dir = config["Dataset"]["paths"]["path_depths"]
        self.seg_dir = config["Dataset"]["paths"]["path_segmentations"]
        
        # File extensions
        self.img_ext = config["Dataset"]["extensions"]["ext_images"]
        self.depth_ext = config["Dataset"]["extensions"]["ext_depths"]
        self.seg_ext = config["Dataset"]["extensions"]["ext_segmentations"]
        
        # Load file paths
        self._load_file_paths()
        
        logger.info(f"Initialized {split} dataset with {len(self.image_paths)} samples")
    
    def _load_file_paths(self):
        """Load and split file paths for the dataset"""
        
        # Find all objects in the dataset
        dataset_path = self.dataset_path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Get all image files across all object subdirectories
        all_image_paths = []
        for obj_dir in glob(os.path.join(dataset_path, "*")):
            if os.path.isdir(obj_dir):
                obj_name = os.path.basename(obj_dir)
                image_pattern = os.path.join(obj_dir, self.image_dir, f"*{self.img_ext}")
                obj_images = glob(image_pattern)
                all_image_paths.extend(obj_images)
        
        if not all_image_paths:
            raise ValueError(f"No images found in {dataset_path}")
        
        # Sort for consistency
        all_image_paths.sort()
        
        # Split dataset
        np.random.seed(self.config["General"]["seed"])
        indices = np.arange(len(all_image_paths))
        np.random.shuffle(indices)
        
        # Calculate split indices
        n_total = len(all_image_paths)
        n_train = int(n_total * self.config["Dataset"]["splits"]["split_train"])
        n_val = int(n_total * self.config["Dataset"]["splits"]["split_val"])
        
        if self.split == "train":
            selected_indices = indices[:n_train]
        elif self.split == "val":
            selected_indices = indices[n_train:n_train + n_val]
        else:  # test
            selected_indices = indices[n_train + n_val:]
        
        # Get selected file paths
        self.image_paths = [all_image_paths[i] for i in selected_indices]
        
        # Generate corresponding depth and segmentation paths
        self.depth_paths = []
        self.seg_paths = []
        
        for img_path in self.image_paths:
            # Parse object directory structure
            obj_dir = os.path.dirname(os.path.dirname(img_path))  # Go up two levels from image
            img_basename = os.path.basename(img_path)
            img_name = os.path.splitext(img_basename)[0]
            
            # Generate corresponding paths
            depth_path = os.path.join(obj_dir, self.depth_dir, img_name + self.depth_ext)
            seg_path = os.path.join(obj_dir, self.seg_dir, img_name + self.seg_ext)
            
            self.depth_paths.append(depth_path)
            self.seg_paths.append(seg_path)
    
    def __len__(self):
        """Return dataset length"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get item from dataset
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Tuple of (image, depth, segmentation) tensors
        """
        try:
            # Load image
            image = Image.open(self.image_paths[idx]).convert("RGB")
            
            # Load depth (if needed)
            if self.type in ["full", "depth"]:
                if os.path.exists(self.depth_paths[idx]):
                    depth = Image.open(self.depth_paths[idx])
                else:
                    # Create dummy depth if file doesn't exist
                    depth = Image.new("L", image.size, 0)
            else:
                depth = Image.new("L", image.size, 0)
            
            # Load segmentation (if needed)
            if self.type in ["full", "segmentation"]:
                if os.path.exists(self.seg_paths[idx]):
                    segmentation = Image.open(self.seg_paths[idx])
                else:
                    # Create dummy segmentation if file doesn't exist
                    segmentation = Image.new("RGB", image.size, (0, 0, 0))
            else:
                segmentation = Image.new("RGB", image.size, (0, 0, 0))
            
            # Apply transforms
            if self.transform_image:
                image = self.transform_image(image)
            
            if self.transform_depth:
                depth = self.transform_depth(depth)
            
            if self.transform_seg:
                segmentation = self.transform_seg(segmentation)
            
            # Convert to tensors if not already
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(np.array(image))
            
            if not isinstance(depth, torch.Tensor):
                depth = torch.tensor(np.array(depth))
            
            if not isinstance(segmentation, torch.Tensor):
                segmentation = torch.tensor(np.array(segmentation))
            
            # Ensure proper dimensions
            if image.dim() == 3 and image.shape[0] != 3:
                image = image.permute(2, 0, 1)  # HWC -> CHW
            
            if depth.dim() == 3:
                depth = depth.unsqueeze(0)  # Add channel dimension
            elif depth.dim() == 2:
                depth = depth.unsqueeze(0)
            
            if segmentation.dim() == 3:
                segmentation = segmentation.unsqueeze(0)  # Add batch-like dimension
            elif segmentation.dim() == 2:
                segmentation = segmentation.unsqueeze(0)
            
            return image, depth, segmentation
            
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            logger.error(f"Paths: img={self.image_paths[idx]}, depth={self.depth_paths[idx]}, seg={self.seg_paths[idx]}")
            
            # Return dummy data to prevent training crashes
            dummy_size = (3, 224, 224)
            image = torch.zeros(dummy_size)
            depth = torch.zeros((1, 224, 224))
            segmentation = torch.zeros((1, 224, 224))
            
            return image, depth, segmentation
    
    def get_stats(self):
        """Get dataset statistics"""
        stats = {
            "total_samples": len(self),
            "split": self.split,
            "type": self.type,
            "dataset_path": self.dataset_path
        }
        
        if len(self) > 0:
            # Sample a few items to get data statistics
            sample_indices = np.random.choice(len(self), min(10, len(self)), replace=False)
            
            image_stats = {"min": [], "max": [], "mean": []}
            depth_stats = {"min": [], "max": [], "mean": []}
            
            for idx in sample_indices:
                try:
                    image, depth, seg = self[idx]
                    
                    if isinstance(image, torch.Tensor):
                        image_stats["min"].append(image.min().item())
                        image_stats["max"].append(image.max().item())
                        image_stats["mean"].append(image.mean().item())
                    
                    if isinstance(depth, torch.Tensor):
                        depth_stats["min"].append(depth.min().item())
                        depth_stats["max"].append(depth.max().item())
                        depth_stats["mean"].append(depth.mean().item())
                        
                except Exception as e:
                    logger.warning(f"Could not compute stats for sample {idx}: {e}")
                    continue
            
            # Aggregate statistics
            if image_stats["min"]:
                stats["image_stats"] = {
                    "min": np.mean(image_stats["min"]),
                    "max": np.mean(image_stats["max"]),
                    "mean": np.mean(image_stats["mean"])
                }
            
            if depth_stats["min"]:
                stats["depth_stats"] = {
                    "min": np.mean(depth_stats["min"]),
                    "max": np.mean(depth_stats["max"]),
                    "mean": np.mean(depth_stats["mean"])
                }
        
        return stats


# Alias for backward compatibility
TactileDepthDataset = TactileDataset