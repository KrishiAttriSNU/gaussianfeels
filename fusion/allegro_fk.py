"""
Allegro Hand Forward Kinematics for FeelSight Real Data
======================================================

Computes finger poses from joint states and base poses for feelsight_real data
that lacks pre-computed finger_poses (unlike simulation data).

Usage:
    from fusion.allegro_fk import AllegroForwardKinematics
    
    fk = AllegroForwardKinematics()
    finger_poses = fk.compute_finger_poses(joint_state, base_pose)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

from torchkin import Robot, get_forward_kinematics_fns
import theseus as th


class AllegroForwardKinematics:
    """Forward kinematics for Allegro hand using URDF and torchkin"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
            
        # Load URDF
        urdf_path = '/home/krishi/gaussianfeels/data/assets/allegro/allegro_digit_left_ball.urdf'
        if not Path(urdf_path).exists():
            logger.error(f"URDF file not found: {urdf_path}")
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
            
        self.robot, self.fkin, self.links, self.joint_map = self._load_robot(urdf_path, 16, device)
        logger.debug(f"✅ Loaded Allegro URDF with {len(self.links)} links")
        
    def _load_robot(self, urdf_file: str, num_dofs: int, device) -> Tuple:
        """Load robot from URDF file and cache FK functions"""
        robot = Robot.from_urdf_file(urdf_file, device=device)
        # Use exact same order as standard approach to match ground truth
        links = {
            'digit_index': 'link_3.0_tip',
            'digit_middle': 'link_7.0_tip', 
            'digit_ring': 'link_11.0_tip',
            'digit_thumb': 'link_15.0_tip',
        }

        # FK function is applied breadth-first, so swap the indices from the allegro convention
        joint_map = torch.tensor(
            [joint.id for joint in robot.joint_map.values() if joint.id < num_dofs],
            device=device,
        )
        
        fkin, *_ = get_forward_kinematics_fns(robot, list(links.values()))
        return (robot, fkin, links, joint_map)
    
    def _hora_to_neural(self, finger_poses: np.ndarray) -> np.ndarray:
        """Convert the DIGIT urdf reference frame to neural SLAM frame"""
        hora_to_neural_inv = np.linalg.inv(np.array([
            [0.000000, -1.000000, 0.000000, 0.000021],
            [0.000000, 0.000000, 1.000000, -0.017545],
            [-1.000000, 0.000000, 0.000000, -0.002132],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ]))
        return finger_poses @ hora_to_neural_inv
    
    def compute_finger_poses(self, joint_state: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
        """
        Compute finger poses from joint state and base pose using forward kinematics
        
        Args:
            joint_state: (16,) array of joint angles
            base_pose: (4, 4) base transformation matrix
            
        Returns:
            finger_poses: (4, 4, 4) array - [thumb, index, middle, ring] transforms
        """
            
        joint_states = torch.tensor(joint_state, device=self.device, dtype=torch.float32)
        
        # Swap index and ring for left-hand, theseus FK requires this
        joint_states_theseus = joint_states.clone()
        joint_states_theseus[[0, 1, 2, 3]], joint_states_theseus[[8, 9, 10, 11]] = (
            joint_states_theseus[[8, 9, 10, 11]],
            joint_states_theseus[[0, 1, 2, 3]],
        )

        # Change to breadth-first order, theseus needs this too
        joint_states_theseus = joint_states_theseus[self.joint_map]
        j = th.Vector(
            tensor=joint_states_theseus.unsqueeze(0),
            name='joint_states',
        )
        
        # Forward kinematics
        link_poses = self.fkin(j.tensor)
        digit_poses = torch.vstack(link_poses).to(self.robot.device)
        digit_poses = th.SE3(tensor=digit_poses).to_matrix().cpu().numpy()

        # Apply base transformation
        base_tf = np.repeat(
            base_pose[np.newaxis, :, :], digit_poses.shape[0], axis=0
        )
        digit_poses = base_tf @ digit_poses
        
        # Apply hora_to_neural transformation
        digit_poses = self._hora_to_neural(digit_poses)
        
        # Match the ground truth ordering exactly: [thumb, index, middle, ring]
        # FK result order from links: [digit_index(0), digit_middle(1), digit_ring(2), digit_thumb(3)]
        # Correct mapping to GT order:
        #   thumb  <- digit_thumb (3)
        #   index  <- digit_index (0)
        #   middle <- digit_middle (1)
        #   ring   <- digit_ring (2)

        finger_poses = np.zeros((4, 4, 4), dtype=np.float32)
        finger_poses[0] = digit_poses[3].astype(np.float32)  # thumb
        finger_poses[1] = digit_poses[0].astype(np.float32)  # index
        finger_poses[2] = digit_poses[1].astype(np.float32)  # middle
        finger_poses[3] = digit_poses[2].astype(np.float32)  # ring
        
        return finger_poses


def compute_missing_finger_poses(allegro_data: dict, frame_idx: int) -> np.ndarray:
    """
    Compute finger poses for a specific frame when missing from data
    
    Args:
        allegro_data: Dictionary containing 'joint_state' and 'base_pose'
        frame_idx: Frame index to compute poses for
        
    Returns:
        finger_poses: (4, 4, 4) array - [thumb, index, middle, ring] transforms
    """
    fk = AllegroForwardKinematics()
    joint_state = allegro_data['joint_state'][frame_idx]
    base_pose = allegro_data['base_pose']
    
    return fk.compute_finger_poses(joint_state, base_pose)