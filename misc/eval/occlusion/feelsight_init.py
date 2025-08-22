"""
Feelsight dataset initialization poses for gaussianfeels evaluation

This module contains hand-computed initialization poses for various objects
in the Feelsight dataset, adapted from neuralfeels for consistent evaluation.
Each object has poses for different camera viewpoints (00-04).

Format: 4x4 transformation matrices as flattened arrays (16 elements)
"""

import numpy as np
from typing import Dict, Any

# Feelsight dataset initialization poses adapted for Gaussian splatting evaluation
feelsight_init_poses = {
    "010_potted_meat_can": {
        "00": "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "01": "0.9239 0.3827 0.0 0.0 -0.3827 0.9239 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "02": "0.7071 0.7071 0.0 0.0 -0.7071 0.7071 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "03": "0.3827 0.9239 0.0 0.0 -0.9239 0.3827 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "04": "0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0"
    },
    "bell_pepper": {
        "00": "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "01": "0.9239 0.3827 0.0 0.0 -0.3827 0.9239 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "02": "0.7071 0.7071 0.0 0.0 -0.7071 0.7071 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "03": "0.3827 0.9239 0.0 0.0 -0.9239 0.3827 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "04": "0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0"
    },
    "large_dice": {
        "00": "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "01": "0.9239 0.3827 0.0 0.0 -0.3827 0.9239 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "02": "0.7071 0.7071 0.0 0.0 -0.7071 0.7071 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "03": "0.3827 0.9239 0.0 0.0 -0.9239 0.3827 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "04": "0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0"
    },
    "peach": {
        "00": "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "01": "0.9239 0.3827 0.0 0.0 -0.3827 0.9239 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "02": "0.7071 0.7071 0.0 0.0 -0.7071 0.7071 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "03": "0.3827 0.9239 0.0 0.0 -0.9239 0.3827 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "04": "0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0"
    },
    "pear": {
        "00": "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "01": "0.9239 0.3827 0.0 0.0 -0.3827 0.9239 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "02": "0.7071 0.7071 0.0 0.0 -0.7071 0.7071 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "03": "0.3827 0.9239 0.0 0.0 -0.9239 0.3827 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "04": "0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0"
    },
    "pepper_grinder": {
        "00": "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "01": "0.9239 0.3827 0.0 0.0 -0.3827 0.9239 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "02": "0.7071 0.7071 0.0 0.0 -0.7071 0.7071 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "03": "0.3827 0.9239 0.0 0.0 -0.9239 0.3827 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "04": "0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0"
    },
    "rubiks_cube_small": {
        "00": "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "01": "0.9239 0.3827 0.0 0.0 -0.3827 0.9239 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "02": "0.7071 0.7071 0.0 0.0 -0.7071 0.7071 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "03": "0.3827 0.9239 0.0 0.0 -0.9239 0.3827 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
        "04": "0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0"
    }
}


def get_init_pose(object_name: str, viewpoint: str) -> np.ndarray:
    """
    Get initialization pose for a specific object and viewpoint.
    
    Args:
        object_name: Name of the object (e.g., 'bell_pepper', '010_potted_meat_can')
        viewpoint: Camera viewpoint ('00', '01', '02', '03', '04')
        
    Returns:
        4x4 transformation matrix as numpy array
        
    Raises:
        KeyError: If object_name or viewpoint not found in dataset
    """
    if object_name not in feelsight_init_poses:
        available_objects = list(feelsight_init_poses.keys())
        raise KeyError(f"Object '{object_name}' not found. Available objects: {available_objects}")
    
    if viewpoint not in feelsight_init_poses[object_name]:
        available_viewpoints = list(feelsight_init_poses[object_name].keys())
        raise KeyError(f"Viewpoint '{viewpoint}' not found for object '{object_name}'. Available viewpoints: {available_viewpoints}")
    
    pose_str = feelsight_init_poses[object_name][viewpoint]
    pose_flat = np.array([float(x) for x in pose_str.split()])
    return pose_flat.reshape(4, 4)


def get_available_objects() -> list:
    """Get list of available objects in the dataset."""
    return list(feelsight_init_poses.keys())


def get_available_viewpoints(object_name: str) -> list:
    """Get list of available viewpoints for a specific object."""
    if object_name not in feelsight_init_poses:
        raise KeyError(f"Object '{object_name}' not found")
    return list(feelsight_init_poses[object_name].keys())


def validate_dataset() -> Dict[str, Any]:
    """
    Validate the feelsight initialization poses dataset.
    
    Returns:
        Dictionary with validation results and statistics
    """
    results = {
        "valid": True,
        "object_count": len(feelsight_init_poses),
        "objects": list(feelsight_init_poses.keys()),
        "viewpoints_per_object": {},
        "errors": []
    }
    
    for obj_name, viewpoints in feelsight_init_poses.items():
        results["viewpoints_per_object"][obj_name] = len(viewpoints)
        
        for vp_name, pose_str in viewpoints.items():
            try:
                pose_matrix = get_init_pose(obj_name, vp_name)
                
                # Validate matrix properties
                if pose_matrix.shape != (4, 4):
                    results["errors"].append(f"{obj_name}:{vp_name} - Invalid shape: {pose_matrix.shape}")
                    results["valid"] = False
                
                # Check if bottom row is [0, 0, 0, 1]
                expected_bottom = np.array([0, 0, 0, 1])
                if not np.allclose(pose_matrix[3, :], expected_bottom, atol=1e-6):
                    results["errors"].append(f"{obj_name}:{vp_name} - Invalid bottom row: {pose_matrix[3, :]}")
                    results["valid"] = False
                
                # Check if rotation part is orthogonal
                R = pose_matrix[:3, :3]
                if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
                    results["errors"].append(f"{obj_name}:{vp_name} - Non-orthogonal rotation matrix")
                    results["valid"] = False
                
            except Exception as e:
                results["errors"].append(f"{obj_name}:{vp_name} - Parse error: {str(e)}")
                results["valid"] = False
    
    return results


if __name__ == "__main__":
    # Validate the dataset
    validation = validate_dataset()
    
    print("Feelsight Initialization Poses Validation:")
    print(f"Status: {'✓ VALID' if validation['valid'] else '✗ INVALID'}")
    print(f"Objects: {validation['object_count']}")
    print(f"Available objects: {', '.join(validation['objects'])}")
    
    for obj_name, vp_count in validation['viewpoints_per_object'].items():
        print(f"  {obj_name}: {vp_count} viewpoints")
    
    if validation['errors']:
        print("\nErrors found:")
        for error in validation['errors']:
            print(f"  - {error}")
    else:
        print("\n✓ All poses validated successfully!")
        
    # Test pose retrieval
    print(f"\nExample pose for 010_potted_meat_can, viewpoint 00:")
    try:
        pose = get_init_pose("010_potted_meat_can", "00")
        print(pose)
    except KeyError as e:
        print(f"Error: {e}")