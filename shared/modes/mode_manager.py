# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Mode management for GaussianFeels with tactile-first default.
Handles graceful transitions between tactile-only, tactile+camera, and camera-dominant modes.
"""

import time
import threading
from enum import Enum
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
import numpy as np


class OperationMode(Enum):
    """GaussianFeels operation modes - ULTRATHINK STRICT"""
    TACTILE_ONLY = "tactile_only"           # Pure tactile sensing
    TACTILE_FIRST = "tactile_first"         # Tactile primary, camera secondary
    BALANCED = "balanced"                   # Equal tactile and camera weight
    CAMERA_FIRST = "camera_first"           # Camera primary, tactile secondary  
    CAMERA_ONLY = "camera_only"             # Pure camera sensing


@dataclass
class ModeTransition:
    """Represents a mode transition with conditions"""
    from_mode: OperationMode
    to_mode: OperationMode
    condition_met: bool = False
    stability_required: float = 5.0  # seconds
    transition_time: float = 2.0     # seconds for gradual transition
    
    
@dataclass
class ModeMetrics:
    """Performance metrics for each mode"""
    rmse: float  # REQUIRED - NO DEFAULTS
    fps: float   # REQUIRED - NO DEFAULTS
    stability_score: float  # REQUIRED - NO DEFAULTS
    error_count: int
    last_update: float = field(default_factory=time.time)
    
    def update(self, rmse: float, fps: float, stability_score: float):
        """Update metrics"""
        self.rmse = rmse
        self.fps = fps
        self.stability_score = stability_score
        self.last_update = time.time()
        
    def is_stable(self, threshold: float = 0.8) -> bool:
        """Check if mode is stable"""
        return self.stability_score >= threshold and self.rmse < 0.1


class ModeManager:
    """Manages operation mode transitions and stability detection"""
    
    def __init__(self, 
                 required_mode: OperationMode,  # REQUIRED - NO DEFAULTS
                 stability_threshold: float,
                 transition_callback: Optional[Callable[[OperationMode, OperationMode], None]] = None):
        
        self.current_mode = required_mode
        self.target_mode = required_mode
        self.stability_threshold = stability_threshold
        self.transition_callback = transition_callback
        
        # Mode metrics tracking
        self.mode_metrics: Dict[OperationMode, ModeMetrics] = {
            mode: ModeMetrics() for mode in OperationMode
        }
        
        # Transition rules
        self.transitions = self._setup_transitions()
        
        # State tracking
        self.is_transitioning = False
        self.transition_start_time = 0.0
        self.last_stability_check = time.time()
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
    def _setup_transitions(self) -> List[ModeTransition]:
        """Setup allowed mode transitions with conditions"""
        return [
            # From TACTILE_ONLY
            ModeTransition(OperationMode.TACTILE_ONLY, OperationMode.TACTILE_FIRST, stability_required=3.0),
            
            # From TACTILE_FIRST (default mode)
            ModeTransition(OperationMode.TACTILE_FIRST, OperationMode.BALANCED, stability_required=5.0),
            ModeTransition(OperationMode.TACTILE_FIRST, OperationMode.TACTILE_ONLY, stability_required=2.0),
            
            # From BALANCED
            ModeTransition(OperationMode.BALANCED, OperationMode.CAMERA_FIRST, stability_required=7.0),
            ModeTransition(OperationMode.BALANCED, OperationMode.TACTILE_FIRST, stability_required=3.0),
            
            # From CAMERA_FIRST
            ModeTransition(OperationMode.CAMERA_FIRST, OperationMode.CAMERA_ONLY, stability_required=10.0),
            ModeTransition(OperationMode.CAMERA_FIRST, OperationMode.BALANCED, stability_required=3.0),
            
            # REMOVED: No emergency transitions allowed
        ]
    
    def update_metrics(self, 
                      mode: OperationMode,
                      rmse: float, 
                      fps: float, 
                      stability_score: float):
        """Update performance metrics for a specific mode"""
        with self._lock:
            self.mode_metrics[mode].update(rmse, fps, stability_score)
            
            # Check for automatic transitions
            if mode == self.current_mode:
                self._check_auto_transitions()
    
    def _check_auto_transitions(self):
        """Check if automatic mode transitions should occur"""
        current_time = time.time()
        
        # Only check every 2 seconds to avoid oscillation
        if current_time - self.last_stability_check < 2.0:
            return
            
        self.last_stability_check = current_time
        current_metrics = self.mode_metrics[self.current_mode]
        
        # ULTRATHINK: No emergency fallback - FAIL FAST
        if (current_metrics.rmse > 0.5 or 
            current_metrics.fps < 10.0 or 
            current_metrics.stability_score < 0.3):
            
            raise RuntimeError(f"CRITICAL FAILURE: Mode {self.current_mode.value} degraded below acceptable thresholds. rmse={current_metrics.rmse}, fps={current_metrics.fps}, stability={current_metrics.stability_score}")
        
        # ULTRATHINK: No automatic transitions - explicit mode changes only
        if not current_metrics.is_stable(self.stability_threshold):
            raise RuntimeError(f"Mode {self.current_mode.value} is unstable. Explicit intervention required.")
    
    def validate_transition_request(self, target_mode: OperationMode) -> bool:
        """Validate if transition to target mode is allowed and conditions are met"""
        if target_mode == self.current_mode:
            return True  # Already in target mode
            
        transition = self._find_transition(self.current_mode, target_mode)
        if not transition:
            raise ValueError(f"No transition rule defined from {self.current_mode.name} to {target_mode.name}")
            
        # Check transition conditions
        if not self._check_transition_conditions(transition):
            raise RuntimeError(
                f"Transition conditions not met for {self.current_mode.name} -> {target_mode.name}. "
                f"Required stability duration: {transition.min_stable_duration}s, "
                f"Error threshold: {transition.error_threshold}"
            )
            
        return True
    
    def execute_validated_transition(self, target_mode: OperationMode) -> bool:
        """Execute pre-validated transition with comprehensive error handling"""
        try:
            if not self.validate_transition_request(target_mode):
                return False
                
            previous_mode = self.current_mode
            
            # Update mode with atomic operation
            self.current_mode = target_mode
            self.mode_metrics[target_mode].last_update = time.time()
            self.mode_metrics[target_mode].transition_count += 1
            
            # Log successful transition
            print(f"Mode transition successful: {previous_mode.name} -> {target_mode.name}")
            
            return True
            
        except Exception as e:
            # Transition failed - maintain current mode and re-raise
            raise RuntimeError(f"Mode transition failed: {e}")
    
    def _find_transition(self, from_mode: OperationMode, to_mode: OperationMode) -> Optional[ModeTransition]:
        """Find transition rule between two modes"""
        for transition in self.transitions:
            if transition.from_mode == from_mode and transition.to_mode == to_mode:
                return transition
        return None
    
    def _check_transition_conditions(self, transition: ModeTransition) -> bool:
        """Check if transition conditions are met"""
        current_metrics = self.mode_metrics[self.current_mode]
        
        # Must be stable for required duration
        stable_duration = time.time() - current_metrics.last_update
        return stable_duration >= transition.stability_required
    
    def _initiate_transition(self, target_mode: OperationMode, reason: str):
        """Initiate mode transition"""
        if self.is_transitioning:
            return  # Already transitioning
            
        if target_mode == self.current_mode:
            return  # Already in target mode
            
        self.target_mode = target_mode
        self.is_transitioning = True
        self.transition_start_time = time.time()
        
        print(f"Mode transition: {self.current_mode.value} → {target_mode.value} ({reason})")
        
        if self.transition_callback:
            self.transition_callback(self.current_mode, target_mode)
    
    def set_mode(self, mode: OperationMode):
        """Set operation mode - STRICT"""
        with self._lock:
            if self.is_transitioning:
                raise RuntimeError("Cannot change mode during transition. Wait for completion.")
            self.current_mode = mode
            self.target_mode = mode
            self.is_transitioning = False
            print(f"Mode set to: {mode.value}")
    
    def get_mode_weights(self) -> Dict[str, float]:
        """Get current sensor fusion weights based on mode"""
        # Handle transitions with gradual weight changes
        if self.is_transitioning:
            progress = min(1.0, (time.time() - self.transition_start_time) / 2.0)  # 2 second transition
            
            if progress >= 1.0:
                # Transition complete
                self.current_mode = self.target_mode
                self.is_transitioning = False
                print(f"Transition completed to: {self.current_mode.value}")
                
            # Interpolate weights during transition
            current_weights = self._get_mode_weights(self.current_mode)
            target_weights = self._get_mode_weights(self.target_mode)
            
            return {
                'tactile': current_weights['tactile'] * (1 - progress) + target_weights['tactile'] * progress,
                'camera': current_weights['camera'] * (1 - progress) + target_weights['camera'] * progress
            }
        
        return self._get_mode_weights(self.current_mode)
    
    def _get_mode_weights(self, mode: OperationMode) -> Dict[str, float]:
        """Get sensor weights for a specific mode"""
        weights = {
            OperationMode.TACTILE_ONLY: {'tactile': 1.0, 'camera': 0.0},
            OperationMode.TACTILE_FIRST: {'tactile': 0.8, 'camera': 0.2},
            OperationMode.BALANCED: {'tactile': 0.5, 'camera': 0.5},
            OperationMode.CAMERA_FIRST: {'tactile': 0.2, 'camera': 0.8},
            OperationMode.CAMERA_ONLY: {'tactile': 0.0, 'camera': 1.0},
        }
        
        if mode not in weights:
            raise ValueError(f"Invalid operation mode: {mode}")
        
        return weights[mode]
    
    def get_status(self) -> Dict:
        """Get comprehensive mode manager status"""
        with self._lock:
            return {
                'current_mode': self.current_mode.value,
                'target_mode': self.target_mode.value,
                'is_transitioning': self.is_transitioning,
                'weights': self.get_mode_weights(),
                'metrics': {
                    mode.value: {
                        'rmse': metrics.rmse,
                        'fps': metrics.fps,
                        'stability_score': metrics.stability_score,
                        'is_stable': metrics.is_stable(self.stability_threshold),
                        'age_seconds': time.time() - metrics.last_update
                    }
                    for mode, metrics in self.mode_metrics.items()
                }
            }
    
    def force_emergency_mode(self):
        """REMOVED: No emergency mode allowed"""
        raise NotImplementedError("Emergency mode removed. Handle failures explicitly.")
    
    def reset_to_default(self):
        """REMOVED: No default mode reset allowed"""
        raise NotImplementedError("Default mode reset removed. Set explicit mode.")


if __name__ == "__main__":
    # Example usage
    def transition_callback(from_mode, to_mode):
        print(f"Transition callback: {from_mode.value} → {to_mode.value}")
    
    manager = ModeManager(
        required_mode=OperationMode.TACTILE_FIRST,
        stability_threshold=0.8,
        transition_callback=transition_callback
    )
    
    # Simulate some mode updates
    import time
    
    print("Starting in tactile-first mode...")
    print(f"Status: {manager.get_status()}")
    
    # Simulate stable tactile operation
    for i in range(10):
        manager.update_metrics(OperationMode.TACTILE_FIRST, 
                             rmse=0.05, fps=45.0, stability_score=0.9)
        time.sleep(0.1)
    
    print(f"After stable operation: {manager.get_status()}")
    
    # Test manual mode change
    manager.set_mode(OperationMode.BALANCED)
    print(f"Manual switch to balanced: {manager.get_status()}")
    
    print("ULTRATHINK MODE: Fast-fail strict operation mode manager ready")