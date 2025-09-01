# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Real-time VRAM and system memory monitoring for GaussianFeels.
Provides live tracking with alerts and automatic memory management.
"""

import torch
import psutil
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class MemorySnapshot:
    """Single memory measurement snapshot"""
    timestamp: float
    gpu_allocated_mb: float
    gpu_cached_mb: float
    gpu_free_mb: float
    system_used_mb: float
    system_available_mb: float
    gaussian_count: int
    
    
class MemoryAlert:
    """Memory alert levels and thresholds"""
    
    def __init__(self, 
                 warning_threshold: float = 0.75,
                 critical_threshold: float = 0.9,
                 emergency_threshold: float = 0.95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold  
        self.emergency_threshold = emergency_threshold
        
    def check_alert_level(self, usage_ratio: float) -> str:
        """Check current alert level based on usage ratio"""
        if usage_ratio >= self.emergency_threshold:
            return "EMERGENCY"
        elif usage_ratio >= self.critical_threshold:
            return "CRITICAL"
        elif usage_ratio >= self.warning_threshold:
            return "WARNING"
        return "OK"


class MemoryMonitor:
    """Real-time memory monitoring with alerts and automatic management"""
    
    def __init__(self, 
                 history_size: int = 1000,
                 update_interval: float = 0.1,
                 alert_callback: Optional[Callable[[str, float], None]] = None):
        self.history_size = history_size
        self.update_interval = update_interval
        self.alert_callback = alert_callback
        
        # Memory history
        self.history: deque = deque(maxlen=history_size)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Alert system
        self.alerts = MemoryAlert()
        self.last_alert_level = "OK"
        
        # GPU info
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required but not available")
        self.gpu_available = True
        self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
            
    def get_current_snapshot(self, gaussian_count: int = 0) -> MemorySnapshot:
        """Get current memory usage snapshot"""
        timestamp = time.time()
        
        if self.gpu_available:
            gpu_allocated = torch.cuda.memory_allocated()
            gpu_cached = torch.cuda.memory_reserved()
            gpu_free = self.gpu_total_memory - gpu_allocated
        else:
            gpu_allocated = gpu_cached = gpu_free = 0
            
        system_memory = psutil.virtual_memory()
        
        return MemorySnapshot(
            timestamp=timestamp,
            gpu_allocated_mb=gpu_allocated / (1024**2),
            gpu_cached_mb=gpu_cached / (1024**2),
            gpu_free_mb=gpu_free / (1024**2),
            system_used_mb=system_memory.used / (1024**2),
            system_available_mb=system_memory.available / (1024**2),
            gaussian_count=gaussian_count
        )
    
    def add_snapshot(self, gaussian_count: int = 0):
        """Add a memory snapshot to history"""
        snapshot = self.get_current_snapshot(gaussian_count)
        self.history.append(snapshot)
        
        # Check for alerts
        if self.gpu_available:
            gpu_usage_ratio = snapshot.gpu_allocated_mb / (self.gpu_total_memory / (1024**2))
            alert_level = self.alerts.check_alert_level(gpu_usage_ratio)
            
            if alert_level != self.last_alert_level and self.alert_callback:
                self.alert_callback(alert_level, gpu_usage_ratio)
                self.last_alert_level = alert_level
                
        return snapshot
    
    def start_monitoring(self, gaussian_count_getter: Optional[Callable[[], int]] = None):
        """Start continuous memory monitoring in background thread"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    gaussian_count = gaussian_count_getter() if gaussian_count_getter else 0
                    self.add_snapshot(gaussian_count)
                    time.sleep(self.update_interval)
                except Exception as e:
                    print(f"Memory monitoring error: {e}")
                    
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop continuous memory monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def get_stats(self, window_seconds: float = 10.0) -> Dict[str, float]:
        """Get memory statistics for recent time window"""
        if not self.history:
            return {}
            
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_snapshots = [s for s in self.history if s.timestamp >= cutoff_time]
        
        if not recent_snapshots:
            recent_snapshots = [self.history[-1]]  # Use latest snapshot
            
        gpu_allocated = [s.gpu_allocated_mb for s in recent_snapshots]
        gpu_free = [s.gpu_free_mb for s in recent_snapshots]
        gaussian_counts = [s.gaussian_count for s in recent_snapshots]
        
        return {
            'gpu_allocated_mean_mb': np.mean(gpu_allocated),
            'gpu_allocated_max_mb': np.max(gpu_allocated),
            'gpu_allocated_min_mb': np.min(gpu_allocated),
            'gpu_free_mean_mb': np.mean(gpu_free),
            'gpu_free_min_mb': np.min(gpu_free),
            'gaussian_count_mean': np.mean(gaussian_counts),
            'gaussian_count_max': np.max(gaussian_counts),
            'samples_count': len(recent_snapshots),
            'time_span_seconds': recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp
        }
    
    def get_trend(self, window_seconds: float = 30.0) -> Dict[str, str]:
        """Analyze memory usage trends"""
        if len(self.history) < 10:
            return {'trend': 'insufficient_data'}
            
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_snapshots = [s for s in self.history if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 5:
            return {'trend': 'insufficient_recent_data'}
            
        # Calculate trends using linear regression
        timestamps = np.array([s.timestamp for s in recent_snapshots])
        gpu_allocated = np.array([s.gpu_allocated_mb for s in recent_snapshots])
        gaussian_counts = np.array([s.gaussian_count for s in recent_snapshots])
        
        # Normalize timestamps
        timestamps = timestamps - timestamps[0]
        
        # GPU memory trend
        gpu_trend = np.polyfit(timestamps, gpu_allocated, 1)[0]  # slope
        gaussian_trend = np.polyfit(timestamps, gaussian_counts, 1)[0]  # slope
        
        def classify_trend(slope, threshold=0.1):
            if slope > threshold:
                return "increasing"
            elif slope < -threshold:
                return "decreasing"
            else:
                return "stable"
                
        return {
            'gpu_memory_trend': classify_trend(gpu_trend, 1.0),  # MB/second
            'gaussian_count_trend': classify_trend(gaussian_trend, 10.0),  # Gaussians/second
            'gpu_trend_rate_mb_per_sec': gpu_trend,
            'gaussian_trend_rate_per_sec': gaussian_trend
        }
    
    def force_cleanup(self):
        """Force GPU memory cleanup"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 = low, 1.0 = critical)"""
        if not self.history:
            return 0.0
            
        latest = self.history[-1]
        total_gpu_mb = self.gpu_total_memory / (1024**2)
        usage_ratio = latest.gpu_allocated_mb / total_gpu_mb
        
        return min(1.0, max(0.0, usage_ratio))
    
    def suggest_action(self) -> str:
        """Suggest memory management action based on current state"""
        # Monitoring-only approach - no automatic resource management
        # System assumes sufficient computational resources without automatic fallbacks
        # This function provides monitoring information only
        pressure = self.get_memory_pressure()
        trends = self.get_trend()
        
        # No automatic fallback mechanisms - monitoring only
        if trends.get('gpu_memory_trend') == 'increasing':
            return "INFO: Memory usage increasing (monitoring only, no automatic management)"
        else:
            return "OK: Memory usage stable (unbounded processing model)"


def create_default_alert_handler() -> Callable[[str, float], None]:
    """Create default alert handler that prints to console"""
    def handle_alert(level: str, usage_ratio: float):
        print(f"MEMORY ALERT [{level}]: GPU usage at {usage_ratio:.1%}")
        if level == "EMERGENCY":
            print("  -> Consider immediate memory cleanup!")
        elif level == "CRITICAL":
            print("  -> Enable aggressive memory management")
        elif level == "WARNING":
            print("  -> Monitor memory usage closely")
    
    return handle_alert


if __name__ == "__main__":
    # Example usage
    monitor = MemoryMonitor(alert_callback=create_default_alert_handler())
    
    # Test monitoring
    print("Starting memory monitoring...")
    monitor.start_monitoring()
    
    try:
        # Simulate some memory allocation
        for i in range(10):
            time.sleep(1)
            snapshot = monitor.add_snapshot(gaussian_count=i*1000)
            print(f"GPU: {snapshot.gpu_allocated_mb:.1f}MB, Gaussians: {snapshot.gaussian_count}")
            
        stats = monitor.get_stats()
        print(f"\nStats: {stats}")
        
        trends = monitor.get_trend()
        print(f"Trends: {trends}")
        
        action = monitor.suggest_action()
        print(f"Suggested action: {action}")
        
    finally:
        monitor.stop_monitoring()
        print("Monitoring stopped.")