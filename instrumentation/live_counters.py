# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Live performance counters for GaussianFeels with real-time FPS, Hz, Gaussians, VRAM, and RMSE tracking.
Implements high-frequency monitoring with minimal overhead.
"""

import time
import threading
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import torch


@dataclass
class PerformanceCounter:
    """Individual performance counter with history"""
    name: str
    value: float = 0.0
    target: float = 0.0
    unit: str = ""
    history: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_update: float = field(default_factory=time.time)
    
    def update(self, value: float):
        """Update counter value and history"""
        self.value = value
        self.last_update = time.time()
        self.history.append((self.last_update, value))
    
    def get_rate(self, window_seconds: float = 1.0) -> float:
        """Calculate rate of change over time window"""
        if len(self.history) < 2:
            return 0.0
            
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_values = [(t, v) for t, v in self.history if t >= cutoff_time]
        
        if len(recent_values) < 2:
            return 0.0
            
        time_span = recent_values[-1][0] - recent_values[0][0]
        value_change = recent_values[-1][1] - recent_values[0][1]
        
        return value_change / time_span if time_span > 0 else 0.0
    
    def get_average(self, window_seconds: float = 5.0) -> float:
        """Calculate average value over time window"""
        if not self.history:
            return 0.0
            
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_values = [v for t, v in self.history if t >= cutoff_time]
        
        return np.mean(recent_values) if recent_values else 0.0
    
    def is_healthy(self, tolerance: float = 0.1) -> bool:
        """Check if counter is within target tolerance"""
        if self.target == 0:
            return True
        return abs(self.value - self.target) / self.target <= tolerance


class LiveCounters:
    """Real-time performance monitoring with live counters"""
    
    def __init__(self, update_interval: float = 0.1):
        self.update_interval = update_interval
        
        # Performance counters
        self.counters = {
            'render_fps': PerformanceCounter('Render FPS', target=45.0, unit='fps'),
            'optimize_hz': PerformanceCounter('Optimize Hz', target=12.5, unit='hz'),
            'interpolate_hz': PerformanceCounter('Interpolate Hz', target=1000.0, unit='hz'),
            'gaussian_count': PerformanceCounter('Active Gaussians', target=300000, unit='count'),
            'vram_usage_mb': PerformanceCounter('VRAM Usage', target=158.0, unit='MB'),
            'rmse': PerformanceCounter('RMSE', target=0.01, unit=''),
            'system_memory_mb': PerformanceCounter('System Memory', target=2048.0, unit='MB'),
            'convergence_rate': PerformanceCounter('Convergence Rate', target=0.001, unit='/iter'),
        }
        
        # Thread management
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Frame timing for FPS calculation
        self.last_frame_times = {
            'render': deque(maxlen=100),
            'optimize': deque(maxlen=50),
            'interpolate': deque(maxlen=1000)
        }
        
        # GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        
    def mark_frame(self, thread_type: str):
        """Mark a frame completion for FPS/Hz calculation"""
        current_time = time.time()
        
        with self._lock:
            if thread_type in self.last_frame_times:
                self.last_frame_times[thread_type].append(current_time)
                
                # Calculate FPS/Hz
                times = list(self.last_frame_times[thread_type])
                if len(times) >= 2:
                    time_span = times[-1] - times[0]
                    frame_count = len(times) - 1
                    rate = frame_count / time_span if time_span > 0 else 0.0
                    
                    # Update appropriate counter
                    if thread_type == 'render':
                        self.counters['render_fps'].update(rate)
                    elif thread_type == 'optimize':
                        self.counters['optimize_hz'].update(rate)
                    elif thread_type == 'interpolate':
                        self.counters['interpolate_hz'].update(rate)
    
    def update_gaussian_count(self, count: int):
        """Update active Gaussian count"""
        self.counters['gaussian_count'].update(count)
    
    def update_rmse(self, rmse: float):
        """Update RMSE value"""
        self.counters['rmse'].update(rmse)
    
    def update_convergence_rate(self, rate: float):
        """Update convergence rate"""
        self.counters['convergence_rate'].update(rate)
    
    def _update_memory_counters(self):
        """Update memory usage counters"""
        if self.gpu_available:
            vram_bytes = torch.cuda.memory_allocated()
            vram_mb = vram_bytes / (1024**2)
            self.counters['vram_usage_mb'].update(vram_mb)
        
        # System memory
        try:
            import psutil
            system_memory = psutil.virtual_memory()
            system_mb = system_memory.used / (1024**2)
            self.counters['system_memory_mb'].update(system_mb)
        except ImportError:
            pass  # psutil not available
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.is_running:
            return
            
        self.is_running = True
        
        def monitor_loop():
            while self.is_running:
                try:
                    self._update_memory_counters()
                    time.sleep(self.update_interval)
                except Exception as e:
                    print(f"Live counter monitoring error: {e}")
                    
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current counter values"""
        with self._lock:
            return {name: counter.value for name, counter in self.counters.items()}
    
    def get_averages(self, window_seconds: float = 5.0) -> Dict[str, float]:
        """Get average values over time window"""
        with self._lock:
            return {name: counter.get_average(window_seconds) 
                   for name, counter in self.counters.items()}
    
    def get_health_status(self) -> Dict[str, bool]:
        """Get health status for all counters"""
        with self._lock:
            return {name: counter.is_healthy() for name, counter in self.counters.items()}
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        current = self.get_current_values()
        averages = self.get_averages()
        health = self.get_health_status()
        
        # Calculate overall health score
        healthy_count = sum(health.values())
        total_count = len(health)
        health_score = healthy_count / total_count if total_count > 0 else 0.0
        
        # Performance targets status
        targets_met = {
            'render_fps': current['render_fps'] >= 30.0,
            'optimize_hz': current['optimize_hz'] >= 10.0,
            'interpolate_hz': current['interpolate_hz'] >= 500.0,  # Allow some tolerance
            'gaussian_count': current['gaussian_count'] <= 350000,  # Allow some headroom
            'vram_usage_mb': current['vram_usage_mb'] <= 474.0,     # Budget limit
            'rmse': current['rmse'] <= 0.05,                       # Convergence target
        }
        
        targets_score = sum(targets_met.values()) / len(targets_met)
        
        return {
            'current_values': current,
            'averages_5s': averages,
            'health_status': health,
            'targets_met': targets_met,
            'health_score': health_score,
            'targets_score': targets_score,
            'overall_score': (health_score + targets_score) / 2.0,
            'timestamp': time.time()
        }
    
    def print_live_dashboard(self):
        """Print live performance dashboard to console"""
        summary = self.get_performance_summary()
        current = summary['current_values']
        targets = summary['targets_met']
        
        print("\n" + "="*60)
        print("           GAUSSIANFEELS LIVE PERFORMANCE")
        print("="*60)
        
        # Performance metrics
        print(f"🎯 Render FPS:     {current['render_fps']:6.1f} {'✅' if targets['render_fps'] else '❌'} (target: 30+ fps)")
        print(f"⚡ Optimize Hz:    {current['optimize_hz']:6.1f} {'✅' if targets['optimize_hz'] else '❌'} (target: 10+ hz)")
        print(f"🔄 Interpolate Hz: {current['interpolate_hz']:6.0f} {'✅' if targets['interpolate_hz'] else '❌'} (target: 500+ hz)")
        
        # Memory and resources
        print(f"🧠 VRAM Usage:     {current['vram_usage_mb']:6.1f} MB {'✅' if targets['vram_usage_mb'] else '❌'} (budget: 474 MB)")
        print(f"📊 System Memory:  {current['system_memory_mb']:6.0f} MB")
        print(f"🔵 Gaussians:      {current['gaussian_count']:6.0f} {'✅' if targets['gaussian_count'] else '❌'} (target: <350k)")
        
        # Quality metrics
        print(f"📈 RMSE:           {current['rmse']:6.4f} {'✅' if targets['rmse'] else '❌'} (target: <0.05)")
        print(f"📉 Convergence:    {current['convergence_rate']:6.4f}")
        
        # Overall status
        overall_score = summary['overall_score']
        if overall_score >= 0.8:
            status = "🟢 EXCELLENT"
        elif overall_score >= 0.6:
            status = "🟡 GOOD"
        elif overall_score >= 0.4:
            status = "🟠 FAIR"
        else:
            status = "🔴 POOR"
            
        print(f"\nOverall Performance: {status} ({overall_score:.1%})")
        print("="*60)
    
    def create_alert_handler(self, 
                           fps_threshold: float = 20.0,
                           vram_threshold: float = 400.0,
                           rmse_threshold: float = 0.1) -> Callable[[], None]:
        """Create alert handler for performance issues"""
        
        def check_alerts():
            current = self.get_current_values()
            alerts = []
            
            if current['render_fps'] < fps_threshold:
                alerts.append(f"Low FPS: {current['render_fps']:.1f}")
                
            if current['vram_usage_mb'] > vram_threshold:
                alerts.append(f"High VRAM: {current['vram_usage_mb']:.1f}MB")
                
            if current['rmse'] > rmse_threshold:
                alerts.append(f"High RMSE: {current['rmse']:.4f}")
                
            if alerts:
                print(f"⚠️  PERFORMANCE ALERTS: {'; '.join(alerts)}")
                
        return check_alerts


# Global instance for easy access
live_counters = LiveCounters()


def mark_render_frame():
    """Mark completion of a render frame"""
    live_counters.mark_frame('render')


def mark_optimize_frame():
    """Mark completion of an optimization step"""
    live_counters.mark_frame('optimize')


def mark_interpolate_frame():
    """Mark completion of an interpolation step"""
    live_counters.mark_frame('interpolate')


if __name__ == "__main__":
    # Example usage
    counters = LiveCounters()
    counters.start_monitoring()
    
    try:
        # Simulate some performance data
        for i in range(100):
            # Simulate render frames at ~45 FPS
            time.sleep(0.022)  # ~45 FPS
            counters.mark_frame('render')
            
            # Simulate optimization at ~12 Hz
            if i % 4 == 0:
                counters.mark_frame('optimize')
                
            # Simulate interpolation at high frequency
            for _ in range(20):
                counters.mark_frame('interpolate')
                
            # Update other metrics
            counters.update_gaussian_count(250000 + i * 100)
            counters.update_rmse(0.01 + 0.001 * np.sin(i * 0.1))
            
            # Print dashboard every 20 iterations
            if i % 20 == 0:
                counters.print_live_dashboard()
                
    finally:
        counters.stop_monitoring()