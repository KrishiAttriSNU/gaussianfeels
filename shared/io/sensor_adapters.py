#!/usr/bin/env python3
"""
Sensor Integration Layer for GaussianFeels
===========================================

Complete sensor adapter system for robust multi-modal perception:
- RealSense D435/D455 camera integration
- DIGIT tactile sensor integration  
- Multi-camera synchronization
- Sensor calibration and pose estimation
- Real-time data streaming

Designed for high-performance tactile-visual fusion.
"""

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import cv2

# All imports required
import pyrealsense2 as rs
import rospy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from geometry_msgs.msg import PoseStamped

@dataclass
class SensorConfig:
    """Configuration for sensor setup"""
    # Camera settings
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_depth: bool = True
    enable_color: bool = True
    depth_format: str = "Z16"
    color_format: str = "RGB8"
    
    # Calibration
    intrinsic_matrix: Optional[np.ndarray] = None
    distortion_coeffs: Optional[np.ndarray] = None
    extrinsic_matrix: Optional[np.ndarray] = None
    
    # Sensor-specific settings
    exposure_time: Optional[int] = None
    gain: Optional[int] = None
    laser_power: Optional[float] = None
    
    # Synchronization
    sync_mode: str = "hardware"  # "hardware", "software", "none"
    master_sensor: Optional[str] = None

@dataclass 
class SensorFrame:
    """Unified sensor data frame"""
    timestamp: float
    sensor_id: str
    sensor_type: str
    
    # Visual data
    color_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    infrared_image: Optional[np.ndarray] = None
    
    # Tactile data
    tactile_image: Optional[np.ndarray] = None
    tactile_depth: Optional[np.ndarray] = None
    contact_mask: Optional[np.ndarray] = None
    
    # Metadata
    pose: Optional[np.ndarray] = None
    intrinsics: Optional[Dict[str, float]] = None
    frame_id: Optional[int] = None
    confidence: Optional[np.ndarray] = None

class BaseSensorAdapter(ABC):
    """Abstract base class for all sensor adapters"""
    
    def __init__(self, sensor_id: str, config: SensorConfig):
        self.sensor_id = sensor_id
        self.config = config
        self.is_streaming = False
        self.frame_count = 0
        self.last_frame: Optional[SensorFrame] = None
        
        # Threading for async operation
        self.stream_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Frame buffer for synchronization
        self.frame_buffer: List[SensorFrame] = []
        self.buffer_size = 10
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the sensor hardware"""
        raise NotImplementedError
    
    @abstractmethod
    def start_streaming(self) -> bool:
        """Start sensor data streaming"""
        raise NotImplementedError
    
    @abstractmethod
    def stop_streaming(self):
        """Stop sensor streaming"""
        raise NotImplementedError
    
    @abstractmethod
    def get_frame(self) -> Optional[SensorFrame]:
        """Get latest sensor frame"""
        raise NotImplementedError
    
    @abstractmethod
    def cleanup(self):
        """Cleanup sensor resources"""
        raise NotImplementedError
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """Get sensor information and status"""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.__class__.__name__,
            'is_streaming': self.is_streaming,
            'frame_count': self.frame_count,
            'config': self.config,
            'last_timestamp': self.last_frame.timestamp if self.last_frame else None
        }

class RealSenseAdapter(BaseSensorAdapter):
    """RealSense camera adapter for robust multi-camera perception"""
    
    def __init__(self, sensor_id: str, config: SensorConfig, device_serial: Optional[str] = None):
        super().__init__(sensor_id, config)
        self.device_serial = device_serial
        self.pipeline = None
        self.profile = None
        self.align_to_color = None
        
        # Dependencies must be installed
    
    def initialize(self) -> bool:
        """Initialize RealSense camera"""
        try:
            print(f"🎥 Initializing RealSense camera {self.sensor_id}...")
            
            # Create pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure specific device if serial provided
            if self.device_serial:
                config.enable_device(self.device_serial)
            
            # Configure streams
            if self.config.enable_color:
                config.enable_stream(
                    rs.stream.color, 
                    self.config.width, 
                    self.config.height, 
                    rs.format.bgr8 if self.config.color_format == "BGR8" else rs.format.rgb8,
                    self.config.fps
                )
            
            if self.config.enable_depth:
                config.enable_stream(
                    rs.stream.depth, 
                    self.config.width, 
                    self.config.height, 
                    rs.format.z16, 
                    self.config.fps
                )
            
            # Start pipeline
            self.profile = self.pipeline.start(config)
            
            # Setup alignment
            if self.config.enable_color and self.config.enable_depth:
                self.align_to_color = rs.align(rs.stream.color)
            
            # Configure sensor settings
            self._configure_sensor_settings()
            
            # Get intrinsics
            if self.config.enable_color:
                color_stream = self.profile.get_stream(rs.stream.color)
                color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                self.config.intrinsic_matrix = np.array([
                    [color_intrinsics.fx, 0, color_intrinsics.ppx],
                    [0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0, 0, 1]
                ])
                self.config.distortion_coeffs = np.array(color_intrinsics.coeffs)
            
            print(f"✅ RealSense camera {self.sensor_id} initialized successfully")
            return True
            
        except (RuntimeError, ImportError, AttributeError) as e:
            print(f"❌ Failed to initialize RealSense camera {self.sensor_id}: {e}")
            return False
    
    def _configure_sensor_settings(self):
        """Configure advanced sensor settings"""
        try:
            # Get sensor from pipeline
            device = self.profile.get_device()
            
            # Configure depth sensor
            if self.config.enable_depth:
                depth_sensor = device.first_depth_sensor()
                
                # Laser power
                if self.config.laser_power is not None and depth_sensor.supports(rs.option.laser_power):
                    depth_sensor.set_option(rs.option.laser_power, self.config.laser_power)
                
                # Set high accuracy preset
                if depth_sensor.supports(rs.option.visual_preset):
                    depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
            
            # Configure color sensor
            if self.config.enable_color:
                color_sensor = device.first_color_sensor()
                
                # Exposure
                if self.config.exposure_time is not None and color_sensor.supports(rs.option.exposure):
                    color_sensor.set_option(rs.option.auto_exposure_enabled, 0)  # Disable auto exposure
                    color_sensor.set_option(rs.option.exposure, self.config.exposure_time)
                
                # Gain
                if self.config.gain is not None and color_sensor.supports(rs.option.gain):
                    color_sensor.set_option(rs.option.gain, self.config.gain)
        
        except (RuntimeError, ValueError, AttributeError) as e:
            raise RuntimeError(f"Failed to configure sensor settings: {e}") from e
    
    def start_streaming(self) -> bool:
        """Start RealSense streaming"""
        if self.is_streaming:
            return True
            
        try:
            self.is_streaming = True
            self.stop_event.clear()
            
            # Start streaming thread
            self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.stream_thread.start()
            
            print(f"🎬 RealSense camera {self.sensor_id} streaming started")
            return True
            
        except (RuntimeError, ValueError, AttributeError) as e:
            self.is_streaming = False
            raise RuntimeError(f"Failed to start RealSense streaming: {e}") from e
    
    def _stream_loop(self):
        """Main streaming loop"""
        while not self.stop_event.is_set() and self.is_streaming:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                
                # Align frames if both color and depth are enabled
                if self.align_to_color:
                    frames = self.align_to_color.process(frames)
                
                # Create sensor frame
                sensor_frame = self._process_frames(frames)
                if sensor_frame:
                    self.last_frame = sensor_frame
                    self._add_to_buffer(sensor_frame)
                    self.frame_count += 1
                
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"Warning: RealSense frame capture failed: {e}")
                time.sleep(0.01)  # Brief pause on error
    
    def _process_frames(self, frames) -> Optional[SensorFrame]:
        """Process RealSense frames into SensorFrame format"""
        try:
            timestamp = time.time()
            
            color_image = None
            depth_image = None
            infrared_image = None
            confidence = None
            
            # Process color frame
            if self.config.enable_color and frames.get_color_frame():
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                if self.config.color_format == "RGB8":
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Process depth frame
            if self.config.enable_depth and frames.get_depth_frame():
                depth_frame = frames.get_depth_frame()
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Apply depth filters if available
                depth_image = self._apply_depth_filters(depth_image)
            
            # Get infrared if available
            if frames.get_infrared_frame():
                ir_frame = frames.get_infrared_frame()
                infrared_image = np.asanyarray(ir_frame.get_data())
            
            # Create intrinsics dictionary
            intrinsics = None
            if self.config.intrinsic_matrix is not None:
                intrinsics = {
                    'fx': float(self.config.intrinsic_matrix[0, 0]),
                    'fy': float(self.config.intrinsic_matrix[1, 1]),
                    'cx': float(self.config.intrinsic_matrix[0, 2]),
                    'cy': float(self.config.intrinsic_matrix[1, 2]),
                    'width': self.config.width,
                    'height': self.config.height
                }
            
            return SensorFrame(
                timestamp=timestamp,
                sensor_id=self.sensor_id,
                sensor_type="RealSense",
                color_image=color_image,
                depth_image=depth_image,
                infrared_image=infrared_image,
                intrinsics=intrinsics,
                frame_id=self.frame_count,
                pose=self.config.extrinsic_matrix
            )
            
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            raise RuntimeError(f"RealSense frame processing failed: {e}. Check sensor connection and frame data.") from e
    
    def _apply_depth_filters(self, depth_image: np.ndarray) -> np.ndarray:
        """Apply depth filtering to improve quality"""
        try:
            # Simple bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(
                depth_image.astype(np.float32), 
                d=5, 
                sigmaColor=50, 
                sigmaSpace=50
            ).astype(np.uint16)
            
            return filtered
        except:
            return depth_image
    
    def get_frame(self) -> Optional[SensorFrame]:
        """Get latest RealSense frame"""
        return self.last_frame
    
    def stop_streaming(self):
        """Stop RealSense streaming"""
        if not self.is_streaming:
            return
            
        print(f"🛑 Stopping RealSense camera {self.sensor_id}...")
        self.is_streaming = False
        self.stop_event.set()
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
    
    def cleanup(self):
        """Cleanup RealSense resources"""
        self.stop_streaming()
        
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
        
        print(f"🧹 RealSense camera {self.sensor_id} cleaned up")
    
    def _add_to_buffer(self, frame: SensorFrame):
        """Add frame to buffer with size limit"""
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

class DIGITAdapter(BaseSensorAdapter):
    """DIGIT tactile sensor adapter for high-resolution touch sensing"""
    
    def __init__(self, sensor_id: str, config: SensorConfig, device_path: str = "/dev/video0"):
        super().__init__(sensor_id, config)
        self.device_path = device_path
        self.cap = None
        self.tactile_predictor = None
        
        # DIGIT-specific settings
        self.gel_width = 15.0  # mm
        self.gel_height = 15.0  # mm
        self.background_image = None
        
    def initialize(self) -> bool:
        """Initialize DIGIT tactile sensor"""
        try:
            print(f"🤏 Initializing DIGIT tactile sensor {self.sensor_id}...")
            
            # Initialize camera capture
            self.cap = cv2.VideoCapture(self.device_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open DIGIT camera at {self.device_path}")
            
            # Configure camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Initialize tactile predictor
            self._initialize_tactile_predictor()
            
            # Capture background image
            self._capture_background()
            
            print(f"✅ DIGIT tactile sensor {self.sensor_id} initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize DIGIT sensor {self.sensor_id}: {e}")
            return False
    
    def _initialize_tactile_predictor(self):
        """Initialize tactile depth prediction model"""
        try:
            # Import tactile predictor
            from .tactile_predictors import TactilePredictorFactory
            
            self.tactile_predictor = TactilePredictorFactory.create_predictor(
                mode="vit",  # Use Vision Transformer by default
                config=self.config
            )
            
        except (RuntimeError, ValueError, ImportError, AttributeError) as e:
            raise RuntimeError(f"Tactile predictor initialization failed: {e}. Check TouchVIT model and dependencies.") from e
    
    def _capture_background(self):
        """Capture background image for contact detection"""
        try:
            print(f"📷 Capturing background image for {self.sensor_id}...")
            
            # Capture several frames and average them
            background_frames = []
            for _ in range(10):
                ret, frame = self.cap.read()
                if ret:
                    background_frames.append(frame.astype(np.float32))
                time.sleep(0.1)
            
            if background_frames:
                self.background_image = np.mean(background_frames, axis=0).astype(np.uint8)
                print(f"✅ Background captured for {self.sensor_id}")
            else:
                print(f"⚠️  Failed to capture background for {self.sensor_id}")
                
        except Exception as e:
            print(f"Warning: Background capture failed: {e}")
    
    def start_streaming(self) -> bool:
        """Start DIGIT streaming"""
        if self.is_streaming:
            return True
            
        try:
            self.is_streaming = True
            self.stop_event.clear()
            
            # Start streaming thread
            self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.stream_thread.start()
            
            print(f"🎬 DIGIT sensor {self.sensor_id} streaming started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start DIGIT streaming: {e}")
            self.is_streaming = False
            return False
    
    def _stream_loop(self):
        """Main DIGIT streaming loop"""
        while not self.stop_event.is_set() and self.is_streaming:
            try:
                ret, frame = self.cap.read()
                if ret:
                    sensor_frame = self._process_tactile_frame(frame)
                    if sensor_frame:
                        self.last_frame = sensor_frame
                        self._add_to_buffer(sensor_frame)
                        self.frame_count += 1
                
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"Warning: DIGIT frame capture failed: {e}")
                time.sleep(0.01)
    
    def _process_tactile_frame(self, frame: np.ndarray) -> Optional[SensorFrame]:
        """Process DIGIT tactile frame"""
        try:
            timestamp = time.time()
            
            # Basic preprocessing
            tactile_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Contact detection
            contact_mask = self._detect_contact(frame)
            
            # Tactile depth prediction
            tactile_depth = None
            if self.tactile_predictor and contact_mask is not None:
                try:
                    tactile_depth = self.tactile_predictor.predict_depth(tactile_image, contact_mask)
                except (RuntimeError, ValueError, TypeError) as e:
                    raise RuntimeError(f"Tactile depth prediction failed for {self.sensor_id}: {e}. Check TouchVIT model and input data.") from e
            
            return SensorFrame(
                timestamp=timestamp,
                sensor_id=self.sensor_id,
                sensor_type="DIGIT",
                tactile_image=tactile_image,
                tactile_depth=tactile_depth,
                contact_mask=contact_mask,
                frame_id=self.frame_count,
                pose=self.config.extrinsic_matrix
            )
            
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            raise RuntimeError(f"DIGIT frame processing failed for {self.sensor_id}: {e}. Check sensor connection and data format.") from e
    
    def _detect_contact(self, current_frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect contact regions by comparing with background"""
        if self.background_image is None:
            return None
            
        try:
            # Compute difference from background
            diff = cv2.absdiff(current_frame, self.background_image)
            
            # Convert to grayscale
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, contact_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            contact_mask = cv2.morphologyEx(contact_mask, cv2.MORPH_CLOSE, kernel)
            contact_mask = cv2.morphologyEx(contact_mask, cv2.MORPH_OPEN, kernel)
            
            return contact_mask
            
        except (RuntimeError, ValueError, cv2.error) as e:
            raise RuntimeError(f"Contact detection failed for {self.sensor_id}: {e}. Check background image and tactile image quality.") from e
    
    def get_frame(self) -> Optional[SensorFrame]:
        """Get latest DIGIT frame"""
        return self.last_frame
    
    def stop_streaming(self):
        """Stop DIGIT streaming"""
        if not self.is_streaming:
            return
            
        print(f"🛑 Stopping DIGIT sensor {self.sensor_id}...")
        self.is_streaming = False
        self.stop_event.set()
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
    
    def cleanup(self):
        """Cleanup DIGIT resources"""
        self.stop_streaming()
        
        if self.cap:
            self.cap.release()
        
        print(f"🧹 DIGIT sensor {self.sensor_id} cleaned up")
    
    def _add_to_buffer(self, frame: SensorFrame):
        """Add frame to buffer with size limit"""
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

class MultiSensorManager:
    """Multi-sensor synchronization and management for tactile-visual fusion"""
    
    def __init__(self, sync_mode: str = "software"):
        self.sensors: Dict[str, BaseSensorAdapter] = {}
        self.sync_mode = sync_mode
        self.master_sensor: Optional[str] = None
        self.frame_sync_buffer: Dict[str, List[SensorFrame]] = {}
        self.sync_tolerance = 0.05  # 50ms sync tolerance
        
        # Synchronization state
        self.sync_thread: Optional[threading.Thread] = None
        self.sync_running = False
        self.synchronized_frames: List[Dict[str, SensorFrame]] = []
        self.max_sync_frames = 100
    
    def add_sensor(self, adapter: BaseSensorAdapter):
        """Add sensor adapter to manager"""
        self.sensors[adapter.sensor_id] = adapter
        self.frame_sync_buffer[adapter.sensor_id] = []
        print(f"📡 Added sensor {adapter.sensor_id} to multi-sensor manager")
    
    def remove_sensor(self, sensor_id: str):
        """Remove sensor from manager"""
        if sensor_id in self.sensors:
            self.sensors[sensor_id].cleanup()
            del self.sensors[sensor_id]
            del self.frame_sync_buffer[sensor_id]
            print(f"🗑️  Removed sensor {sensor_id} from manager")
    
    def initialize_all(self) -> bool:
        """Initialize all sensors"""
        print(f"🚀 Initializing {len(self.sensors)} sensors...")
        
        success_count = 0
        for sensor_id, adapter in self.sensors.items():
            if adapter.initialize():
                success_count += 1
            else:
                print(f"❌ Failed to initialize sensor {sensor_id}")
        
        if success_count == len(self.sensors):
            print(f"✅ All {success_count} sensors initialized successfully")
            return True
        else:
            print(f"⚠️  {success_count}/{len(self.sensors)} sensors initialized")
            return False
    
    def start_streaming_all(self) -> bool:
        """Start streaming on all sensors"""
        print(f"🎬 Starting streaming on {len(self.sensors)} sensors...")
        
        success_count = 0
        for sensor_id, adapter in self.sensors.items():
            if adapter.start_streaming():
                success_count += 1
            else:
                print(f"❌ Failed to start streaming on sensor {sensor_id}")
        
        # Start synchronization thread
        if success_count > 0:
            self._start_synchronization()
        
        if success_count == len(self.sensors):
            print(f"✅ All {success_count} sensors streaming")
            return True
        else:
            print(f"⚠️  {success_count}/{len(self.sensors)} sensors streaming")
            return False
    
    def _start_synchronization(self):
        """Start frame synchronization thread"""
        if self.sync_mode == "none" or len(self.sensors) <= 1:
            return
            
        self.sync_running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        print(f"🔄 Frame synchronization started ({self.sync_mode} mode)")
    
    def _sync_loop(self):
        """Main synchronization loop"""
        while self.sync_running:
            try:
                # Collect frames from all sensors
                current_frames = {}
                for sensor_id, adapter in self.sensors.items():
                    frame = adapter.get_frame()
                    if frame:
                        current_frames[sensor_id] = frame
                        self.frame_sync_buffer[sensor_id].append(frame)
                        
                        # Limit buffer size
                        if len(self.frame_sync_buffer[sensor_id]) > 50:
                            self.frame_sync_buffer[sensor_id].pop(0)
                
                # Synchronize frames
                if len(current_frames) > 1:
                    synced_frame_set = self._synchronize_frames()
                    if synced_frame_set:
                        self.synchronized_frames.append(synced_frame_set)
                        
                        # Limit synchronized frame buffer
                        if len(self.synchronized_frames) > self.max_sync_frames:
                            self.synchronized_frames.pop(0)
                
                time.sleep(0.01)  # 100Hz sync rate
                
            except (RuntimeError, ValueError, KeyError) as e:
                raise RuntimeError(f"Frame synchronization failed: {e}. Check sensor connections and timing.") from e
    
    def _synchronize_frames(self) -> Optional[Dict[str, SensorFrame]]:
        """Synchronize frames from multiple sensors"""
        if self.sync_mode == "software":
            return self._software_sync()
        elif self.sync_mode == "hardware":
            return self._hardware_sync()
        else:
            return None
    
    def _software_sync(self) -> Optional[Dict[str, SensorFrame]]:
        """Software-based frame synchronization"""
        try:
            # Find reference timestamp
            ref_timestamp = None
            ref_sensor = self.master_sensor or next(iter(self.sensors.keys()))
            
            if ref_sensor in self.frame_sync_buffer and self.frame_sync_buffer[ref_sensor]:
                ref_frame = self.frame_sync_buffer[ref_sensor][-1]
                ref_timestamp = ref_frame.timestamp
            
            if ref_timestamp is None:
                return None
            
            # Find closest frames from other sensors
            synced_frames = {ref_sensor: self.frame_sync_buffer[ref_sensor][-1]}
            
            for sensor_id, frame_buffer in self.frame_sync_buffer.items():
                if sensor_id == ref_sensor or not frame_buffer:
                    continue
                
                # Find frame closest to reference timestamp
                best_frame = None
                best_diff = float('inf')
                
                for frame in frame_buffer:
                    time_diff = abs(frame.timestamp - ref_timestamp)
                    if time_diff < best_diff:
                        best_diff = time_diff
                        best_frame = frame
                
                # Only include if within sync tolerance
                if best_frame and best_diff <= self.sync_tolerance:
                    synced_frames[sensor_id] = best_frame
            
            # Only return if we have frames from multiple sensors
            return synced_frames if len(synced_frames) > 1 else None
            
        except (RuntimeError, ValueError, KeyError) as e:
            raise RuntimeError(f"Software frame synchronization failed: {e}. Check sensor timing and frame buffers.") from e
    
    def _hardware_sync(self) -> Optional[Dict[str, SensorFrame]]:
        """Hardware-based frame synchronization"""
        # For hardware sync, assume frames are already synchronized
        # This would require hardware trigger setup
        current_frames = {}
        for sensor_id, adapter in self.sensors.items():
            frame = adapter.get_frame()
            if frame:
                current_frames[sensor_id] = frame
        
        return current_frames if len(current_frames) > 1 else None
    
    def get_synchronized_frame_set(self) -> Optional[Dict[str, SensorFrame]]:
        """Get latest synchronized frame set"""
        return self.synchronized_frames[-1] if self.synchronized_frames else None
    
    def get_sensor_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sensors"""
        status = {}
        for sensor_id, adapter in self.sensors.items():
            status[sensor_id] = adapter.get_sensor_info()
        return status
    
    def stop_streaming_all(self):
        """Stop streaming on all sensors"""
        print(f"🛑 Stopping streaming on {len(self.sensors)} sensors...")
        
        # Stop synchronization
        self.sync_running = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=2.0)
        
        # Stop all sensors
        for adapter in self.sensors.values():
            adapter.stop_streaming()
    
    def cleanup_all(self):
        """Cleanup all sensors"""
        print(f"🧹 Cleaning up {len(self.sensors)} sensors...")
        
        self.stop_streaming_all()
        
        for adapter in self.sensors.values():
            adapter.cleanup()
        
        self.sensors.clear()
        self.frame_sync_buffer.clear()

# Factory class for creating sensor adapters
class SensorFactory:
    """Factory for creating sensor adapters"""
    
    @staticmethod
    def create_realsense_adapter(sensor_id: str, 
                               width: int = 640, 
                               height: int = 480,
                               fps: int = 30,
                               device_serial: Optional[str] = None) -> RealSenseAdapter:
        """Create RealSense adapter with standard configuration"""
        config = SensorConfig(
            width=width,
            height=height,
            fps=fps,
            enable_depth=True,
            enable_color=True
        )
        return RealSenseAdapter(sensor_id, config, device_serial)
    
    @staticmethod
    def create_digit_adapter(sensor_id: str,
                           device_path: str = "/dev/video0",
                           width: int = 320,
                           height: int = 240,
                           fps: int = 60) -> DIGITAdapter:
        """Create DIGIT adapter with standard configuration"""
        config = SensorConfig(
            width=width,
            height=height, 
            fps=fps,
            enable_color=True,
            enable_depth=False
        )
        return DIGITAdapter(sensor_id, config, device_path)
    
    @staticmethod
    def create_standard_setup() -> MultiSensorManager:
        """Create standard multi-camera tactile-visual sensor setup"""
        manager = MultiSensorManager(sync_mode="software")
        
        # Add RealSense cameras (standard 3-camera setup)
        try:
            realsense1 = SensorFactory.create_realsense_adapter("realsense_1")
            realsense2 = SensorFactory.create_realsense_adapter("realsense_2") 
            manager.add_sensor(realsense1)
            manager.add_sensor(realsense2)
        except (RuntimeError, ValueError, OSError, ImportError) as e:
            raise RuntimeError(f"RealSense camera creation failed: {e}. Check hardware connections and drivers.") from e
        
        # Add DIGIT sensors
        try:
            digit_left = SensorFactory.create_digit_adapter("digit_left", "/dev/video0")
            digit_right = SensorFactory.create_digit_adapter("digit_right", "/dev/video1")
            manager.add_sensor(digit_left)
            manager.add_sensor(digit_right)
        except (RuntimeError, ValueError, OSError, ImportError) as e:
            raise RuntimeError(f"DIGIT sensor creation failed: {e}. Check hardware connections and device paths.") from e
        
        return manager

if __name__ == "__main__":
    # Example usage
    print("🧪 Testing sensor integration...")
    
    # Create sensor manager
    manager = SensorFactory.create_standard_setup()
    
    try:
        # Initialize and start streaming
        if manager.initialize_all():
            if manager.start_streaming_all():
                print("✅ All sensors streaming successfully!")
                
                # Test synchronization for 5 seconds
                start_time = time.time()
                while time.time() - start_time < 5.0:
                    synced_frames = manager.get_synchronized_frame_set()
                    if synced_frames:
                        print(f"📊 Synchronized frame set: {list(synced_frames.keys())}")
                    time.sleep(0.5)
            
    finally:
        manager.cleanup_all()
        print("🏁 Test completed")