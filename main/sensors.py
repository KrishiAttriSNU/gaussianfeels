"""
GaussianFeels Live Sensor Integration

Real-time sensor data acquisition and processing for live Gaussian splatting.
Supports RealSense cameras, tactile sensors, and robot interfaces.
"""

import time
import threading
import queue
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import cv2
from pathlib import Path

# Optional dependencies
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

try:
    import rospy
    from sensor_msgs.msg import Image, PointCloud2, CompressedImage
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from .datasets import FrameData

@dataclass
class SensorConfig:
    """Configuration for sensor setup"""
    name: str
    sensor_type: str  # "realsense", "tactile", "robot"
    enabled: bool = True
    fps: int = 30
    resolution: Tuple[int, int] = (640, 480)
    device_id: Optional[str] = None
    calibration_file: Optional[Path] = None
    transform: np.ndarray = None  # 4x4 transform matrix
    
    def __post_init__(self):
        if self.transform is None:
            self.transform = np.eye(4)

@dataclass
class SensorData:
    """Container for sensor data"""
    timestamp: float
    sensor_name: str
    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    tactile_image: Optional[np.ndarray] = None
    tactile_depth: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseSensor(ABC):
    """Abstract base class for sensors"""
    
    def __init__(self, config: SensorConfig):
        self.config = config
        self.running = False
        self.data_queue = queue.Queue(maxsize=100)
        self.callbacks: List[Callable[[SensorData], None]] = []
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the sensor"""
        raise NotImplementedError("Subclasses must implement initialize()")
    
    @abstractmethod
    def capture_frame(self) -> Optional[SensorData]:
        """Capture a single frame"""
        raise NotImplementedError("Subclasses must implement capture_frame()")
    
    @abstractmethod
    def cleanup(self):
        """Cleanup sensor resources"""
        raise NotImplementedError("Subclasses must implement cleanup()")
    
    def add_callback(self, callback: Callable[[SensorData], None]):
        """Add data callback"""
        self.callbacks.append(callback)
    
    def start_streaming(self):
        """Start continuous data streaming"""
        if self.running:
            return
        
        if not self.initialize():
            raise RuntimeError(f"Failed to initialize sensor {self.config.name}")
        
        self.running = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        
        print(f"✅ Started streaming {self.config.name}")
    
    def stop_streaming(self):
        """Stop data streaming"""
        self.running = False
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join(timeout=1.0)
        self.cleanup()
        print(f"🛑 Stopped streaming {self.config.name}")
    
    def _stream_loop(self):
        """Main streaming loop"""
        target_dt = 1.0 / self.config.fps
        
        while self.running:
            start_time = time.time()
            
            try:
                data = self.capture_frame()
                if data is not None:
                    # Add to queue
                    try:
                        self.data_queue.put_nowait(data)
                    except queue.Full:
                        # Remove oldest data if queue is full
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(data)
                        except queue.Empty:
                            # Queue was already empty, ignore
                            continue
                    
                    # Call callbacks
                    for callback in self.callbacks:
                        try:
                            callback(data)
                        except Exception as e:
                            print(f"Warning: Callback failed for {self.config.name}: {e}")
            
            except Exception as e:
                print(f"Error in {self.config.name} streaming: {e}")
                time.sleep(0.1)
                continue
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_latest_data(self) -> Optional[SensorData]:
        """Get the most recent sensor data"""
        latest = None
        while not self.data_queue.empty():
            try:
                latest = self.data_queue.get_nowait()
            except queue.Empty:
                break
        return latest

class RealSenseSensor(BaseSensor):
    """Intel RealSense RGB-D camera sensor"""
    
    def __init__(self, config: SensorConfig):
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 not available. Install with: pip install pyrealsense2")
        
        super().__init__(config)
        self.pipeline = None
        self.align = None
        
    def initialize(self) -> bool:
        """Initialize RealSense camera"""
        try:
            # Create pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure streams
            w, h = self.config.resolution
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, self.config.fps)
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, self.config.fps)
            
            # Enable specific device if specified
            if self.config.device_id:
                config.enable_device(self.config.device_id)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            
            # Create alignment object
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            # Get camera intrinsics
            color_profile = profile.get_stream(rs.stream.color)
            intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            self.intrinsics = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ])
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize RealSense {self.config.name}: {e}")
            return False
    
    def capture_frame(self) -> Optional[SensorData]:
        """Capture RGB-D frame from RealSense"""
        if not self.pipeline:
            return None
        
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align frames
            aligned_frames = self.align.process(frames)
            
            # Get color and depth frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None
            
            # Convert to numpy arrays
            rgb_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            # Convert depth to meters
            depth_scale = frames.get_profile().get_device().first_depth_sensor().get_depth_scale()
            depth_image = depth_image.astype(np.float32) * depth_scale
            
            return SensorData(
                timestamp=time.time(),
                sensor_name=self.config.name,
                rgb_image=rgb_image,
                depth_image=depth_image,
                intrinsics=self.intrinsics,
                pose=self.config.transform,
                metadata={'depth_scale': depth_scale}
            )
            
        except Exception as e:
            print(f"Failed to capture from {self.config.name}: {e}")
            return None
    
    def cleanup(self):
        """Cleanup RealSense resources"""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

class TactileSensor(BaseSensor):
    """Generic tactile sensor interface"""
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.sensor_client = None
        
    def initialize(self) -> bool:
        """Initialize tactile sensor"""
        try:
            # Check if we have a custom sensor client provided
            if hasattr(self, '_custom_sensor_client'):
                self.sensor_client = self._custom_sensor_client
                return True
            
            # Try to initialize based on sensor type from config
            sensor_type = self.config.sensor_type.lower()
            
            if sensor_type == "digit":
                self.sensor_client = self._init_digit_sensor()
            elif sensor_type == "gelsight":
                self.sensor_client = self._init_gelsight_sensor()
            elif sensor_type == "tacto":
                self.sensor_client = self._init_tacto_sensor()
            elif sensor_type == "custom":
                # Allow custom sensor implementation via external registration
                if hasattr(self, '_custom_sensor_client') and self._custom_sensor_client is not None:
                    self.sensor_client = self._custom_sensor_client
                else:
                    raise RuntimeError(
                        f"Custom sensor type specified for {self.config.name} but no implementation provided. "
                        "Set _custom_sensor_client before initialization."
                    )
            else:
                # Production-grade error with supported types and diagnostic information
                available_types = ["digit", "gelsight", "tacto", "custom"]
                raise RuntimeError(
                    f"Unsupported tactile sensor type '{sensor_type}' for {self.config.name}. "
                    f"Supported types: {available_types}. "
                    f"For custom implementations, use sensor_type='custom' and set _custom_sensor_client. "
                    f"Current config: {self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)}"
                )
            
            return self.sensor_client is not None
            
        except Exception as e:
            print(f"Failed to initialize tactile sensor {self.config.name}: {e}")
            return False
    
    def _init_digit_sensor(self):
        """Initialize DIGIT tactile sensor"""
        try:
            # Attempt to import digit-sensor package
            import digit_interface
            client = digit_interface.DigitSensorClient(
                sensor_name=self.config.name,
                resolution=self.config.resolution,
                fps=self.config.fps
            )
            if self.config.device_id:
                client.connect(self.config.device_id)
            else:
                client.connect()  # Auto-detect
            return client
        except ImportError:
            print("DIGIT sensor support not available. Install with: pip install digit-interface")
            return None
        except Exception as e:
            print(f"Failed to initialize DIGIT sensor: {e}")
            return None
    
    def _init_gelsight_sensor(self):
        """Initialize GelSight tactile sensor"""
        try:
            # Attempt to import gelsight package
            import gelsight
            client = gelsight.GelSightClient(
                device_id=self.config.device_id,
                fps=self.config.fps
            )
            client.connect()
            return client
        except ImportError:
            print("GelSight sensor support not available. Install gelsight package.")
            return None
        except Exception as e:
            print(f"Failed to initialize GelSight sensor: {e}")
            return None
    
    def _init_tacto_sensor(self):
        """Initialize TACTO simulator"""
        try:
            # Attempt to import tacto simulator
            import tacto
            client = tacto.Sensor(
                width=self.config.resolution[0],
                height=self.config.resolution[1],
                config_path=self.config.calibration_file
            )
            return client
        except ImportError:
            print("TACTO simulator not available. Install with: pip install tacto")
            return None
        except Exception as e:
            print(f"Failed to initialize TACTO simulator: {e}")
            return None
    
    # Mock sensor implementation removed for strict fail-fast operation
    
    def capture_frame(self) -> Optional[SensorData]:
        """Capture tactile frame"""
        if not self.sensor_client:
            return None
        
        try:
            tactile_data = self.sensor_client.get_tactile_frame()
            
            return SensorData(
                timestamp=time.time(),
                sensor_name=self.config.name,
                tactile_image=tactile_data['image'],
                tactile_depth=tactile_data['depth'],
                pose=self.config.transform,
                metadata=tactile_data.get('metadata', {})
            )
            
        except Exception as e:
            print(f"Failed to capture from tactile sensor {self.config.name}: {e}")
            return None
    
    def cleanup(self):
        """Cleanup tactile sensor"""
        if self.sensor_client and hasattr(self.sensor_client, 'disconnect'):
            self.sensor_client.disconnect()

    def disconnect(self):
        """Disconnect tactile sensor client if supported"""
        if self.sensor_client and hasattr(self.sensor_client, 'disconnect'):
            try:
                self.sensor_client.disconnect()
            except Exception as e:
                print(f"Warning: Failed to disconnect sensor client: {e}")

class ROSSensor(BaseSensor):
    """ROS-based sensor interface"""
    
    def __init__(self, config: SensorConfig):
        if not ROS_AVAILABLE:
            raise ImportError("ROS not available. Install ROS and rospy")
        
        super().__init__(config)
        self.bridge = CvBridge()
        self.subscribers = {}
        self.latest_pose = None
        
    def initialize(self) -> bool:
        """Initialize ROS sensor"""
        try:
            # Initialize ROS node if not already initialized
            if not rospy.get_node_uri():
                rospy.init_node(f'gaussianfeels_sensor_{self.config.name}', anonymous=True)
            
            # Setup subscribers based on sensor type
            if self.config.sensor_type == "realsense_ros":
                self._setup_realsense_subscribers()
            elif self.config.sensor_type == "tactile_ros":
                self._setup_tactile_subscribers()
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize ROS sensor {self.config.name}: {e}")
            return False
    
    def _setup_realsense_subscribers(self):
        """Setup RealSense ROS subscribers"""
        base_topic = f"/{self.config.name}"
        
        # RGB image
        rgb_topic = f"{base_topic}/color/image_raw"
        self.subscribers['rgb'] = rospy.Subscriber(
            rgb_topic, Image, self._rgb_callback, queue_size=1
        )
        
        # Depth image
        depth_topic = f"{base_topic}/depth/image_rect_raw"
        self.subscribers['depth'] = rospy.Subscriber(
            depth_topic, Image, self._depth_callback, queue_size=1
        )
        
        # Camera info would be subscribed to get intrinsics
        
    def _setup_tactile_subscribers(self):
        """Setup tactile ROS subscribers"""
        base_topic = f"/{self.config.name}"
        
        # Tactile image
        tactile_topic = f"{base_topic}/tactile_image"
        self.subscribers['tactile'] = rospy.Subscriber(
            tactile_topic, Image, self._tactile_callback, queue_size=1
        )
    
    def _rgb_callback(self, msg):
        """RGB image callback"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.latest_rgb = cv_image
        except Exception as e:
            print(f"RGB callback error: {e}")
    
    def _depth_callback(self, msg):
        """Depth image callback"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.latest_depth = cv_image
        except Exception as e:
            print(f"Depth callback error: {e}")
    
    def _tactile_callback(self, msg):
        """Tactile image callback"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.latest_tactile = cv_image
        except Exception as e:
            print(f"Tactile callback error: {e}")
    
    def capture_frame(self) -> Optional[SensorData]:
        """Capture frame from ROS topics"""
        # This would collect the latest data from ROS callbacks
        # Simplified implementation
        data = SensorData(
            timestamp=time.time(),
            sensor_name=self.config.name,
            pose=self.config.transform
        )
        
        if hasattr(self, 'latest_rgb'):
            data.rgb_image = self.latest_rgb
        if hasattr(self, 'latest_depth'):
            data.depth_image = self.latest_depth
        if hasattr(self, 'latest_tactile'):
            data.tactile_image = self.latest_tactile
        
        return data
    
    def cleanup(self):
        """Cleanup ROS subscribers"""
        for sub in self.subscribers.values():
            sub.unregister()

class SensorManager:
    """Manages multiple sensors and data synchronization"""
    
    def __init__(self):
        self.sensors: Dict[str, BaseSensor] = {}
        self.frame_callbacks: List[Callable[[FrameData], None]] = []
        self.sync_window = 0.1  # 100ms synchronization window
        self.frame_buffer = {}
        self.running = False
        
    def add_sensor(self, sensor: BaseSensor):
        """Add a sensor to the manager"""
        self.sensors[sensor.config.name] = sensor
        sensor.add_callback(self._sensor_data_callback)
        print(f"📷 Added sensor: {sensor.config.name}")
    
    def remove_sensor(self, sensor_name: str):
        """Remove a sensor"""
        if sensor_name in self.sensors:
            self.sensors[sensor_name].stop_streaming()
            del self.sensors[sensor_name]
            print(f"🗑️ Removed sensor: {sensor_name}")
    
    def add_frame_callback(self, callback: Callable[[FrameData], None]):
        """Add frame data callback"""
        self.frame_callbacks.append(callback)
    
    def start_all_sensors(self):
        """Start all sensors"""
        self.running = True
        
        for sensor in self.sensors.values():
            if sensor.config.enabled:
                try:
                    sensor.start_streaming()
                except Exception as e:
                    print(f"Failed to start {sensor.config.name}: {e}")
        
        # Start synchronization thread
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        
        print(f"✅ Started {len(self.sensors)} sensors")
    
    def stop_all_sensors(self):
        """Stop all sensors"""
        self.running = False
        
        for sensor in self.sensors.values():
            sensor.stop_streaming()
        
        if hasattr(self, 'sync_thread'):
            self.sync_thread.join(timeout=1.0)
        
        print("🛑 Stopped all sensors")
    
    def _sensor_data_callback(self, sensor_data: SensorData):
        """Handle incoming sensor data"""
        timestamp = sensor_data.timestamp
        
        # Add to frame buffer
        if timestamp not in self.frame_buffer:
            self.frame_buffer[timestamp] = {}
        
        self.frame_buffer[timestamp][sensor_data.sensor_name] = sensor_data
    
    def _sync_loop(self):
        """Synchronize sensor data into frames"""
        frame_id = 0
        
        while self.running:
            current_time = time.time()
            
            # Find timestamps to process
            timestamps_to_process = []
            for timestamp in list(self.frame_buffer.keys()):
                if current_time - timestamp > self.sync_window:
                    timestamps_to_process.append(timestamp)
            
            # Process each timestamp
            for timestamp in sorted(timestamps_to_process):  # Process in order
                frame_data = self._create_synchronized_frame(timestamp, frame_id)
                if frame_data:
                    # Call frame callbacks
                    for callback in self.frame_callbacks:
                        try:
                            callback(frame_data)
                        except Exception as e:
                            print(f"Frame callback error: {e}")
                    
                    frame_id += 1
                
                # Remove processed timestamp
                del self.frame_buffer[timestamp]
            
            time.sleep(0.01)  # 100Hz sync loop
    
    def _create_synchronized_frame(self, timestamp: float, frame_id: int) -> Optional[FrameData]:
        """Create synchronized frame from sensor data"""
        sensor_data = self.frame_buffer.get(timestamp, {})
        
        if not sensor_data:
            return None
        
        # Collect data by modality
        rgb_images = {}
        depth_images = {}
        tactile_images = {}
        tactile_depth = {}
        camera_poses = {}
        camera_intrinsics = {}
        tactile_poses = {}
        
        for sensor_name, data in sensor_data.items():
            if data.rgb_image is not None:
                rgb_images[sensor_name] = data.rgb_image
            if data.depth_image is not None:
                depth_images[sensor_name] = data.depth_image
            if data.tactile_image is not None:
                tactile_images[sensor_name] = data.tactile_image
            if data.tactile_depth is not None:
                tactile_depth[sensor_name] = data.tactile_depth
            if data.pose is not None:
                if 'tactile' in sensor_name:
                    tactile_poses[sensor_name] = data.pose
                else:
                    camera_poses[sensor_name] = data.pose
            if data.intrinsics is not None:
                camera_intrinsics[sensor_name] = data.intrinsics
        
        # Create FrameData
        frame = FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            rgb_images=rgb_images if rgb_images else None,
            depth_images=depth_images if depth_images else None,
            tactile_images=tactile_images if tactile_images else None,
            tactile_depth=tactile_depth if tactile_depth else None,
            camera_poses=camera_poses if camera_poses else None,
            camera_intrinsics=camera_intrinsics if camera_intrinsics else None,
            tactile_poses=tactile_poses if tactile_poses else None,
            metadata={'synchronized': True, 'sensor_count': len(sensor_data)}
        )
        
        return frame
    
    def get_sensor_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sensors"""
        status = {}
        
        for name, sensor in self.sensors.items():
            latest_data = sensor.get_latest_data()
            
            status[name] = {
                'running': sensor.running,
                'enabled': sensor.config.enabled,
                'fps': sensor.config.fps,
                'queue_size': sensor.data_queue.qsize(),
                'last_timestamp': latest_data.timestamp if latest_data else None,
                'data_rate': self._calculate_data_rate(sensor)
            }
        
        return status
    
    def _calculate_data_rate(self, sensor: BaseSensor) -> float:
        """Calculate sensor data rate"""
        # Simplified implementation
        return sensor.config.fps if sensor.running else 0.0

def create_sensor_from_config(config: SensorConfig) -> BaseSensor:
    """Factory function to create sensors from configuration"""
    sensor_type = config.sensor_type.lower()
    
    if sensor_type == "realsense":
        return RealSenseSensor(config)
    elif sensor_type == "tactile":
        return TactileSensor(config)
    elif sensor_type.endswith("_ros"):
        return ROSSensor(config)
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

def setup_default_sensors() -> SensorManager:
    """Setup default sensor configuration"""
    manager = SensorManager()
    
    # Add RealSense camera
    realsense_config = SensorConfig(
        name="realsense_front",
        sensor_type="realsense",
        fps=30,
        resolution=(640, 480)
    )
    
    try:
        realsense_sensor = create_sensor_from_config(realsense_config)
        manager.add_sensor(realsense_sensor)
    except Exception as e:
        print(f"Could not add RealSense sensor: {e}")
    
    # Only real sensor backends allowed - mock sensors removed for strict fail-fast operation
    
    return manager