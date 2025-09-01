# GaussianFeels Project Structure

The project has been reorganized for better modularity and clarity:

## 📁 Core Modules

### `camera/` - Camera-specific functionality
- `camera_pipeline.py` - Main camera processing pipeline
- `io/` - Camera I/O operations (depth filters, segmentation, RGBD data)
- `utils/` - Camera-specific utilities (geometry, transforms)

### `tactile/` - Tactile sensing functionality
- `gaussianfeels/` - Tactile-specific GaussianFeels modules
  - `contrib/` - External contributions (tactile transformer)
  - `datasets/` - Tactile datasets and data utilities
  - `geometry/` - Tactile geometry processing
  - `modules/` - Tactile sensor modules (Allegro hand, sensors)
- `io/` - Tactile I/O operations (predictors)

### `fusion/` - Multi-modal fusion
- `core/` - Core fusion algorithms
  - `fusion.py` - Object-centric multi-modal fusion pipeline
  - `pose_optimizer.py` - Pose optimization for fusion
- `loss/` - Fusion-specific loss functions
- `main.py` - Fusion entry point

### `gaussian/` - Core Gaussian Splatting
- `backend/` - Backend implementations
- `core/` - Core Gaussian algorithms
  - `gaussian_field.py` - Main Gaussian field implementation
  - `coarse_to_fine_optimizer.py` - Optimization strategies
  - `densify_prune.py` - Field management
- `render/` - Rendering pipeline
- `spatial/` - Spatial data structures
- `utils/` - Gaussian-specific utilities

### `shared/` - Shared utilities
- `config/` - Configuration system
- `io/` - General I/O operations
- `memory/` - Memory management
- `modes/` - Operation modes
- `registration/` - Point cloud registration
- `transforms/` - Coordinate transforms
- `utils/` - General utilities

### `main/` - Main application
- All core application files (config, datasets, main, trainer, etc.)

### `instrumentation/` - Performance monitoring
- Live counters and performance tracking

### `evaluation/` - Evaluation framework
- Evaluation metrics and testing

## 📁 Supporting Directories

- `data/` - Dataset storage
- `docs/` - Documentation
- `tests/` - Test files
- `scripts/` - Utility scripts
- `webui/` - Web user interface
- `misc/` - Miscellaneous utilities and configurations
- `validation/` - Mathematical validation
- `references/` - Reference implementations

## Key Benefits

1. **Clear separation of concerns**: Each module focuses on its specific functionality
2. **Better maintainability**: Related code is grouped together
3. **Improved imports**: Cleaner, more logical import paths
4. **Modular design**: Easy to add new modalities or components
5. **Academic structure**: Well-organized for research and publications

## Import Examples

```python
# Camera functionality
from camera.io.segmentation import SegmentationProcessor
from camera.utils.camera_geometry import CameraGeometry

# Tactile functionality
from tactile.gaussianfeels.modules.sensor import TactileSensor

# Fusion
from fusion.core.fusion import ObjectCentricFusion
from fusion.loss.tactile_loss import tactile_surface_loss

# Gaussian splatting
from gaussian.core.gaussian_field import ObjectGaussianMap
from gaussian.render.rasterizer import GaussianRasterizer

# Shared utilities
from shared.config.config_system import GaussianFeelsConfig
from shared.memory.memory_monitor import MemoryMonitor

# Main application
from main.trainer import GaussianFeelsTrainer
```

This structure follows academic and industrial best practices for complex multi-modal systems.