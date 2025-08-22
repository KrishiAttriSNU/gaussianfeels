# GaussianFeels System Architecture

## Architecture Decision Record (ADR)

**Status**: Implemented  
**Date**: 2025-08-02  
**Decision**: 3-Thread Architecture with Object-Centric Fusion Pipeline

## Executive Summary

Implemented a 3-thread architecture following Consensus Final specifications for GaussianFeels tactile-visual object reconstruction. The system provides real-time processing with strict performance targets and memory constraints.

## Core Architecture Components

### 1. Threading Engine (`gaussianfeels/threads/engine.py`)

**Decision**: 3-Thread Architecture
- **Data Thread**: 100Hz sensor processing
- **Render Thread**: 30-60 FPS rendering with 1kHz interpolation  
- **Optimizer Thread**: 10-15 Hz Gaussian parameter updates

**Rationale**:
- Separates I/O-bound sensor processing from compute-intensive optimization
- Enables real-time rendering through high-frequency interpolation
- Maintains deterministic performance under load

**Key Features**:
- Thread-safe communication via queues
- 1kHz interpolation WITHOUT parameter updates (critical requirement)
- Memory caps enforcement (≤300k active Gaussians)
- Configurable thread priorities for real-time performance

### 2. Object-Centric Fusion (`gaussianfeels/core/fusion.py`)

**Decision**: Object-centric coordinate system with tactile-primary fusion
- Transform all inputs: pc_world → pc_obj per frame
- Tactile primary (0.8 weight), vision secondary (0.2 weight)
- Configurable integration modes: concurrent|tactile_first|vision_first

**Rationale**:
- Object-centric coordinates provide stable reference frame during manipulation
- Tactile data provides more reliable contact information than vision
- Multiple integration modes support different sensor configurations

**Key Features**:
- Maintains T_WO(i) pose per frame with validation
- Enforces depth×mask rule before back-projection
- Temporal consistency validation with pose delta limits
- Memory-efficient batch processing

### 3. Tactile Predictors (`gaussianfeels/io/tactile_predictors.py`)

**Decision**: Factory pattern with multiple prediction modes
- VISTaC: Vision-based tactile depth prediction
- ViT: Vision Transformer with memory optimization
- NormalFlow: Optical flow-based depth estimation  
- GT: Ground truth for validation

**Rationale**:
- Different tactile sensors may require different prediction algorithms
- Factory pattern enables runtime mode selection
- Memory optimization critical for real-time performance

**Key Features**:
- Mandatory depth×mask rule enforcement
- Point reduction for ViT mode (50% sampling)
- Back-projection with configurable camera intrinsics
- Comprehensive error handling and validation

## Quality Attributes

### Performance Requirements
- **Data Thread**: 100Hz sustained (10ms cycle time)
- **Render Thread**: 30-60 FPS (16.7-33.3ms cycle time)
- **Optimizer Thread**: 10-15 Hz (66.7-100ms cycle time)
- **Interpolation**: 1kHz WITHOUT parameter updates

### Memory Constraints
- **Active Gaussians**: ≤300,000 maximum
- **Memory Usage**: ≤158MB for active set
- **Point Limits**: 15k tactile, 50k vision points per frame

### Reliability Requirements
- Thread-safe communication with bounded queues
- Graceful degradation under high load
- Automatic memory cap enforcement
- Pose validation with delta limits (≤1m per frame)

## Integration Patterns

### Concurrent Mode (Default)
```python
# Both modalities processed simultaneously
fusion_result = fusion.fuse_modalities(
    tactile_data=tactile_sensors,
    vision_data=camera_data,
    camera_params=cam_params
)
```

### Tactile-First Mode
```python  
# Tactile processed first, vision follows with 50ms delay
fusion.integration_mode = IntegrationMode.TACTILE_FIRST
```

### Vision-First Mode
```python
# Vision processed first, tactile follows with 16.7ms delay  
fusion.integration_mode = IntegrationMode.VISION_FIRST
```

## Component Interactions

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Data Thread   │───▶│   Fusion Core    │───▶│   Optimizer Thread  │
│     100Hz       │    │  Object-Centric  │    │      10-15Hz        │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                       │                          │
         ▼                       ▼                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Tactile Predict │    │  Pose Tracking   │    │ Gaussian Updates    │
│ VISTaC/ViT/etc  │    │    T_WO(i)       │    │ Densify/Prune       │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Render Thread   │
                       │   30-60 FPS      │
                       │ 1kHz Interpolate │
                       └──────────────────┘
```

## Trade-offs and Alternatives

### Threading Model
**Chosen**: 3-Thread Architecture  
**Alternative**: Single-threaded pipeline  
**Trade-off**: Complexity vs. performance - 3 threads enable real-time processing but require careful synchronization

### Coordinate System
**Chosen**: Object-centric (pc_world → pc_obj)  
**Alternative**: World-centric coordinates  
**Trade-off**: Transform overhead vs. stability during manipulation

### Fusion Strategy  
**Chosen**: Tactile-primary with configurable weighting  
**Alternative**: Equal weighting or adaptive fusion  
**Trade-off**: Domain expertise vs. learned optimization

## Operational Considerations

### Deployment
- Requires CUDA-capable GPU for real-time performance
- Thread priorities may require elevated permissions
- Memory limits enforced automatically

### Monitoring
- Performance metrics available via `get_performance_metrics()`
- Memory usage tracked per component
- Queue sizes monitored for bottleneck detection

### Scaling
- Thread counts configurable via ThreadConfig
- Memory limits adjustable per deployment
- Predictor modes selectable at runtime

## Success Criteria Alignment

✅ **Threading Architecture**: 3-thread implementation with correct frequencies  
✅ **Object-centric Transforms**: pc_world → pc_obj per frame  
✅ **Memory Caps**: ≤300k Gaussian enforcement  
✅ **Tactile Integration**: depth×mask rule enforcement  
✅ **Performance Targets**: 100Hz data, 30-60 FPS render, 10-15 Hz optimization  
✅ **1kHz Interpolation**: Without parameter updates (critical requirement)

## Future Enhancements

1. **Adaptive Threading**: Dynamic thread count based on load
2. **Learned Fusion Weights**: Replace fixed 0.8/0.2 with learned weights  
3. **Multi-GPU Support**: Distribute workload across multiple GPUs
4. **Advanced Predictors**: Integration of newer tactile prediction models
5. **Temporal Consistency**: Enhanced temporal loss functions

---

**Architecture Review**: Approved by System Architect  
**Implementation Status**: Core components implemented and validated  
**Next Steps**: Integration testing with real sensor data