# Fire Detection Using Image Processing

## Overview
This project implements a real-time fire detection system using classical computer vision techniques. The system analyzes video footage to detect fire by combining motion detection, color-based segmentation, and temporal validation. It provides an interactive dashboard for monitoring and tuning detection parameters in real-time.

## Features
- **Hybrid Motion Detection**: Supports both MOG2 (Mixture of Gaussians) background subtraction and frame differencing
- **HSV Color Filtering**: Specialized fire color detection in HSV color space
- **Temporal Validation**: Reduces false positives through multi-frame confirmation
- **Interactive Parameter Tuning**: Real-time adjustment of detection thresholds
- **Visual Dashboard**: Multi-panel view showing detection stages
- **Alarm System**: Automatic logging and frame capture upon fire detection


## Detection Pipeline

### 1. Motion Detection
The system offers two motion detection methods:

#### MOG2 (Mixture of Gaussians - Default)
- **Algorithm**: Advanced background subtraction using Gaussian mixture models
- **Parameters**:
  - `history=1000`: Number of frames for background model
  - `varThreshold=50`: Threshold for pixel classification
  - `detectShadows=True`: Shadow detection enabled
  - `learning_rate=0.001`: Adaptation rate for background model
- **Processing**: 
  - Shadow removal (threshold at 250)
  - Morphological operations (erosion + dilation) for noise reduction

#### Frame Differencing (Alternative)
- **Method**: Absolute difference between consecutive frames
- **Processing**:
  - Gaussian blur (5x5 kernel) for noise reduction
  - Threshold value: `th_m=21` (adjustable)
  - Morphological opening and dilation

### 2. Fire Color Detection (HSV)
- **Color Space**: HSV (Hue, Saturation, Value)
- **Fire Color Range**:
  - **Hue**: 0-35 (red-orange-yellow spectrum)
  - **Saturation**: 60-255 (filters out white lights - low saturation indicates white/spotlight)
  - **Value**: 135-255 (brightness threshold)
- **White Light Filtering**: S>60 threshold specifically eliminates false positives from white lights and reflections
- **Morphological Processing**:
  - Closing operation (15x15 kernel) to fill holes in fire regions
  - Dilation (3x3 kernel) to strengthen detected areas

### 3. Candidate Generation
- **Method**: Bitwise AND of motion mask and color mask
- **Additional Processing**: Closing operation (5x5 kernel, 2 iterations)
- **Optimization**: Only processes moving regions, reducing computational cost

### 4. Temporal Validation
Reduces false positives through multi-frame confirmation:
- **Fire Area Threshold**: `fire_area_th=250` pixels
- **Confirmation Frames**: `fire_confirm_frames=10` consecutive frames
- **Counter Logic**:
  - Increments when both fire area and candidate area thresholds are met
  - Decrements if conditions not met
  - Fire confirmed when counter reaches threshold

### 5. Spatial Filtering
- **Minimum Area**: `min_area=120` pixels
- **Aspect Ratio**: Rejects boxes with ratio >4.0 or <0.2 (filters out unrealistic fire shapes)
- **Bounding Box**: Red rectangles drawn around detected fire regions




