# YOLOv8-Based Real-Time Video Analytics System for Mobile Phone Detection in Classroom Environments

## Overview

This project implements an AI-driven real-time mobile phone detection system specifically designed for classroom surveillance and examination monitoring. The system leverages YOLOv8 deep learning architecture to detect unauthorized mobile phone usage in educational environments with high accuracy and efficiency.

## Features

- **Real-time Detection**: Processes video streams at 18-22 FPS with minimal latency
- **High Accuracy**: Achieves 93.47% detection accuracy with 100% recall
- **Adaptive Alerting**: Smart alert system prevents audio spamming with 5-second intervals
- **Region of Interest (ROI)**: Optimized processing focusing on lower two-thirds of frame
- **Multithreaded Architecture**: Parallel frame acquisition and processing
- **Frame Optimization**: Frame skipping strategy reduces computational overhead by 30%
- **Real-time Alerts**: Instant audio notifications for unauthorized phone usage

## System Architecture

The system consists of five main phases:
1. **System Initialization and Resource Allocation**
2. **Multithreaded Frame Acquisition and Preprocessing**
3. **YOLOv8-Based Object Detection and Classification**
4. **Adaptive Alert Triggering Mechanism**
5. **Real-time Video Display and System Termination**

Diagram showcasing the model's training flow from dataset collection to validation:
![System Architecture](assets/system_architecture.jpg)

## Sample Detections
These snapshots illustrate YOLOv8 detecting mobile phones in real-world classroom footage:
![Sample Detections](assets/sample_detections_grid.jpg)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Precision | 93.47% |
| Recall | 100.00% |
| F1-Score | 96.62% |
| Specificity | 93.00% |
| False Positive Rate | 7.00% |
| mAP @0.5 | 99.50% |
| mAP @0.5:0.95 | 87.39% |
| Processing Speed | 18-22 FPS |

## Requirements

### Hardware Requirements
- Webcam/Camera (minimum 640x480 resolution)
- CPU: Intel i5 or equivalent (recommended)
- RAM: 8GB minimum, 16GB recommended
- GPU: NVIDIA GPU with CUDA support (optional but recommended)

### Software Requirements
- Python 3.8 or higher
- OpenCV 4.x
- PyTorch
- Ultralytics YOLOv8
- NumPy
- Threading support

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/yolov8-phone-detection.git
cd yolov8-phone-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLOv8 model**
```bash
# The system will automatically download yolov8n.pt on first run
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

5. **Add alarm sound**
- Place your alarm sound file as `warning.mp3` in the project directory
- Or modify the code to use your preferred audio file

## Usage

### Basic Usage

```bash
python phone_detection.py
```

### Configuration Options

The system can be configured by modifying the following parameters in the main script:

```python
# Frame processing settings
FRAME_SKIP = 3              # Process every 3rd frame
CONFIDENCE_THRESHOLD = 0.3   # Minimum confidence for detection
FRAME_WIDTH = 640           # Camera frame width
FRAME_HEIGHT = 480          # Camera frame height
FPS = 11.10                 # Target frame rate

# Alert settings
ALERT_INTERVAL = 5          # Seconds between alerts
ALARM_FILE = "warning.mp3"  # Path to alarm sound

# ROI settings
ROI_START_Y = 160           # Start of region of interest (y-coordinate)
```

### Advanced Usage

```python
from phone_detection import PhoneDetectionSystem

# Initialize system
detector = PhoneDetectionSystem(
    model_path="yolov8n.pt",
    confidence_threshold=0.3,
    frame_skip=3
)

# Start detection
detector.run()
```

## System Components

### 1. Frame Acquisition Thread
- Continuously captures frames from webcam
- Implements bounded queue (max 3 frames) to prevent memory overflow
- Runs independently from detection pipeline

### 2. YOLOv8 Detection Engine
- Uses pre-trained YOLOv8n model for object detection
- Filters detections for mobile phones (COCO class 67)
- Applies confidence threshold filtering

### 3. Region of Interest (ROI) Processing
- Focuses on lower two-thirds of frame where phones are likely to appear
- Reduces computational load by ~30%
- Improves detection efficiency in classroom settings

### 4. Adaptive Alert System
- Prevents continuous alarm triggering
- 5-second cooldown period between alerts
- Asynchronous audio playback to avoid blocking detection

## File Structure

```
yolov8-phone-detection/
├── phone_detection.py          # Main detection script
├── requirements.txt            # Python dependencies
├── yolov8n.pt                 # YOLOv8 model weights
├── warning.mp3                # Alert sound file
├── README.md                  # This file
├── assets/
│   ├── architecture_diagram.png
│   └── sample_detections.png
├── docs/
│   └── research_paper.pdf
└── utils/
    ├── __init__.py
    ├── detection_utils.py     # Utility functions
    └── visualization.py       # Visualization helpers
```

## Configuration

### Camera Settings
```python
# Modify these settings in phone_detection.py
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 11.10)
```

### Detection Parameters
```python
# YOLOv8 model settings
results = model(frame, 
                imgsz=640,
                conf=0.3,
                verbose=False)
```

### ROI Configuration
```python
# Define region of interest (lower 2/3 of frame)
roi_start_y = frame.shape[0] // 3
roi = frame[roi_start_y:, :]
```

## Troubleshooting

### Common Issues

1. **Camera not detected**
   ```bash
   # Check available cameras
   ls /dev/video*  # Linux
   # Try different camera indices in code (0, 1, 2, etc.)
   ```

2. **Low FPS performance**
   - Reduce frame resolution
   - Increase frame skip value
   - Use GPU acceleration if available

3. **False positives**
   - Increase confidence threshold
   - Adjust ROI boundaries
   - Ensure proper lighting conditions

4. **Audio alerts not working**
   - Check audio file path
   - Verify system audio settings
   - Install required audio libraries

### Performance Optimization

1. **For better accuracy:**
   - Use higher resolution camera
   - Improve classroom lighting
   - Fine-tune confidence threshold

2. **For better performance:**
   - Use GPU acceleration
   - Increase frame skip value
   - Reduce camera resolution

## Ethical Considerations

- **Privacy Protection**: Implement data anonymization techniques
- **Consent**: Ensure proper consent from students and staff
- **Transparency**: Clearly communicate system usage and purpose
- **Data Security**: Secure storage and transmission of video data
- **Bias Mitigation**: Regular testing across diverse populations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Research Paper

This implementation is based on the research paper:
**"YOLOv8-Based Real-Time Video Analytics System for Mobile Phone Detection in Classroom Environments"**

Authors: Shreya B, Pooja Shree S, Jeyakumar G  
Institution: Amrita School of Computing, Coimbatore, Amrita Vishwa Vidyapeetham, India

## Citation

If you use this code in your research, please cite:

```bibtex
@article{shreya2024yolov8,
  title={YOLOv8-Based Real-Time Video Analytics System for Mobile Phone Detection in Classroom Environments},
  author={Shreya, B and Pooja Shree, S and Jeyakumar, G},
  journal={I-SMAC Conference Proceedings},
  year={2024},
  institution={Amrita Vishwa Vidyapeetham}
}
```

## Contact

For questions or support, please contact:
- Shreya B: cb.sc.u4cse23347@cb.students.amrita.edu
- Pooja Shree S: cb.sc.u4cse23346@cb.students.amrita.edu
- Jeyakumar G (Corresponding Author): g_jeyakumar@cb.amrita.edu

## Acknowledgments

- Ultralytics team for YOLOv8 framework
- OpenCV community for computer vision tools
- Amrita School of Computing for research support

---

**Note**: This system is designed for educational purposes and should be deployed with proper ethical guidelines and institutional policies in place.
