# Face Detection and Recognition System

This project implements a lightweight, real-time face detection and recognition system that runs on CPU without requiring heavy deep learning frameworks. It uses ONNX-optimized SCRFD for face detection and MobileFaceNet for face recognition, making it easy to deploy without installing PyTorch, or PaddlePaddle. The system efficiently processes both images and videos, comparing detected faces against a database of known faces.

1. Developed a lightweight face detection and recognition pipeline using ONNX-optimized SCRFD and MobileFaceNet models, achieving real-time performance without heavy ML frameworks

2. Implemented a robust face processing system with 5-point landmark detection, affine transformation, and adaptive face alignment, enhancing recognition accuracy across diverse poses and lighting conditions

3. Engineered a scalable video processing framework with parallel face tracking, CSV-based result logging, and automated face database management, supporting both real-time and batch processing workflows
   
## Features

- Real-time face detection using lightweight SCRFD model
- Efficient face recognition using compact MobileFaceNet
- CPU-friendly video processing
- Accurate 5-point facial landmark detection
- Highly customizable detection parameters
- Detailed CSV output for face matching results
- Simple face database management

## Installation

Quick setup with minimal dependencies:

```bash
pip install opencv-python numpy onnxruntime pandas scikit-image
```

That's it! No need to install heavy frameworks like PyTorch, TensorFlow, or PaddlePaddle.

## Required Models

1. Face Detection Model (SCRFD):
   - File: `scrfd.onnx`

2. Face Recognition Model (MobileFaceNet):
   - File: `MobileFaceNet.onnx`

## Project Structure

```
scrfd+MobileFaceNet/
├── video_demo_onnx.py    # Main script
├── scrfd.onnx           # Face detection model
├── MobileFaceNet.onnx   # Face recognition model
└── database/            # Directory for reference face images
```

## Usage

```bash
python3 video_demo_onnx.py \
    --video_path <path_to_video> \
    --detector_backend scrfd.onnx \
    --facial_recognition MobileFaceNet.onnx \
    --db_path ./database \
    --expand_percentage 0 \
    --conf_thres_face_detect 0.35 \
    --iou_thres_face_detect 0.7 \
    --cosine_distanche_thres 0.5 \
    --face_area_threshold 12544
```

### Parameters

- `--video_path`: Path to input video file
- `--detector_backend`: Path to SCRFD model file (default: scrfd.onnx)
- `--facial_recognition`: Path to MobileFaceNet model file (default: MobileFaceNet.onnx)
- `--db_path`: Path to the database directory containing reference faces
- `--expand_percentage`: Percentage to expand detected face area (default: 0)
- `--conf_thres_face_detect`: Confidence threshold for face detection (default: 0.35)
- `--iou_thres_face_detect`: IOU threshold for face detection (default: 0.7)
- `--cosine_distanche_thres`: Threshold for face recognition matching (default: 0.5)
- `--face_area_threshold`: Minimum area for detected faces (default: 12544)
- `--refresh_database`: Whether to refresh the face database (default: True)
- `--draw_kps`: Whether to draw facial keypoints (default: True)

## Output

The script generates several outputs:

1. Processed video files:
   - `video_res_onnx.mp4`: Main output video with face detection boxes, recognition results, and confidence scores
   - Located in the output directory specified by `--opts_dir` (default: `./video_res_onnx/`)

2. CSV files with face comparison results:
   - Complete results: `face_comparison_results_<video_name>.csv`
   - Matched faces only: `true_<video_name>.csv`

### Output Directory Structure

```
video_res_onnx/                                              # Default output directory
├── video_res_onnx.mp4                                      # Main output video with detection results
├── crop_res/                                               # Cropped face images (frame_XXXX_Y.png)
├── video_frames/                                           # Original video frames (frame_XXXX.png)
├── warpAffine_res/                                         # Aligned face images (frame_XXXX_Y.png)
├── res_frames/                                             # Processed frames with annotations (frame_XXXX.png)
└── csv_results/                                            # Face comparison results
    ├── face_comparison_results_<video_name>.csv            # All comparison results
    └── true_<video_name>.csv                               # Only matched faces
```

Note: In filenames, XXXX represents the frame number and Y represents the face index in that frame.

## Troubleshooting

If faces are not being detected:

1. Try lowering the confidence threshold:
```bash
--conf_thres_face_detect 0.25
```

2. Reduce the minimum face area for smaller faces:
```bash
--face_area_threshold 3600
```

3. Adjust IOU threshold if faces are close together:
```bash
--iou_thres_face_detect 0.5
```

## References

- SCRFD: [PaddleHub](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_bnkps_shape640x640.onnx)
- MobileFaceNet: [PaddlePaddle](https://github.com/PaddlePaddle/PaddleX/blob/revert-3532-add_ts_show/docs/module_usage/tutorials/cv_modules/face_feature.md)
