# Computer Vision Project – Object Detection & Segmentation

This repository contains a computer vision project implementing **state-of-the-art (SOTA) open-source algorithms** for object detection and segmentation: **YOLO**, **Mask R-CNN (TorchVision)**, and **Faster R-CNN**.

---

## Project Overview

This project demonstrates video-based object detection and segmentation using three widely adopted state-of-the-art algorithms:

1. **YOLO (You Only Look Once)**  
   A one-stage detector designed for high-speed, real-time object detection.

2. **Mask R-CNN (TorchVision)**  
   A two-stage detector that performs object detection and **instance-level segmentation**, providing pixel-wise object masks.

3. **Faster R-CNN**  
   A two-stage object detection algorithm that balances detection accuracy and computational efficiency, serving as a strong baseline for comparison.

All three algorithms process video inputs and generate output videos with detected objects visualized using bounding boxes and (where applicable) segmentation masks.

---

## Features

- Video-based object detection and segmentation  
- Comparison of three state-of-the-art algorithms  
- Pretrained models (no dataset training required)  
- Live video visualization during inference  
- Automatically saved output videos  

---

## Installation

1. **Clone the repository**  
    ```bash
    git clone https://github.com/myingineer/cv_project.git
    cd cv_project
    ```

2. Create a Virtual Environment 
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: `venv\Scripts\activate`
    ```

3. Install Dependencies 
    ```bash
    pip install -r requirements.txt
    ```

4.  Mac-specific step (if using macOS)
    ```bash
    /Applications/Python\ 3.14/Install\ Certificates.command
    ```
    **Make sure the version number `(3.14)` matches your installed Python version**

### Usage
1. Algorithm selection
    Depending on the algorithm you want to try out, place your video file in the corresponding `videos` folder:
        - `running_Yolo/videos` → for YOLO
        - `running_Mask_RCNN/videos` → for Mask R-CNN
        - `running_Faster_RCNN/videos` → for Faster R-CNN
    Guidelines
    - Use simple file names (preferably `.mp4` format).
    Update the video file name in the script:
        - YOLO: `yolo.py`, line **8**
        - Mask R-CNN: `Mask_rcnn.py`, line **46**
        - Faster R-CNN: `faster_rcnn.py`, line **44**

2. Running the Script
    - For your selected algorithm, run the file for the said code
    ```bash
    python yolo.py       # For YOLO
    python Mask_rcnn.py  # For Mask R-CNN
    python faster_rcnn.py # For Faster R-CNN
    ```

3. View Outputs
    - The processed video will play automatically during execution.
    - Press `q` to stop the video loop.
    - The output video is saved in the `output_videos` directory, which is automatically created on first run.

## NOTES
- All models use **pretrained COCO weights**, eliminating the need for custom training datasets.
- Frame skipping and confidence thresholding are used to balance performance and accuracy.
- The project is designed for **qualitative and comparative evaluation** of detection and segmentation methods.

## AUTHOR
Alexander Soromtochukwu Emeka-Akam
Applied AI Student @ IU International University of Applied Sciences
Berlin, Germany
- GitHub: [myingineer](https://github.com/myingineer/)
- LinkedIn: [Alexander Soromtochukwu Emeka-Akam](https://www.linkedin.com/in/myingineer/)
