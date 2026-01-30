# Computer Vision Project – Object Detection & Segmentation

This repository contains a computer vision project implementing **state-of-the-art (SOTA) open-source algorithms** for object detection and segmentation: **YOLO** and **Mask R-CNN (TorchVision)**.  

---

## Project Overview

This project demonstrates real-time object detection and segmentation using two powerful algorithms:

1. **YOLO (You Only Look Once)** – High-speed real-time object detection.  
2. **Mask R-CNN (TorchVision)** – Accurate object detection with instance segmentation capabilities.  

Both algorithms are implemented to process videos and produce output videos with detected objects highlighted.

---

## Features

- Real-time object detection  
- High accuracy and speed  
- Easy integration with various applications  
- Output videos with detected objects automatically saved  

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
    **Make sure the version number (3.14) matches your installed Python version**

### Usage
1. Algorithm selection
    - Depending on the algorithm you want to try out, place your video file in the corresponding `videos` folder:
        - `running_Yolo/videos` → for YOLO
        - `running_Mask_RCNN/videos` → for Mask R-CNN
    - Use simple file names (preferably `.mp4` format).
    - Update the video file name in the script:
        - YOLO: `yolo.py`, line **8**
        - Mask R-CNN: `Mask_rcnn.py`, line **46**

2. Running the Script
    - For your selected algorithm, run the file for the said code
    ```bash
    python yolo.py       # For YOLO
    python Mask_rcnn.py  # For Mask R-CNN
    ```

3. View Outputs
    - The output video with detected objects will be saved in the `output_videos` folder which automatically generates once you run the code.