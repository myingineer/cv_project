# A COMPUTER VISION PROJECT

## This repository houses a computer vision project for object detection using three Open Source CV SOTA. In this repository,
- I implemented the YOLO algorithm for real-time object detection and segmentation
- I also implemented the Mask_R_CNN also for real-time object detection and segmentation using **TORCH VISION**

### Project Overview
- This project utilizes the YOLO (You Only Look Once) algorithm for real-time object detection and segmentation.
- It also incorporates the Mask R-CNN model from TorchVision for enhanced object detection and segmentation capabilities.

### Features
- Real-time object detection
- High accuracy and speed
- Easy integration with various applications

### Installation

1. Clone the repository 
    ```bash
    git clone https://github.com/myingineer/cv_project.git
    cd cv_project
    ```

2. Create a Virtual Environment 
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install Dependencies 
    ```bash
    pip install -r requirements.txt
    ```

    If you are on a Mac, inside the terminal run this code 
    ```bash
    /Applications/Python\ 3.14/Install\ Certificates.command
    ```
    Bear in mind the **3.14** must match the Python version you have installed 

### Usage
1. Algorithm selection
    - Depending on the algorithm you want to try out, you can load your own videos into the videos folder of that algorithm.
    - Use an easy name for the video file (`peferably mp4 format`)
    - If you copied a video to `running_Yolo/videos` folder, in the **yolo.py**. file line **8**, update the file in **quotes** to your video file name.
    - If you copied a video to **running_Mask_RCNN/videos**, in the **Mask_rcnn.py**. file line **46**, update the file in **quotes** to your video file name.

2. Running the Script
    - For your selected algorithm, run the file for the said code

3. Viewing Outputs
    - The output video with detected objects will be saved in the `output_videos` folder which automatically generates once you run the code.