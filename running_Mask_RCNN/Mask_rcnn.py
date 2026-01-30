import cv2                      
import torch                    # PyTorch core library
import torchvision              # Torchvision provides pretrained vision models
from torchvision.transforms import functional as F  # Image → tensor conversion
import numpy as np 

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load pretrained Mask R-CNN model
# ----------------------------

# Load Mask R-CNN with a ResNet-50 backbone pretrained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    weights="DEFAULT"
)

# Move the model to GPU or CPU
model.to(device)

# Set the model to evaluation mode (important for inference)
model.eval()

# List of object classes used by COCO-trained models
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ----------------------------
# Load input video
# ----------------------------

# Open the video file
cap = cv2.VideoCapture("running_Mask_RCNN/videos/2.mp4")

# Get video properties (needed to save output)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Frame height
fps = int(cap.get(cv2.CAP_PROP_FPS))              # Frames per second

# Create a VideoWriter object to save the processed video
out = cv2.VideoWriter(
    "output_videos/output_mask_rcnn.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)                                 
)

# Minimum confidence score to accept a detection
CONF_THRESHOLD = 0.5

# ----------------------------
# Main video processing loop
# ----------------------------

while cap.isOpened():  

    # Read a frame from the video           
    ret, frame = cap.read()     

    # Break the loop if no frame is read  
    if not ret:                    
        break

    # ----------------------------
    # Preprocess frame
    # ----------------------------

    # Convert BGR (OpenCV format) to RGB (PyTorch format)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert image to PyTorch tensor and normalize
    tensor = F.to_tensor(rgb).to(device)

    # ----------------------------
    # Run inference
    # ----------------------------

    # Disable gradient computation (faster and memory-efficient)
    with torch.no_grad():
        predictions = model([tensor])[0]  # Run model on one frame

    # ----------------------------
    # Extract predictions
    # ----------------------------

    boxes = predictions["boxes"]    # Bounding box coordinates
    labels = predictions["labels"]  # Class IDs
    scores = predictions["scores"]  # Confidence scores
    masks = predictions["masks"]    # Segmentation masks

    # ----------------------------
    # Loop through detections
    # ----------------------------

    for i in range(len(boxes)):     # Iterate over all detected objects

        # Skip detections below confidence threshold
        if scores[i] < CONF_THRESHOLD:
            continue

        # Get bounding box coordinates
        x1, y1, x2, y2 = boxes[i].int().cpu().numpy()

        # Get class name from class ID
        label = COCO_CLASSES[labels[i]]

        # Round confidence score for display
        conf = round(float(scores[i]), 2)

        # ----------------------------
        # Draw bounding box
        # ----------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),  
            2             
        )

        # ----------------------------
        # Draw label + confidence
        # ----------------------------

        cv2.putText(
            frame,
            f"{label} {conf}",
            (x1, max(30, y1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ----------------------------
        # Process segmentation mask
        # ----------------------------

        # Extract the mask and convert to NumPy array
        mask = masks[i, 0].cpu().numpy()

        # Convert mask probabilities to binary mask
        mask = mask > 0.5

        # Create a colored mask overlay
        colored_mask = np.zeros_like(frame)
        colored_mask[mask] = (0, 255, 0)

        # Blend the mask with the original frame
        frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0)

    # ----------------------------
    # Save and display frame
    # ----------------------------

    out.write(frame)                # Write frame to output video
    cv2.imshow("Mask R-CNN", frame) # Display frame

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup resources
# ----------------------------

cap.release()       # Release video capture
out.release()       # Release video writer
cv2.destroyAllWindows()  # Close OpenCV windows
