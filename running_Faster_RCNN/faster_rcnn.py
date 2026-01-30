import cv2
import torch
import torchvision
from torchvision.transforms import functional as F  # Image → tensor conversion


# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Faster R-CNN with ResNet-50 backbone pretrained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)

# Move model to GPU or CPU
model.to(device)

# Set model to evaluation mode (important for inference)
model.eval()

# ----------------------------
# COCO dataset class labels
# ----------------------------
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

# Open video file
cap = cv2.VideoCapture("running_Faster_RCNN/videos/2.mp4")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
fps = int(cap.get(cv2.CAP_PROP_FPS))               # Frames per second

# Create VideoWriter to save output
out = cv2.VideoWriter(
    "output_videos/output_faster_rcnn.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

PAUSE_MS = 1500  # Pause duration in milliseconds after detections

# Minimum confidence threshold
CONF_THRESHOLD = 0.5

frame_id = 0  # Frame counter

# ----------------------------
# Main video processing loop
# ----------------------------
while cap.isOpened():

    # Read next frame from video
    ret, frame = cap.read()

    # Exit loop if video ends
    if not ret:
        break

    frame_id += 1

    # Skip frames to improve performance
    if frame_id % 5 != 0:
        out.write(frame)
        cv2.imshow("Faster R-CNN", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ----------------------------
    # Preprocess frame
    # ----------------------------

    # Convert frame from BGR (OpenCV) to RGB (PyTorch)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert image to PyTorch tensor and move to device
    tensor = F.to_tensor(rgb).to(device)

    # ----------------------------
    # Run inference
    # ----------------------------

    # Disable gradients for faster inference
    with torch.no_grad():
        predictions = model([tensor])[0]

    # Extract predictions
    boxes = predictions["boxes"]     # Bounding boxes
    labels = predictions["labels"]   # Class IDs
    scores = predictions["scores"]   # Confidence scores

    # ----------------------------
    # Draw detections
    # ----------------------------

    detections = 0  # Count detections in this frame

    for i in range(len(boxes)):

        # Ignore low-confidence detections
        if scores[i] < CONF_THRESHOLD:
            continue

        detections += 1  # Valid detection found

        # Get bounding box coordinates
        x1, y1, x2, y2 = boxes[i].int().cpu().numpy()

        # Get class label name
        label = COCO_CLASSES[labels[i]]

        # Round confidence for display
        conf = round(float(scores[i]), 2)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        # Draw label and confidence score
        cv2.putText(
            frame,
            f"{label} {conf}",
            (x1, max(30, y1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            3
        )

    # ----------------------------
    # Save and display frame
    # ----------------------------
    out.write(frame)                     # Save frame to output video
    cv2.imshow("Faster R-CNN", frame)    # Display frame live

    # Exit on 'q' key
    if detections > 0:
        key = cv2.waitKey(PAUSE_MS)
    else:
        key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break


# ----------------------------
# Cleanup resources
# ----------------------------
cap.release()            # Release video capture
out.release()            # Release video writer
cv2.destroyAllWindows()  # Close all OpenCV windows
