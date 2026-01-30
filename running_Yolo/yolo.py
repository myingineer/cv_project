from ultralytics import YOLO
import cv2
import cvzone

# ----------------------------
# Video input
# ----------------------------
cap = cv2.VideoCapture('running_Yolo/videos/2.mp4')

# ----------------------------
# Load YOLOv8 SEGMENTATION model
# ----------------------------
model = YOLO('yolo_weights/yolov8n-seg.pt')

# ----------------------------
# COCO class names
# ----------------------------
classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ----------------------------
# Output video writer
# ----------------------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Get width of the frames
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get height of the frames
fps = int(cap.get(cv2.CAP_PROP_FPS)) # Get frames per second

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(
    "yolo__output_videos/output_video_yolov8_segmentation.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# ----------------------------
# Processing loop
# ----------------------------
while cap.isOpened():

    # Read frame
    success, img = cap.read()

    # Break the loop if no frame is read
    if not success:
        break

    # Perform inference
    results = model(img, stream=True)

    # Iterate over detections
    for r in results:
        boxes = r.boxes

        # Iterate over each box
        for box in boxes:

            # Extract confidence and class id
            conf = float(box.conf[0])
            if conf < 0.5:   # CONFIDENCE THRESHOLD
                continue

            conf = round(conf, 2)
            cls = int(box.cls[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cvzone.cornerRect(
                img,
                (x1, y1, x2 - x1, y2 - y1),
                l=9,
                rt=3,
                colorR=(255, 0, 255)
            )

            # Draw label
            cvzone.putTextRect(
                img,
                f'{classNames[cls]} {conf}',
                (max(0, x1), max(35, y1)),
                scale=2,
                thickness=3,
                offset=3
            )

    # Write frame to output video
    out.write(img)

    # Display the frame
    cv2.imshow("YOLOv8 Segmentation", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
