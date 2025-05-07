import cv2
import numpy as np

# Load YOLOv3 model
weights_path = "yolov3-tiny.weights"  # Path to yolov3.weights
config_path = "yolov3-tiny (1).cfg"       # Path to yolov3.cfg
names_path = "coco.names"        # Path to coco.names

# Load the class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the network
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny (1).cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Set up colors for bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Start the video capture (use 0 for default camera, or change to a video file path)
cap = cv2.VideoCapture(0)  # 0 = Default camera, 1 = External camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

    height, width, channels = frame.shape

    # Preprocessing the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Analyze the detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]  # The confidence scores for each class
            class_id = np.argmax(scores)  # Get the highest score class index
            confidence = scores[class_id]  # Confidence of the detection
            
            if confidence > 0.5:  # Filter weak detections
                # Get coordinates of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Calculate top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to reduce overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes on the frame
    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("YOLOv3 Real-Time Object Detection", frame)

    # Press 'q' to exit the real-time detection loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()