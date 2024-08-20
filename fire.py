import cv2
import torch
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('.', 'custom', path='yolov5s.pt', source='local')

# Initialize the camera (or specify a video file path)
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

def is_fire_color(roi):
    # Convert the region of interest (ROI) from BGR to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define the color ranges for detecting fire (red and orange)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    # Create masks for the defined color ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine the masks to detect fire colors
    mask = mask1 + mask2 + mask3
    
    # Check if a significant area of the ROI contains fire colors
    return cv2.countNonZero(mask) > 0.05 * roi.size

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the captured frame
    frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Apply Gaussian blur to reduce noise
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)  # Adjust brightness and contrast

    # Perform object detection
    results = model(frame)

    # Process and display the detection results
    for det in results.xyxy[0]:  # Iterate through detected objects
        x1, y1, x2, y2, conf, cls = det.tolist()  # Extract bounding box coordinates and confidence
        if conf > 0.2:  # Set a lower confidence threshold for detection
            roi = frame[int(y1):int(y2), int(x1):int(x2)]  # Define the region of interest
            if is_fire_color(roi):  # Check if the ROI contains fire colors
                label = f"Potential Fire: {conf:.2f}"  # Create label for detected fire
                # Draw a rectangle around the detected fire
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # Put the label text above the rectangle
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print("Potential fire detected! Alert!")  # Print alert message

    # Display the processed frame
    cv2.imshow('Fire Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
