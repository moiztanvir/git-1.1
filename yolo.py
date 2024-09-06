# from ultralytics import YOLO
# import cv2

# # Load the YOLOv8 model
# model = YOLO('yolov8s.pt')

# # Open the video capture
# cap = cv2.VideoCapture('c6.mp4')

# # Get the video properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Create a video writer to save the annotated video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('annotated_video.mp4', fourcc, fps, (width, height))

# # Loop through the video frames
# while True:
#     # Read a frame from the video
#     ret, frame = cap.read()
    
#     if not ret:
#         break
    
#     # Perform object detection on the frame
#     results = model(frame)
    
#     # Visualize the detection results
#     annotated_frame = results[0].plot()
    
#     # Write the annotated frame to the output video
#     out.write(annotated_frame)
    
#     # Display the annotated frame (optional)
#     # cv2_imshow('Object Detection', annotated_frame)
    
#     # Wait for the user to press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and video writer, and close all windows
# cap.release()
# out.release()
# cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Open the video capture
cap = cv2.VideoCapture('c6.mp4')

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a video writer to save the annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('annotated_video.mp4', fourcc, fps, (width, height))

# Function to calculate the dominant color in an ROI
def get_dominant_color(roi):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.reshape((roi.shape[0] * roi.shape[1], 3))
    unique, counts = np.unique(roi, axis=0, return_counts=True)
    dominant_color = unique[np.argmax(counts)]
    return tuple(dominant_color)

# Function to perform simple text detection (not accurate, for illustration purposes)
def detect_text_region(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform object detection on the frame
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Assuming class ID 0 corresponds to 'person'
                # Extract coordinates from the tensor and convert to integers
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                roi = frame[y1:y2, x1:x2]
                
                # Detect the dominant color in the bounding box
                dominant_color = get_dominant_color(roi)
                color_bgr = (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))  # Convert RGB to BGR
                
                # Detect text regions (for illustration purposes)
                text_regions = detect_text_region(roi)
                
                # Draw the bounding box with the detected color
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                
                # Draw detected text regions (not actual text, just regions)
                for contour in text_regions:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Annotate the frame with detected text and color information (dummy text)
                text_label = "Person"
                cv2.putText(frame, text_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)

    # Write the annotated frame to the output video
    out.write(frame)
    
    # Display the annotated frame (optional)
    # cv2.imshow('Object Detection', frame)
    
    # Wait for the user to press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and video writer, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
