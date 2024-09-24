import jetson.inference
import jetson.utils
import cv2
import numpy as np

# Initialize the object detection network
net = jetson.inference.detectNet(argv=['--model=frutas.onnx', '--labels=labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'])

# Initialize video source (camera) and output (display)
camera = jetson.utils.videoSource("/dev/video0")  # using a USB camera
display = jetson.utils.videoOutput()  # creating an output window

# Define Canny Edge Detection parameters
low_threshold = 50
high_threshold = 150

while display.IsStreaming():
    # Capture image from the camera
    img = camera.Capture()

    # Convert to numpy for OpenCV processing
    frame = jetson.utils.cudaToNumpy(img)

    # Convert the image to grayscale (Canny edge detection works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Convert edges back to color so we can overlay with the original
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Combine edges with the original image
    combined_img = cv2.addWeighted(frame, 0.8, edges_color, 0.2, 0)

    # Perform object detection
    detections = net.Detect(img)

    # Iterate over detections to display fruit name and confidence percentage inside the fruit
    for detection in detections:
        class_id = detection.ClassID
        confidence = detection.Confidence * 100  # confidence in percentage
        class_desc = net.GetClassDesc(class_id)  # Get the fruit name

        # Get the coordinates of the detected fruit
        x_left, y_top = int(detection.Left), int(detection.Top)
        x_right, y_bottom = int(detection.Right), int(detection.Bottom)

        # Calculate the center of the detected fruit area for placing the text
        x_center = (x_left + x_right) // 2
        y_center = (y_top + y_bottom) // 2

        # Draw the fruit name and confidence percentage inside the fruit
        text = f"{class_desc}: {confidence:.2f}%"

        # Use OpenCV to add text in the detected fruit area without drawing any bounding box
        font_scale = 1.0
        font_thickness = 2
        font_color = (0, 255, 0)  # Green color for text
        cv2.putText(combined_img, text, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

    # Convert back to CUDA for display
    cuda_output = jetson.utils.cudaFromNumpy(combined_img)

    # Render the image with the detections and edges
    display.Render(cuda_output)

    # Set the display status message
    display.SetStatus("Fruit Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

    # Break the loop if the camera or display is not streaming
    if not camera.IsStreaming() or not display.IsStreaming():
        break
