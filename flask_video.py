import cv2
import torch
import time
import requests
import base64
import threading
from flask import Flask, render_template, Response
from ultralytics import RTDETR

app = Flask(__name__)

# Load the RTDETR model
model = RTDETR(r'D:\Project Design - Web App\StrayFlask\best.pt')
model.conf = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} for inference.")
model.to(device)

# Define your video input source
video_source = r"D:\CCTV.mp4"  # Change to your video file path

imgsz = (960, 544)

# Detection tracking variables
consecutive_detections = 0
required_consecutive_frames = 20
highest_confidence_frame = None
highest_confidence_score = 0

# Initialize video capture
cap = cv2.VideoCapture(video_source)

def initialize_video_capture():
    global cap
    if not cap.isOpened():
        print("Error: Unable to open video source. Check the file path or connection.")
        cap = cv2.VideoCapture(video_source)

def encode_image_to_base64(image):
    """Encode an image to a Base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def generate_frames():
    global consecutive_detections, highest_confidence_frame, highest_confidence_score

    initialize_video_capture()
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from video source. Restarting capture...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(1)
            continue

        # Resize the frame
        resized_frame = cv2.resize(frame, imgsz)

        # Run object detection
        with torch.no_grad():
            results = model.predict(resized_frame, imgsz=imgsz)

        detected = False

        # Check for detected objects
        for result in results:
            for det in result.boxes.data:
                class_id = int(det[-1])
                confidence = float(det[-2])
                x1, y1, x2, y2 = map(int, det[:4])  # Get bounding box coordinates (x1, y1, x2, y2)

                # Ensure the coordinates are within frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(resized_frame.shape[1], x2)
                y2 = min(resized_frame.shape[0], y2)

                if class_id in [0, 1]:  # Assuming 1 = dog, 0 = cat
                    detected = True

                    # Update the highest confidence frame with the cropped image
                    if confidence > highest_confidence_score:
                        highest_confidence_score = confidence
                        cropped_image = resized_frame[y1:y2, x1:x2]
                        highest_confidence_frame = cropped_image

        if detected:
            consecutive_detections += 1

            # Check if we have reached the required number of consecutive detections
            if consecutive_detections >= required_consecutive_frames:
                if highest_confidence_frame is not None:
                    # Encode the cropped image to Base64
                    base64_image = encode_image_to_base64(highest_confidence_frame)
                    print("Cropped snapshot encoded as Base64.")

                # Send the request to pin the detected animal on the map
                try:
                    response = requests.post('http://127.0.0.1:8000/api/pin', json={
                        'animal_type': 'dog' if class_id == 1 else 'cat',
                        'coordinates': [121.039295, 14.631141],
                        'snapshot': base64_image
                    }, verify=False)

                    if response.status_code == 200:
                        print(f"Successfully pinned {'dog' if class_id == 1 else 'cat'} on the map.")
                    else:
                        print(f"Failed to pin animal: {response.status_code}")
                except Exception as e:
                    print(f"Error sending request to pin animal: {e}")

                # Reset counters after sending the request
                consecutive_detections = 0
                highest_confidence_frame = None
                highest_confidence_score = 0

        else:
            # Reset the detection counter if no object is detected
            consecutive_detections = 0
            highest_confidence_frame = None
            highest_confidence_score = 0

        # Annotate the frame
        annotated_frame = results[0].plot()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(annotated_frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            print("Error: Failed to encode frame.")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def start_frame_generation():
    """Start frame generation in a background thread."""
    threading.Thread(target=lambda: list(generate_frames()), daemon=True).start()

# Route to serve the video stream
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to display the webpage
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    ssl_context = (r'D:\Project Design - Web App\StrayFlask\ssl\selfsigned.crt',
                   r'D:\Project Design - Web App\StrayFlask\ssl\selfsigned.key')
    # Start frame generation in the background
    start_frame_generation()
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=ssl_context)
