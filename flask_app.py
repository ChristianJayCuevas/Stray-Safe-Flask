import cv2
import torch
import time
from flask import Flask, render_template, Response
from ultralytics import RTDETR

app = Flask(__name__)

# Load the RTDETR model
model = RTDETR(r'D:\Project Design - Web App\StrayFlask\best.pt')
model.conf = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} for inference.")
model.to(device)

# Define your RTSP stream URL
rtsp_url = "rtsp://ADMIN:12345@192.168.1.5:554/cam/realmonitor?channel=2&subtype=0"

imgsz = (960, 544)

def initialize_video_capture(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream. Check the URL or camera connection.")
    return cap

def generate_frames():
    while True:
        cap = initialize_video_capture(rtsp_url)
        if not cap.isOpened():
            time.sleep(5)  # Wait before retrying
            continue

        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from RTSP stream. Restarting capture...")
                cap.release()
                time.sleep(5)  # Wait before retrying
                break

            # Resize the frame
            resized_frame = cv2.resize(frame, imgsz)

            # Run object detection
            with torch.no_grad():
                results = model.predict(resized_frame, imgsz=imgsz)

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
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=ssl_context)
