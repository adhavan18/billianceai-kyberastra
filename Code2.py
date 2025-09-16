import os
import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import base64
from collections import defaultdict
from email import encoders

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from mimetypes import guess_type as guess_mime_type

# Load the YOLOv8 model
model = YOLO(r"A:\Shoplifting-Detection\Shoplifting-Detection\best2.pt")

# Open the video file
video_path = r"A:\Shoplifting-Detection\Shoplifting-Detection\CCTV 2_CAM 12_main_20241025165730.dav"
cap = cv2.VideoCapture(video_path)

# Get video properties for saving
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define variables to save multiple frames of shoplifting image
shoplifting_images_dir = "shoplifting_frames"
os.makedirs(shoplifting_images_dir, exist_ok=True)
captured_frames = []
max_frames = 5  # Number of frames to capture

# Store the track history and confidence scores
track_history = defaultdict(lambda: [])
confidence_history = defaultdict(lambda: None)  # To store previous confidence scores

# Confidence drop threshold for detecting shoplifting!!!!!!!!!!!!!!!!!
CONFIDENCE_DROP_THRESHOLD = 0.3

# Loop through the video frames
while cap.isOpened() and len(captured_frames) < max_frames:
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        confidences = results[0].boxes.conf.cpu().tolist()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(boxes)

        annotated_frame = results[0].plot()

        shoplifting_detected = False

        for box, confidence, track_id in zip(boxes, confidences, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            if confidence_history[track_id] is not None:
                previous_confidence = confidence_history[track_id]
                confidence_drop = previous_confidence - confidence

                if confidence_drop > CONFIDENCE_DROP_THRESHOLD:
                    color = (0, 0, 255)
                    label = "Shoplifting Detected!"
                    shoplifting_detected = True
                else:
                    color = (0, 255, 0)
                    label = "Tracking"
            else:
                color = (0, 255, 0)
                label = "Tracking"

            confidence_history[track_id] = confidence

            cv2.rectangle(annotated_frame,
                          (int(x - w / 2), int(y - h / 2)),
                          (int(x + w / 2), int(y + h / 2)),
                          color, 2)
            cv2.putText(annotated_frame, label, (int(x - w / 2), int(y - h / 2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        if shoplifting_detected:
            frame_filename = os.path.join(shoplifting_images_dir, f"frame_{len(captured_frames)}.jpg")
            cv2.imwrite(frame_filename, annotated_frame)
            captured_frames.append(frame_filename)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Captured {len(captured_frames)} frames with shoplifting detection.")

# Create a collage of the captured frames
def create_collage(image_paths, collage_path, rows, cols):
    images = [cv2.imread(img) for img in image_paths]
    if not images:
        return

    img_h, img_w, _ = images[0].shape
    collage = np.zeros((img_h * rows, img_w * cols, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        collage[row * img_h: (row + 1) * img_h, col * img_w: (col + 1) * img_w] = img

    cv2.imwrite(collage_path, collage)

collage_path = "shoplifting_collage.jpg"
create_collage(captured_frames, collage_path, rows=2, cols=3)

print(f"Collage created and saved at: {collage_path}")

# Gmail API setup change mail here!!!!!!!!!!!!!!!!!
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
OUR_EMAIL = 'hamsavardhini.m@gmail.com'

def gmail_authenticate():
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('Billiance_Gmail_API_Cred.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

def create_message_with_attachment(to, subject, body, file_path):
    message = MIMEMultipart()
    message['to'] = to
    message['from'] = OUR_EMAIL
    message['subject'] = subject

    msg_body = MIMEText(body)
    message.attach(msg_body)

    content_type, encoding = guess_mime_type(file_path)
    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
    
    main_type, sub_type = content_type.split('/', 1)

    with open(file_path, 'rb') as f:
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(f.read())
    
    encoders.encode_base64(msg)

    filename = os.path.basename(file_path)
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    
    message.attach(msg)

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw_message}

def send_message(service, user_id, message):
    try:
        sent_message = service.users().messages().send(userId=user_id, body=message).execute()
        print(f"Message sent successfully: {sent_message['id']}")
    except Exception as error:
        print(f"An error occurred: {error}")
 
def send_email():
    service = gmail_authenticate()
    email_message = create_message_with_attachment(
        to="vaikundamanig01@gmail.com",  # Replace with recipient email here !!!!!!!!!!!!!!!!!!!
        subject="Shoplifting Detected - Collage",
        body="Please find attached  collage of detected shoplifting incidents.",
        file_path=collage_path
    )
    send_message(service, "me", email_message)

# Call the send_email function to send the collage
send_email()
