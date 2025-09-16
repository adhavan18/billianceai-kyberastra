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

# Store the track history and confidence scores
track_history = defaultdict(lambda: [])
confidence_history = defaultdict(lambda: None)  # To store previous confidence scores

# Confidence drop threshold for detecting shoplifting
CONFIDENCE_DROP_THRESHOLD = 0.29

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
OUR_EMAIL = 'hamsavardhini.m@gmail.com'

# Gmail authentication function
def gmail_authenticate():
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('A:\Shoplifting-Detection\Shoplifting-Detection\Billiance_Gmail_API_Cred.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

# Function to create email with attachment
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

# Function to send email
def send_message(service, user_id, message):
    try:
        sent_message = service.users().messages().send(userId=user_id, body=message).execute()
        print(f"Message sent successfully: {sent_message['id']}")
    except Exception as error:
        print(f"An error occurred: {error}")

# Function to send an email with the detected frame
def send_email(frame_filename):
    service = gmail_authenticate()
    email_message = create_message_with_attachment(
        to="vaikundamanig01@gmail.com",  # Updated recipient email
        subject="Shoplifting Detected - Frame",
        body="Shoplifting has been detected. Please find the attached image .", 
        file_path=frame_filename
    )
    send_message(service, "me", email_message)

# Loop through the video frames and detect shoplifting
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()

    if success:
        frame_count += 1
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        confidences = results[0].boxes.conf.cpu().tolist()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(boxes)

        annotated_frame = results[0].plot()

        for box, confidence, track_id in zip(boxes, confidences, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            if confidence_history[track_id] is not None:
                previous_confidence = confidence_history[track_id]
                confidence_drop = previous_confidence - confidence

                # If confidence drop exceeds the threshold, send the frame as email
                if confidence_drop > CONFIDENCE_DROP_THRESHOLD:
                    color = (0, 0, 255)
                    label = "Shoplifting Detected!"
                    
                    # Save the detected frame
                    frame_filename = f"shoplifting_frame_{frame_count}.jpg"
                    cv2.imwrite(frame_filename, annotated_frame)
                    
                    # Send the frame via email
                    send_email(frame_filename)
                    
                else:
                    color = (0, 255, 0)
                    label = "Tracking"
            else:
                color = (0, 255, 0)
                label = "Tracking"

            confidence_history[track_id] = confidence

            # Draw the bounding box and label on the frame
            cv2.rectangle(annotated_frame,
                          (int(x - w / 2), int(y - h / 2)),
                          (int(x + w / 2), int(y + h / 2)),
                          color, 2)
            cv2.putText(annotated_frame, label, (int(x - w / 2), int(y - h / 2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Draw the tracking line
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        # Display the video frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

print("Video processing completed.")