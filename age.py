from transformers import pipeline
import cv2
from PIL import Image
import time

# Load model
pipe = pipeline("image-classification", model="dima806/fairface_age_image_detection")

# Open webcam
cap = cv2.VideoCapture(0)
last_update_time = 0
update_interval = 2  # Update age every 2 seconds
current_predictions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current time
    now = time.time()

    # Update predictions only every `update_interval` seconds
    if now - last_update_time >= update_interval:
        # Convert frame to PIL image (for model)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Get top 2 predictions (sorted by confidence)
        current_predictions = pipe(pil_image)[:2]
        last_update_time = now

    # Display the live webcam feed
    display_frame = frame.copy()

    # Overlay the top 2 predictions (if available)
    if current_predictions:
        for i, pred in enumerate(current_predictions):
            age = pred["label"]
            prob = round(pred["score"] * 100, 2)
            text = f"{i+1}. Age: {age} ({prob}%)"
            
            # Position each prediction line below the previous one
            cv2.putText(
                display_frame,
                text,
                (20, 50 + i * 40),  # Adjust vertical spacing
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,  # Slightly smaller font
                (0, 255, 0),  # Green
                2,
                cv2.LINE_AA,
            )

    # Show the frame
    cv2.imshow("Webcam - Age Detection (Top 2)", display_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()