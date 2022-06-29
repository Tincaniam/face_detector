import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm).
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# From video capture source.
video_source = cv2.VideoCapture('video_preview_h264.mp4')

# Iterate infinitely over frames.
while True:
    # Read current frame.
    successful_frame_read, frame = video_source.read()

    # Must convert to grayscale.
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # Draw rectangles around each face.
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Tincaniam Face Detector', frame)

    # Wait 1 millisecond before continuing.
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

    # Release the VideoCapture object
    video_source.release

print("Code completed.")
