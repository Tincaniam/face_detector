import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm).
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in.
img = cv2.imread('mul_faces.jpg')

# Must convert to grayscale.
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around each face.
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#print(face_coordinates)
#
cv2.imshow('Tincaniam Face Detector', img)
cv2.waitKey()

print("Code Completed")
