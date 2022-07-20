import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
scaling_factor = 0.5
frame = cv2.imread("z1.jpg")
frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in face_rects:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

cv2.imshow("z1.jpg", frame)
cv2.waitKey(0)
print(f'Found {len(face_rects)} faces!')

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

frame = cv2.imread("z1.jpg")
frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
smile_rects = smile_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=80)
eye_rects = eye_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in smile_rects:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
for (x, y, w, h) in eye_rects:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
for (x, y, w, h) in face_rects:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

cv2.imshow("z1.jpg", frame)
cv2.waitKey(0)
print(f'Found {len(smile_rects)} smiles!')
print(f'Found {len(eye_rects)} eyes!')
print(f'Found {len(face_rects)} faces!')
print("\n\n\n")




face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
scaling_factor = 0.5
frame = cv2.imread("z3.jpg")
frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in face_rects:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

cv2.imshow("z3.jpg", frame)
cv2.waitKey(0)
print(f'Found {len(face_rects)} faces!')

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

frame = cv2.imread("z3.jpg")
frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
smile_rects = smile_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=7)
eye_rects = eye_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=2)

for (x, y, w, h) in smile_rects:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
for (x, y, w, h) in eye_rects:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
for (x, y, w, h) in face_rects:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

cv2.imshow("z3.jpg", frame)
cv2.waitKey(0)
print(f'Found {len(smile_rects)} smiles!')
print(f'Found {len(eye_rects)} eyes!')
print(f'Found {len(face_rects)} faces!')



hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    gray_filter = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
    for (xa, ya, xb, yb) in boxes:
        cv2.rectangle(frame, (xa, ya), (xb, yb), (255, 0, 0), 1)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

cv2.startWindowThread()
cap = cv2.VideoCapture('video2.mp4')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    gray_filter = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    for (xa, ya, xb, yb) in boxes:
        cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 0), 1)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    print(len(boxes))
cap.release()
cv2.destroyAllWindows()
