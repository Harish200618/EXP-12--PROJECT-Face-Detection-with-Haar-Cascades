# Exp 12- Face Detection using Haar Cascades with OpenCV and Matplotlib
## Name: Harish S
## Reg no: 212224240052
## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
model = cv2.imread('image_01.jpeg',0)
withglass = cv2.imread('image_02.jpeg',0)
group = cv2.imread('image_03.jpeg',0)
plt.figure(figsize=(20,10))
plt.subplot(131);plt.imshow(cv2.resize(model, (1000, 1000)),cmap='gray');plt.title("Model")
plt.subplot(132);plt.imshow(cv2.resize(withglass, (1000, 1000)),cmap='gray');plt.title("Model with glass")
plt.subplot(133);plt.imshow(cv2.resize(group, (1000, 1000)),cmap='gray');plt.title("Group")
plt.show()
```
## Cascade Files
```
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(
        f"Face cascade could not be loaded.\nPath: {face_cascade_path}"
    )

print("Face cascade loaded successfully!")
print(face_cascade_path)
```
## Face Detection Model with Glass
```
def detect_face(img):
    face_img = img.copy()

    # Convert to grayscale if the image is colored
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    face_rects = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in face_rects:
        cv2.rectangle(
            face_img,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            3
        )

    return face_img
result = detect_face(withglass)

plt.figure(figsize=(10, 8))
plt.imshow(result, cmap='gray')
plt.title("Face Detection - Model with Glass")
plt.axis('off')
plt.show()

```
## Face Detection
```
result = detect_face(model)

plt.figure(figsize=(10, 8))
plt.imshow(result, cmap='gray')
plt.title("Face Detection - Model")
plt.axis('off')
plt.show()
```
## Eye Detection - Model with Glass
```
def detect_eyes(img):
    face_img = img.copy()

    # Convert to grayscale
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    # First detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Detect eyes inside each detected face
    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                face_img,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (255, 255, 255),
                2
            )

    return face_img
result = detect_eyes(withglass)

plt.figure(figsize=(10, 8))
plt.imshow(result, cmap='gray')
plt.title("Eye Detection - Model with Glass")
plt.axis('off')
plt.show()
```
## Video Face Detection
```
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the camera. "
        "Check whether your webcam is connected or being used by another application."
    )

plt.ion()

fig, ax = plt.subplots(figsize=(10, 7))

ret, frame = cap.read()

if not ret:
    cap.release()
    plt.close(fig)
    raise RuntimeError("Could not read the first frame from the camera.")

frame = detect_face(frame)

im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
ax.set_title("Video Face Detection")
ax.axis('off')

while plt.fignum_exists(fig.number):

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame from camera.")
        break

    frame = detect_face(frame)

    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    plt.pause(0.01)

cap.release()
plt.ioff()
plt.close(fig)
```
## Output
## Original Image
<img width="440" height="418" alt="download" src="https://github.com/user-attachments/assets/3193166f-a09b-460b-ab6e-eac57e54d9b8" />

## Cascade Files
<img width="345" height="418" alt="download" src="https://github.com/user-attachments/assets/c6bd3ed1-3504-4677-8914-324e2ac23e92" />

## Face Detection Model with Glass
<img width="440" height="418" alt="download" src="https://github.com/user-attachments/assets/62ff992a-2278-4949-943d-d5e6235495f9" />

## Face Detection
<img width="345" height="418" alt="download" src="https://github.com/user-attachments/assets/9ba3b9d1-d6ce-48e5-9f10-a452952fe3d5" />

## Eye Detection - Model with Glass
<img width="440" height="418" alt="download" src="https://github.com/user-attachments/assets/bb03e26a-d15a-49f8-845c-de415daaf351" />

## Video Face Detection
<img width="549" height="433" alt="download" src="https://github.com/user-attachments/assets/327af3c1-a2b4-4fef-9f26-7580679c2e70" />


## Result :
Thus, to write a Python program using OpenCV to perform image manipulations for the given objectives is executed sucessfully.
