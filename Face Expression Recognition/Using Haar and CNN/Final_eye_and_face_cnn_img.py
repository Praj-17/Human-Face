from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import face_recognition
import numpy as np

eye_classifier = cv2.CascadeClassifier(r'Face Expression Recognition\\Using Haar and CNN\\haarcascade_eye.xml')
classifier =load_model(r'model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


# Define function that will do detection
def detect_eye(color):
  gray = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
  print(gray)
  eyes = eye_classifier.detectMultiScale(gray, 1.3, 5)
  if len(eyes) == 0:
      print("No Faces detected")
      return 0
  # Now draw rectangle over the eyes
  else:
   height , width =frame.size
   dimensions = (0, height, height+width, width)
  return dimensions


# frame = cv2.imread("E:\\CODING PLAYGROUND\\CODE\\Deep Leaning\\Human Face\\Gallery\\1.jpeg")
frame = cv2.imread("2.jpeg")

faces =face_recognition.face_locations(frame,model='hog')
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
print(len(faces))

if len(faces) == 0:faces = detect_eye(frame)

for (x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    roi_gray = gray[y:y+h,x:x+w]
    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



    if np.sum([roi_gray])!=0:
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)

        prediction = classifier.predict(roi)[0]
        label=emotion_labels[prediction.argmax()]
        label_position = (x,y)
        print(label)
        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    else:
        cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
cv2.imshow('Emotion Detector',frame)
cv2.waitkey(0)
cv2.destroyAllWindows()