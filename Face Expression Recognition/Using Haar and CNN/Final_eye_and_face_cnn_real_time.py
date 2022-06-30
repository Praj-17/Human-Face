from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import face_recognition
import numpy as np

eye_classifier = cv2.CascadeClassifier(r'Face Expression Recognition\\Using Haar and CNN\\haarcascade_eye.xml')
classifier =load_model(r'model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

# Define function that will do detection
def detect_eye(gray, color):
  """ Input = greyscale image or frame from video stream
      Output = Image with rectangle box in the face
  """
  # Now get the tuples that detect the faces using above cascade

  # faces are the tuples of 4 numbers
  # x,y => upperleft corner coordinates of face
  # width(w) of rectangle in the face
  # height(h) of rectangle in the face
  # grey means the input image to the detector
  # 1.3 is the kernel size or size of image reduced when applying the detection
  # 5 is the number of neighbors after which we accept that is a face
  
  # Now iterate over the faces and detect eyes
  
    
  # we now need two region of interests(ROI) grey and color for eyes one to detect and another to draw rectangle
  # Detect eyes now
  eyes = eye_classifier.detectMultiScale(gray, 1.3, 5)
  if len(eyes) == 0:
      print("No eyes detected")
      return 0
  # Now draw rectangle over the eyes
  else:
   height , width =frame.size
   dimensions = (0, height, height+width, width)
  return dimensions

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces =face_recognition.face_locations(frame,model='cnn')
    
    if len(faces) == 0:
        detect_eye(gray, frame)

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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()