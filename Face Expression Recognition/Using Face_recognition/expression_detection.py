from locale import currency
import cv2
from cv2 import VideoCapture
import face_recognition
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json



capture = VideoCapture(0)
all_face_locations = []

#Load the model and the weights
face_exp_model = model_from_json(open("model2.json","r").read())
face_exp_model.load_weights("model_weights.h5")
emotions_label = ('angry', 'disgust','fear','happy','sad','surprise','neutral')
while True:
#_________________________FACE_DETECTION__________________
    ret, current_frame = capture.read()
    
    all_face_locations = face_recognition.face_locations(current_frame,model='hog')
    #This is an optional step 
    current_frame_small= cv2.resize(current_frame,(0,0),fx =0.25, fy =0.25)
    #Fx and Fy are the scaling factors, 0,0 means we dont want to change its width and hieght(in proportion)
    
    
    
    for index, current_face_location in enumerate(all_face_locations):
        (top,right,bottom, left) = current_face_location
        # top = top   *4
        # right = right *4
        # bottom = bottom*4
        # left =  left  *4
        print(f"Found face {index} at top:{top},right:{right}, bottom:{bottom},left:{left}")
        print(current_frame)
       
        current_face_image = current_frame[top:bottom,left:right]
        print(current_face_image)
        cv2.rectangle(current_frame,(left,top), (right,bottom), (0,0,255),2)
#_________________________Image_PREPROCESSING_________________
       #Preprocessing inputm convert it to an image like as the data in dataset
        current_face_image = cv2.cvtColor(current_face_image,cv2.COLOR_BGR2GRAY)
        
        #resixing each image to (48,48) size since all the images trained the given model have been trained on the 48,48 sized images with grayscale
        current_face_image = cv2.resize(current_face_image,(48,48))
        #convert the PIL image into a 3D numpy array
        img_pixels = image.img_to_array(current_face_image)
        
        #expand the shape of an array to a single row multiple coloumns
        img_pixels = np.expand_dims(img_pixels,axis=0)
        
        #pixels are in range of [0,255]. normalise all pixels in scale of [0,1] 
        img_pixels/= 255
       #_________________________Prediction________________
       
        #do predictions from the model get the prediction values for all 7 expressions
       
        exp_predictions = face_exp_model.predict(img_pixels)
        print(exp_predictions)
        #find max indexed lable from emotions_label
        max_index = np.argmax(exp_predictions[0])
        #get corresponding lable from emotions_label
        emotion_label = emotions_label[max_index]
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,emotion_label,(left,bottom),font,0.5,(25,200,255),1)
        
    cv2.imshow("Webcam video", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()