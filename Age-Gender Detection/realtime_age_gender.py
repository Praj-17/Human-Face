import cv2
from cv2 import VideoCapture
from cv2 import putText
import face_recognition

capture = VideoCapture(0)
all_face_locations = []
gender_label_list = ['Male', 'Female']
age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)','(38-43)','(48-53)', '(60-100)']
gender_protext = "models\gender_deploy.prototxt"
gender_caffemodel = "models\gender_net.caffemodel"
age_protext = "models\\age_deploy.prototxt"
age_caffemodel = "models\\age_net.caffemodel"

while True:
    ret, current_frame = capture.read()
    
    
    #This is an optional step 
    current_frame_small = cv2.resize(current_frame,(0,0),fx =0.25, fy =0.25)
    #Fx and Fy are the scalin g factors, 0,0 means we dont want to change its width and hieght(in proportion)
    
    all_face_locations = face_recognition.face_locations(current_frame_small,model='hog')
    
    for index, current_face_location in enumerate(all_face_locations):
        (top,right,bottom, left) = current_face_location
        top    = top*4
        bottom = bottom*4
        left    = left*4
        right   = right*4
        print(f"Found face {index+1} at top:{top},right:{right}, bottom:{bottom},left:{left}")
        current_face_image = current_frame[top:bottom,left:right]
        
        #Age_Gender_MODEL_MEAN_VALUES calcultated by using the numpy .mean()
        MEAN_VALUES = (78.4263377603,87.7689143744,114.895847746)
        
        #Creating blob for face detection model
        current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227,227), MEAN_VALUES, swapRB = False)
        
        
        gender_cov_net = cv2.dnn.readNet(gender_caffemodel,gender_protext)
        gender_cov_net.setInput(current_face_image_blob)
        gender_predictions = gender_cov_net.forward()
        gender = gender_label_list[gender_predictions[0].argmax()]
        age_cov_net = cv2.dnn.readNet(age_caffemodel,age_protext)
        age_cov_net.setInput(current_face_image_blob)
        age_predictions = age_cov_net.forward()
        age = age_label_list[age_predictions[0].argmax()]
      
        cv2.rectangle(current_frame,(left,top), (right,bottom), (0,0,255),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2,putText(current_frame,gender +"" + age+ "yrs", (left,bottom),font,0.5, (0,255,0),1)
        cv2.imshow("Webcam video", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()