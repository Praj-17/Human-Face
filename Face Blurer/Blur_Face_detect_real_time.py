import cv2
from cv2 import VideoCapture
import face_recognition

capture = VideoCapture(0)
all_face_locations = []

while True:
    ret, current_frame = capture.read()
    
    
    #This is an optional step 
    current_frame_small = cv2.resize(current_frame,(0,0),fx =0.25, fy =0.25)
    #Fx and Fy are the scaling factors, 0,0 means we dont want to change its width and hieght(in proportion)
    
    all_face_locations = face_recognition.face_locations(current_frame_small,model='hog')
    
    for index, current_face_location in enumerate(all_face_locations):
        (top,right,bottom, left) = current_face_location
        print(f"Found face {index} at top:{top},right:{right}, bottom:{bottom},left:{left}")
        current_face_image = current_frame_small[top:bottom,left:right]
        top    = top*4
        bottom = bottom*4
        left    = left*4
        right   = right*4
        print(f"Found face {index+1} at top:{top},right:{right}, bottom:{bottom},left:{left}")
        current_face_image = current_frame[top:bottom,left:right]
        #Blur the sized face and save it to the same array itself
        current_face_image = cv2.GaussianBlur(current_face_image,(99,99),30)
        #paste the blured face into the actual array
        current_frame[top:bottom,left:right] = current_face_image
        cv2.rectangle(current_frame,(left,top), (right,bottom), (0,0,255),2)
        cv2.imshow("Webcam video", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()