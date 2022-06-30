import cv2
import face_recognition
image= cv2.imread('1.jpg')
all_face_locations = face_recognition.face_locations(image,model = 'cnn')

print(f"There are {len(all_face_locations)} faces in the image")

for index, current_face_location in enumerate(all_face_locations):
    (top,right,bottom, left) = current_face_location
    print(f"Found face {index} at top:{top},right:{right}, bottom:{bottom},left:{left}")
    current_face_image = image[top:bottom,left:right]
    cv2.imshow('Face no'+ str(index+1),current_face_image)
    cv2.waitKey(0)
