import cv2
import face_recognition


#load the sample image and get the 128 face embeddings that is vecotrs from them
praj_image= face_recognition.load_image_file('1.jpg')
modi_image= face_recognition.load_image_file('modi2.jpg')
trump_image= face_recognition.load_image_file('trump.jpg')

#here we are assuming that the image is having only a single face
face_encodings_praj = face_recognition.face_encodings(praj_image)[0]
face_encodings_modi = face_recognition.face_encodings(modi_image)[0]
face_encodings_trump = face_recognition.face_encodings(trump_image)[0]

known_face_encodings = [face_encodings_praj,face_encodings_modi, face_encodings_trump]
known_face_names = ["Prajwal", "Narendra Modi", "Donald Trump"]

#load the unknown image to recognise faces in it
img_to_detect = cv2.imread('modi.jpg')
original_img = img_to_detect

#detect the encodings of the current image
recognize_image_encoddings = face_recognition.face_encodings(img_to_detect)[0]

face_distances = face_recognition.face_distance(known_face_encodings,recognize_image_encoddings)

for i ,face_distance in enumerate(face_distances):
    print("The claculated face distance is {:.2} from sample image {}".format(face_distance,known_face_names[i]))
    


