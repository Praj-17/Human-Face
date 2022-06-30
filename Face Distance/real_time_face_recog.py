import cv2
import face_recognition

capture = cv2.VideoCapture(0)
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
#loop through every frome in the image
while True:
    ret, current_frame = capture.read()
    #Resize the current image to 1/4 size for faster processing
    current_frame_small = cv2.resize(current_frame, (0,0), fx= 0.25, fy =0.25)
    

    #detect all faces in the image
    #arguments are image, no of time to unsaple and the model to be loaded
    all_face_locations = face_recognition.face_locations(current_frame_small,model = 'hog',number_of_times_to_upsample=1)
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)
    all_face_names = []
    for current_face_location, current_face_encoding in zip(all_face_locations,all_face_encodings):
        (top,right,bottom, left) = current_face_location
        # print(f"Found face {index} at top:{top},right:{right}, bottom:{bottom},left:{left}")
        
        top   *=4
        right *=4
        bottom*=4
        left  *=4
        
        all_matches = face_recognition.compare_faces(known_face_encodings,current_face_encoding)
        print(all_matches)
        
        #string to hold the label
        name_of_person = 'Unknown_face'
        #Check if the all_matches at least one time
        #if yes get the index number of the face that is in the first index of all matches
        if True in all_matches:
            first_match_index = all_matches.index(True)
            print(first_match_index)
            name_of_person = known_face_names[first_match_index]
        #Draw rectangle around the code
        cv2.rectangle(current_frame,(left,top), (right,bottom),(255,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(current_frame,name_of_person, (left,bottom), font, 0.5, (255,255,0),1 )
        # current_face_image = image[top:bottom,left:right]
        
    cv2.imshow('Face no',current_frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
            break
capture.release()
cv2.destroyAllWindows() 