import face_recognition
from PIL import Image, ImageDraw
from scipy.misc import face


#load the image file

image = face_recognition.load_image_file('modi2.jpg')

#get the landmarks
face_landmarks = face_recognition.face_landmarks(image)
print(face_landmarks)

for face_landmark in face_landmarks:
    pil_image = Image.fromarray(image)
    drawing = ImageDraw.Draw(pil_image)
    drawing.line(face_landmark['chin'],fill = (0,255,0),width =3)
    drawing.line(face_landmark['left_eyebrow'],fill = (0,255,0),width =3)
    drawing.line(face_landmark['right_eyebrow'],fill = (0,255,0),width =3)
    drawing.line(face_landmark['nose_bridge'],fill = (0,255,0),width =3)
    drawing.line(face_landmark['nose_tip'],fill = (0,255,0),width =3)
    drawing.line(face_landmark['left_eye'],fill = (0,255,0),width =3)
    drawing.line(face_landmark['right_eye'],fill = (0,255,0),width =3)
    drawing.line(face_landmark['top_lip'],fill = (0,255,0),width =3)
    drawing.line(face_landmark['bottom_lip'],fill = (0,255,0),width =3)
pil_image.show()
pil_image.save("edited.jpg")