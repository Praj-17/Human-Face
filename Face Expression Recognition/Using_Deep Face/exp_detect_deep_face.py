import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
img  = cv2.imread("E:\CODING PLAYGROUND\CODE\Deep Leaning\Human Face\Gallery\happy.jfif")
# plt.imshow(img)
# plt.waitforbuttonpress()
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.waitforbuttonpress()
predictions = DeepFace.analyze(img)
print(predictions)
