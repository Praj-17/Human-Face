import numpy as np
import argparse
import cv2

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",required =True,help = "1.png")
ap.add_argument("-p", "--prototxt",required =True,help = "deploy.prototxt.txt")
ap.add_argument("-m", "--model",required =True,help = "weights.caffemodel")
ap.add_argument("-c", "--confidence", type =float, default = 0.5, help = "0.5")
args = vars(ap.parse_args())
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
print(net)


