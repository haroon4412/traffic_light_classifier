from traffic_light_classifier import TrafficLightClassifier
import os
import cv2

dataset = os.listdir('test')
tl_classifier = TrafficLightClassifier()

for cat in dataset:
    image_names = os.listdir(os.path.join("test", cat))
    for image in image_names:
        img = cv2.imread(os.path.join("test", cat, image))
        result = tl_classifier.classify(img)
        print(cat, result)
