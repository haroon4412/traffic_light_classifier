# Traffic light classifier

A simple code to classify color of a traffic light.


## How to run

Import the TrafficLightClassifer class as shown below

```bash
  import cv2
  from traffic_light_classifier import TrafficLightClassifier

  tl_classifier = TrafficLightClassifier()

  image = cv2.imread("./test/green/checkg2.jpg")
  result = tl_classifier.classify(image)

  print(result)
```
