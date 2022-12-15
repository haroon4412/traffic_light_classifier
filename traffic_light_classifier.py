import numpy as np
import cv2

class TrafficLightClassifier:
    
    def __init__(self):
        self.kernel = np.ones((7, 7), np.float32)/49
        self.color = {
                0: "red",
                1: "yellow",
                2: "green",
                3: "background"
            }
    
    def classify(self, light):
        """ To classify light

        Args:
            light numpy array: Image of cropped light
        """
        img = np.std(light, axis = 2)
        img = img * min(10,(255/np.amax(img) ) - 1)
        img = cv2.filter2D(src=img, ddepth=-1, kernel=self.kernel)
        img2 = (img>75)*255
        h = img2.shape[0]
        edges = np.array([h/4,h/2,h/2+h/4])
        if len(np.argwhere(img2)[:,0]) < 3:
            probabilities = 3
        else:
            mean = np.median(np.argwhere(img2)[:,0])
            probabilities = np.argmin(abs(edges-mean))
        if np.std(np.argwhere(img2)[:,0]) > 5:
            img = np.max(light, axis = 2) 
            img = cv2.filter2D(src=img, ddepth=-1, kernel=self.kernel)
            img2 = (img>200)*255
            h = img2.shape[0]
            edges = np.array([h/4,h/2,h/2+h/4])
            if len(np.argwhere(img2)[:,0]) < 3:
                probabilities = 3
            else:
                mean = np.median(np.argwhere(img2)[:,0])
                probabilities = np.argmin(abs(edges-mean))
        return self.color[probabilities]