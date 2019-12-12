import cv2
import scenedetect

class FrameAnalysis(object):

    def __init__(self,path):
        pass

    def blurryness(self):
        return cv2.Laplacian(self.image, cv2.CV_64F).var()


class VideoAnalysis(object):
    pass
