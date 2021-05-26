import cv2
import utils

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    utils.showCamera(cap)