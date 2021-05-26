import cv2
import numpy as np
from simple_pid import PID
import matplotlib.pyplot as plt
#from gpiozero import Servo
from time import sleep

#servo = 25
#myservo = Servo(servo)
plt.style.use('ggplot')

pid = PID(2, 0.0, 0.0, setpoint=0,output_limits=(-100, 100))
curveList = []
avgVal=10

errorsX =[]
errorsY =[]
correctionsX = []
correctionsY = []



def showCamera(cap):
    intialTrackbarVals = [256, 136, 216, 239]
    initializeTrackBar(intialTrackbarVals)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (683, 384))
        getLaneCurve(frame)
#INPUT
        cv2.imshow('Input', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def thresholding(img):
    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([0,0,255])
    upperWhite = np.array([0, 155, 255])
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    return maskWhite


def getLaneCurve(image):
    imgCopy = image.copy()
    imgResult = image.copy()
    imgThres = thresholding(image)
    hT, wT, c = image.shape
    points = valTrackbar()
    imgWarp = warpImg(imgThres, points, wT, hT)
    imgWarpPoints = drawPoints(imgCopy, points)
    middlePoint, imgHist = getHistogram(imgWarp,display=True,minPer=0.5,region=4)
    curveAveragePoint, imgHist = getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint-middlePoint

    curveList.append(curveRaw)
    if len(curveList)>avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
    print("Curva: ",curve)
    servoControll(curve)

    #### STEP 5



    cv2.imshow('Threshold',imgThres)
    cv2.imshow('Warp', imgWarp)
    cv2.imshow('Warp Points', imgWarpPoints)
    cv2.imshow('Histogram', imgHist)

    return None


def warpImg(img,points,w,h,inv = False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp

def empty(a):
    pass

def initializeTrackBar(initialTrackBarVals,wT=683,hT=384):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width top", "Trackbars", initialTrackBarVals[0], wT//2, empty)
    cv2.createTrackbar("Height top", "Trackbars", initialTrackBarVals[1], hT, empty)
    cv2.createTrackbar("Width bottom", "Trackbars", initialTrackBarVals[2], wT, empty)
    cv2.createTrackbar("Height bottom", "Trackbars", initialTrackBarVals[3], hT, empty)

def valTrackbar(wT=683, hT=240):
    widthTop = cv2.getTrackbarPos("Width top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height bottom", "Trackbars")
    #print("widthTop:", widthTop)
    #print("heightTop:", heightTop)
    #print("widthBottom:", widthBottom)
    #print("heightBottom:", heightBottom)
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop), (widthBottom, heightBottom), (wT-widthBottom, heightBottom)])
    return points

def drawPoints(img,points):
    for x in range(4):
        cv2.circle(img, (int(points[x][0]),int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return  img

def getHistogram(img,minPer=0.1,display = False, region=1 ):

    if region ==1:
        histValues = np.sum(img, axis=0)
    else:
        histValues = np.sum(img[img.shape[0]//region:,:], axis=0)


    #print(histValues)
    maxValue = np.max(histValues)
    minValue = minPer*maxValue

    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray))
    #print(basePoint)

    if display:
        imgHist= np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for x, intensity in enumerate(histValues):
            cv2.line(imgHist,(x, int(img.shape[0])),(x,int(img.shape[0]-intensity//255//region)),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint, imgHist
    return basePoint

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def servoControll(error):

    value = mapA(error,-100,100,-0.4,0.4)
    print("Correcao: ",pid(error))
    print("Output: ", value)
    # myservo.value = value


def ArduMap(value,fromMin,fromMax,toMin,toMax):
    result = (value*toMax)/fromMax
    return result

def mapA(x, in_min, in_max, out_min, out_max):
    valor = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    valor = round(valor,2)
    if valor > 0.4:
        return 0.4
    if valor < float(-0.4):
        return float(-0.4)
    else:
        return valor

