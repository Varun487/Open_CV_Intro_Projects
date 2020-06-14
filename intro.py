import cv2
import numpy as np


def read_show_image(time):
    # showing an image
    img = cv2.imread("./Resources/lena.jpeg")
    cv2.imshow("Output", img)
    cv2.waitKey(time)


def use_web_cam():
    # Video capture web cam
    cap = cv2.VideoCapture(0)  # can give path to video to show the video, or the web cam Id, default Id is 0
    cap.set(3, 640)  # set width
    cap.set(4, 480)  # set height
    cap.set(10, 100)  # set brightness

    while True:
        success, img = cap.read()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def basic_functions():
    img = cv2.imread("./Resources/lena.jpeg")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (11, 11),
                               0)  # kernel size has to always be pairs of odd numbers, higher the numbers, higher the blur
    imgCanny = cv2.Canny(img, 100, 100)  # edge detector, higher the val, lower the no of edges
    # to increase thickness of edge - dilate
    kernel = np.ones((5, 5), np.uint8)
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)  # greater iterations, thicker edges
    imgEroded = cv2.erode(imgDilation, kernel, iterations=1)

    cv2.imshow("Original", img)
    cv2.imshow("Gray Scale", imgGray)
    cv2.imshow("Blur gray scale", imgBlur)
    cv2.imshow("Canny image", imgCanny)
    cv2.imshow("Dilation image", imgDilation)
    cv2.imshow("Eroded image", imgEroded)
    cv2.waitKey(0)


def change_shape_size():
    img = cv2.imread("./Resources/lena.jpeg")
    print(img.shape)

    imgResizeSmall = cv2.resize(img, (100, 100))  # width, height
    print(imgResizeSmall.shape)

    imgResizeBig = cv2.resize(img, (1000, 1000))  # width, height
    print(imgResizeBig.shape)

    imgCropped = img[0:100, 100:200]  # height, width

    cv2.imshow("image", img)
    cv2.imshow("image resized small ", imgResizeSmall)
    cv2.imshow("image resized big ", imgResizeBig)
    cv2.imshow("image Cropped ", imgCropped)
    cv2.waitKey(0)

def drawing_on_image():
    img = np.zeros((512, 512,3), np.uint8)

    # color whole image blue
    # img[:] = 255,0,0

    # color particular part blue
    img[100:200, 100:300] = 255,0,0

    # draw a line
    # img, starting pt, ending pt, color, thickness
    cv2.line(img, (0,0), (300,300), (0,255,0), 3)
    # can also use img dimensions - image.shape -> height, width, channels
    cv2.line(img, (0,0), (img.shape[1], img.shape[0]), (0,255,0), 3)

    # draw a rectangle
    cv2.rectangle(img, (0,0), (200,200), (0,0,255), cv2.FILLED) # can give line thickness number instead of filled

    # draw a circle
    cv2.circle(img, (400, 50), 30, (255,255,0), 5)

    # put text on image
    # image. text, starting pt, font, scale, color, thickness
    cv2.putText(img, " OPENCV ", (100,400), cv2.FONT_HERSHEY_COMPLEX, 1, (150,150,0))

    cv2.imshow("image", img)
    cv2.waitKey(0)

def warp_perspective():
    # need to try to isolate King of Spades and warp perspective
    img = cv2.imread("./Resources/cards.jpg")

    # approx width, height of upright card
    width, height = 250, 350

    # values of 4 edge points of King of spades
    pts1 = np.float32([[111,219], [287, 188], [154, 482], [352, 440]])
    pts2 = np.float32([[0,0], [width,0], [0,height], [width,height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgFinal = cv2.warpPerspective(img, matrix, (width, height))

    cv2.imshow("Playing Cards", img)
    cv2.imshow("Playing Cards Final", imgFinal)
    cv2.waitKey(0)


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

def using_stacking():
    img = cv2.imread("./Resources/lena.jpeg")
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    imgHor = np.hstack((img,img))
    imgVer = np.vstack((img,img))

    # using img stack function - allows to resize and use different channels of colors
    # scale, tuple -> arrays of how images should be stacked
    imgStack = stackImages(1,([img,imgGray,img],[img,img,img]))

    cv2.imshow("Horizontal stacked image", imgHor)
    cv2.imshow("Vertical stacked image", imgVer)
    cv2.imshow("ImageStack",imgStack)

    cv2.waitKey(0)


def detecting_colors():

    def empty(a):
        pass

    # detect orange color in the image
    path = "./Resources/lambo.png"
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 240)

    # original trackbars - original
    # cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
    # cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, empty)
    # cv2.createTrackbar("Sat Min", "Trackbars", 0, 255, empty)
    # cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
    # cv2.createTrackbar("Val Min", "Trackbars", 0, 255, empty)
    # cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

    # trackbars created after testing, will get the mask of the car by default
    cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "Trackbars", 19, 179, empty)
    cv2.createTrackbar("Sat Min", "Trackbars", 110, 255, empty)
    cv2.createTrackbar("Sat Max", "Trackbars", 240, 255, empty)
    cv2.createTrackbar("Val Min", "Trackbars", 153, 255, empty)
    cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

    while True:
        img = cv2.imread(path)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # to get the values where orange is white and rest all are black
        hue_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
        hue_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
        sat_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
        sat_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
        val_min = cv2.getTrackbarPos("Val Min", "Trackbars")
        val_max = cv2.getTrackbarPos("Val Max", "Trackbars")

        lower = np.array([hue_min, sat_min, val_min])
        upper = np.array([hue_max, sat_max, val_max])
        mask = cv2.inRange(imgHSV, lower, upper)

        # creating new image with orange using bitwise and
        imgResult = cv2.bitwise_and(img, img, mask=mask)

        # stacking the individual images
        imgStack =  stackImages(0.6, ([img, imgHSV], [mask, imgResult]))

        # cv2.imshow("Original", img)
        # cv2.imshow("HSV", imgHSV)
        # cv2.imshow("Mask HSV", mask)
        # cv2.imshow("Resulting image", imgResult)
        cv2.imshow("Final comparision", imgStack)
        cv2.waitKey(1)

def detect_shapes():
    # classify the shapes, find points, area of shape
    path = "./Resources/shapes.png"
    img = cv2.imread(path)

    imgBlank = np.zeros_like(img)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    imgContour = img.copy()

    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        print("Area of shape",area)
        # to remove extra noise
        if area > 500:
            cv2.drawContours(imgContour, contour, -1, (0,200,100), 3)
            perimeter = cv2.arcLength(contour, True)
            print("Perimeter of shape", perimeter)
            approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
            print("Number of approx corners",len(approx))
            objectCorners = len(approx)
            x,y,w,h = cv2.boundingRect(approx)

            shape = "None"

            if objectCorners == 3:
                shape = "Triangle"
            elif objectCorners == 4:
                aspRatio = w/float(h)
                if aspRatio>0.95 and aspRatio<1.05:
                    shape = "Square"
                else:
                    shape = "Rectangle"
            elif objectCorners > 4:
                shape = "Circle"
            cv2.rectangle(imgContour, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(imgContour, shape, (x, (y+h+10)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

    imgStack = stackImages(0.7, ([img, imgGray, imgBlur] ,[imgCanny, imgContour, imgBlank]))

    cv2.imshow("Stacked image", imgStack)

    cv2.waitKey(0)

def face_detect():
    faceCascade = cv2.CascadeClassifier('./Resources/haarcascade_frontalface_default.xml')
    img = cv2.imread('./Resources/lena.jpeg')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray,1.1,4)

    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(img, "Lena", (x,y+h+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)

    imgBlank = np.zeros_like(img)

    imgStack = stackImages(1, ([img, imgGray], [imgBlank, imgBlank]))

    cv2.imshow("Result", imgStack)
    cv2.waitKey(0)

# reads image in resources and shows it for time given in milliseconds, can give 0 to show indefinitely
# read_show_image(2000)

# opens up the web cam, runs till you press q key
# use_web_cam()

# basic manipulations of images in opencv, press any key to exit
# basic_functions()

# to change the shape and size of an image
# change_shape_size()

# drawing on an image
# drawing_on_image()

# warp perspective
# warp_perspective()

# how to stack and join images
# using_stacking()

# detects and separates a color
# detecting_colors()

# detect shapes
# detect_shapes()

# detect faces
# face_detect()
