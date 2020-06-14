from intro import *

# Video capture web cam
cap = cv2.VideoCapture(0)  # can give path to video to show the video, or the web cam Id, default Id is 0
cap.set(3, 640)  # set width
cap.set(4, 480)  # set height
cap.set(10, 100)  # set brightness


def get_contours(img, mask):
    imgContour = img.copy()
    x, y, w, h = 0, 0, 0, 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        #print("Area of shape",area)
        # to remove extra noise
        if area > 500:
            cv2.drawContours(imgContour, contour, -1, (255,0,0), 3)
            perimeter = cv2.arcLength(contour, True)
            #print("Perimeter of shape", perimeter)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            #print("Number of approx corners", len(approx))
            #objectCorners = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            #cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return (imgContour, x+w//2, y)

def empty(a):
    pass


cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)

# original trackbars - original
cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "Trackbars", 56, 179, empty)
cv2.createTrackbar("Sat Min", "Trackbars", 94, 255, empty)
cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
cv2.createTrackbar("Val Min", "Trackbars", 198, 255, empty)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

green_color_points = []
orange_color_points = []

while True:
    success, img = cap.read()

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # to get the values where orange is white and rest all are black
    hue_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
    hue_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
    sat_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
    sat_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
    val_min = cv2.getTrackbarPos("Val Min", "Trackbars")
    val_max = cv2.getTrackbarPos("Val Max", "Trackbars")

    imgBlank = np.zeros_like(img)

    img_result = img.copy()

    lower_orange = np.array([0, 94, 198])
    upper_orange = np.array([56, 255, 255])
    mask_orange = cv2.inRange(imgHSV, lower_orange, upper_orange)

    lower_green = np.array([61, 15, 0])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(imgHSV, lower_green, upper_green)

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    mask = cv2.inRange(imgHSV, lower, upper)

    # creating new image with orange using bitwise and
    imgMask = cv2.bitwise_and(img, img, mask=mask)
    img_orange = cv2.bitwise_and(img, img, mask=mask_orange)
    img_green = cv2.bitwise_and(img, img, mask=mask_green)

    img_green_contour,x_green,y_green = get_contours(img,mask_green)
    img_orange_contour, x_orange, y_orange = get_contours(img, mask_orange)

    if x_green!=0 and y_green!=0:
        green_color_points.append((x_green, y_green))

    for green_point in green_color_points:
        cv2.circle(img_result, (green_point[0], green_point[1]), 10, (0, 255, 0), cv2.FILLED)

    if x_orange!=0 and y_orange!=0:
        orange_color_points.append((x_orange, y_orange))

    for orange_point in orange_color_points:
        cv2.circle(img_result, (orange_point[0], orange_point[1]), 10, (0, 165, 255), cv2.FILLED)

    imgStack = stackImages(0.7, ([img, imgHSV, img_orange, img_green], [img_green_contour, img_orange_contour, img_result, imgBlank]))

    cv2.imshow("Video", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
