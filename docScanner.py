from intro import *

img = cv2.imread("./Resources/page.jpeg")
img_contours = img.copy()

def get_contours(imgThresh):
    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 0
    biggest = []
    print(len(contours))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            #cv2.drawContours(img_contours, contour, -1, (255,0,0), 3)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(img_contours, biggest, -1, (250, 0, 0), 50)
    return biggest

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (11, 11), 0)
imgCanny = cv2.Canny(img, 300, 300)
kernel = np.ones((5,5))
imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
imgThres = cv2.erode(imgDial, kernel, iterations=1)
biggest = get_contours(imgThres)

print(biggest)
print(img.shape)

width = img.shape[1]
height = img.shape[0]

print(biggest.shape)

pts1 = np.float32(biggest)
pts2 = np.float32([[0,0], [width,0], [0,height], [width,height]])
pts2 = np.float32([[width,0], [0,0], [0,height], [width,height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgFinal = cv2.warpPerspective(img, matrix, (width, height))

imgStack = stackImages(0.25, ([img, imgGray, imgBlur, imgCanny], [imgDial, imgThres, img_contours, imgFinal]))

cv2.imshow("steps", imgStack)
cv2.imshow("page", imgFinal)
cv2.waitKey(0)
