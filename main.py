import cv2
import imutils
import numpy as np


if __name__ == '__main__':
    img = cv2.imread("got.jpg")
    cv2.imwrite("got2.jpg", img)

    roi = img[60:160, 320:420]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

    resized = cv2.resize(img, (200, 200))
    cv2.imshow("Resized", resized)
    cv2.waitKey(0)

    h, w = img.shape[0:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -45, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)

    blurred = cv2.blur(img, ksize=(10, 10))
    cv2.imshow("Blurred", blurred)
    cv2.waitKey(0)

    resized = imutils.resize(img, width=460)
    bresized = imutils.resize(blurred, width=460)
    summing = np.hstack((resized, bresized))
    cv2.imshow("Summing", summing)
    cv2.waitKey(0)

    output = img.copy()
    cv2.rectangle(output, (270, 50), (420, 260), (0, 0, 255), 2)
    cv2.imshow("Rectangle", output)
    cv2.waitKey(0)

    output2 = img.copy()
    cv2.line(output2, (0, 0), (200, 200), (255, 0, 0), 5)
    cv2.imshow("Line", output2)
    cv2.waitKey(0)

    output3 = img.copy()
    cv2.circle(output3, (100, 100), 50, (0, 0, 255), 2)
    cv2.imshow("Circle", output3)
    cv2.waitKey(0)

    output4 = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output4, 'Stanislaw', (10, 500), font, 4, (255, 0, 0), 2, cv2.LINE_4)
    cv2.imshow("Text", output4)
    cv2.waitKey(0)
