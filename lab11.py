import cv2
import numpy as np
import math

if __name__ == '__main__':

    cap = cv2.VideoCapture("car2.mp4")
    ret, frame = cap.read()

    cv2.imshow("image", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayscale", gray_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    blur_kernel_size = (15, 15)
    gray_blur = cv2.GaussianBlur(gray_img, blur_kernel_size, 0)
    cv2.imshow("blur", gray_blur)
    cv2.waitKey()
    cv2.destroyAllWindows()

    canny_low_threshold = 20
    canny_high_threshold = 100


    def canny(img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


    blur_canny = canny(gray_blur, canny_low_threshold, canny_high_threshold)
    cv2.imshow("canny", blur_canny)
    cv2.waitKey()
    cv2.destroyAllWindows()

    height, width = frame.shape[:2]
    h = 200
    w = 1000
    x = 450
    y = 0
    img1 = cv2.cvtColor(blur_canny[x: x + h, y: y + w], cv2.COLOR_GRAY2BGR)
    img2 = np.zeros_like(frame)
    img2[x:x + h, y: y + w] = img1
    cv2.imshow("grayscale", img2)
    cv2.waitKey()
    cv2.destroyAllWindows()

    dst = cv2.Canny(img2, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite("linesVid.jpg", cdstP)
    cv2.imshow("Lines Huff", cdstP)
    cv2.waitKey()
    cv2.destroyAllWindows()

    lines = cv2.imread("linesVid.jpg")
    sum_image = cv2.addWeighted(frame, 0.8, lines, 1, 0)
    cv2.imshow("Lines Huff", sum_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture('car2.mp4')
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Liczba klatek: " + str(frames))
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        h = 600
        w = 1920
        x = 400
        y = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img1 = gray[x: x + h, y: y + w]
        img2 = np.zeros_like(gray)
        img2[x:x + h, y: y + w] = img1
        dst = cv2.Canny(img2, 10, 200, None, 3)
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 2000 * (-b)), int(y0 + 5000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 2000 * a))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

        sum_image = cv2.addWeighted(frame, 0.8, cdstP, 1, 0)
        cv2.putText(sum_image, 'Stanislaw Jablonski', (900, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        cv2.imshow("Lines", sum_image)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
