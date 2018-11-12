import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(img, title="image", full_screen=False, wait=True):
    if full_screen:
        cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(title, img)
    if wait:
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()


def perform_canny_edge_detector(gray_image):
    edges = cv2.Canny(gray_image, 100, 200)
    plt.subplot(121), plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def perform_harris_corners(gray, img):
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(corners, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    show_image(img, "Harris Corners", True)

    return np.array(np.where((corners > 0.01 * corners.max()) == True))


def calc_sift(corners, gray):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = [cv2.KeyPoint(corners[1, i], corners[0, i], _size=5) for i in xrange(corners.shape[1])]
    sift.compute(gray, kp)
    img = cv2.drawKeypoints(gray, kp, None)
    show_image(img, "SIFT", True)


def match_images(image1_path, image2_path, out_image_path=None, show_result=False, ratio=0.75):
    img1 = cv2.imread(image1_path, 0)  # queryImage
    img2 = cv2.imread(image2_path, 0)  # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2, outImg=None)

    if show_result:
        plt.imshow(img3)
        plt.show()


def hough_transform(image_path, out_image_path=None, show_result=False, threshold=200):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    lines = np.squeeze(lines, axis=1)

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if show_result:
        show_image(img)

    if out_image_path is not None:
        cv2.imwrite(out_image_path, img)


def main():
    filename = 'lena.jpg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    perform_canny_edge_detector(gray)

    corners = perform_harris_corners(gray_float, img)

    calc_sift(corners, gray)

    match_images('openu1.jpg', 'openu2.jpg', 'openu_matching.jpg', show_result=True, ratio=0.7)

    hough_transform('lions.jpg', 'lions_hough.jpg', show_result=True, threshold=181)


if __name__ == '__main__':
    main()
