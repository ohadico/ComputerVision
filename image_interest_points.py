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


def main():
    filename = 'lena.jpg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    perform_canny_edge_detector(gray)

    corners = perform_harris_corners(gray_float, img)

    calc_sift(corners, gray)


if __name__ == '__main__':
    main()