import cv2
import numpy as np
from matplotlib import pyplot as plt

MAX_WINDOW_HEIGHT = 600
MAX_WINDOW_WIDTH = 1000


def show_image(img, title="image", full_screen=False, wait=True):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL & cv2.WND_PROP_FULLSCREEN)

    if full_screen:
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        height, width, channels = img.shape

        window_height = height
        window_width = width

        if window_height > MAX_WINDOW_HEIGHT:
            window_width = int(MAX_WINDOW_HEIGHT / float(window_height) * width)
            window_height = MAX_WINDOW_HEIGHT

        if window_width > MAX_WINDOW_WIDTH:
            window_height = int(MAX_WINDOW_WIDTH / float(window_width) * height)
            window_width = MAX_WINDOW_WIDTH

        cv2.resizeWindow(title, window_width, window_height)

    cv2.imshow(title, img)

    if wait:
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()


def get_canny_edges(image_path, out_image_path=None, show_result=False,
                    low_threshold=100, high_threshold=200, aperture_size=3):
    print("Perform Canny Edge Detector")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, low_threshold, high_threshold, aperture_size)

    if out_image_path is not None:
        cv2.imwrite(out_image_path, edges)

    if show_result:
        plt.subplot(121)
        plt.imshow(gray, cmap='gray')
        plt.title('Original Image')
        plt.xticks([]), plt.yticks([])

        plt.subplot(122)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Image')
        plt.xticks([]), plt.yticks([])
        plt.show()

    return edges


def get_harris_corners(image_path, out_image_path=None, show_result=False,
                       block_size=2, k_size=3, k=0.04):
    print("Perform Harris Corner Detection")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    corners = cv2.cornerHarris(gray_float, block_size, k_size, k)

    if out_image_path is not None or show_result:
        # result is dilated for marking the corners, not important
        dst = cv2.dilate(corners, None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > 0.01 * dst.max()] = [0, 0, 255]

        if out_image_path is not None:
            cv2.imwrite(out_image_path, img)

        if show_result:
            show_image(img, "Harris Corners")

    return np.array(np.where((corners > 0.01 * corners.max()) == True))


def calc_sift(kp_indices, gray, show_result=False):
    print("Calculate SIFT for each keypoint")

    sift = cv2.xfeatures2d.SIFT_create()
    kp = [cv2.KeyPoint(kp_indices[1, i], kp_indices[0, i], _size=5) for i in xrange(kp_indices.shape[1])]
    sift.compute(gray, kp)

    if show_result:
        img = cv2.drawKeypoints(gray, kp, None)
        show_image(img, "SIFT")

    return kp


def match_images(image1_path, image2_path, out_image_path=None, show_result=False, ratio=0.75):
    print("Matching 2 images")

    img1 = cv2.imread(image1_path, 0)
    img2 = cv2.imread(image2_path, 0)

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

    if out_image_path is not None or show_result:
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2, outImg=None)

        if out_image_path is not None:
            cv2.imwrite(out_image_path, img3)

        if show_result:
            plt.imshow(img3)
            plt.show()

    return good


def hough_transform(image_path, out_image_path=None, show_result=False, threshold=200):
    print("Perform Hough Transform")

    img = cv2.imread(image_path)
    edges = get_canny_edges(image_path, low_threshold=50, high_threshold=150)
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
        show_image(img, "Hough Transform")

    if out_image_path is not None:
        cv2.imwrite(out_image_path, img)


def main():
    lena_file = 'lena.jpg'

    get_canny_edges(lena_file, 'lena_canny.jpg', show_result=True)

    corners = get_harris_corners(lena_file, 'lena_harris.jpg', show_result=True)

    img = cv2.imread(lena_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    calc_sift(corners, gray, show_result=True)

    match_images('openu1.jpg', 'openu2.jpg', 'openu_matching.jpg', show_result=True, ratio=0.7)

    hough_transform('lions.jpg', 'lions_hough.jpg', show_result=True, threshold=181)


if __name__ == '__main__':
    main()
