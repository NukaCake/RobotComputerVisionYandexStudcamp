import cv2
import time
import numpy as np

import os
import subprocess
import torch
from PIL import Image
import numpy as np
from convert_map import *
from test import generate_grid


class TopCamera:
    def __init__(self):
        self.direction = 'Left'
        self.initUndistort()
        self.url = 'imgs/output_video.avi'
        self.cap = self.initCapture()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.isVideo = True
        # Define the points for the trapezoid (clockwise order)
        A = np.float32([[370, 170], [370, 925], [1365, 930], [1400, 180]])
        B = np.float32([[370, 130], [370, 930], [1400, 930], [1450, 130]])

        # Calculate the perspective transformation matrix
        self.M = cv2.getPerspectiveTransform(A, B)

    def split_video(self):
        iters = 0
        while True:
            iters += 1
            if iters % 4 == 0:
                ret, frame = self.cap.read()
                if not ret:
                    print('Error. Stop process')
                    break
                frame = self.undistort(frame)

    def initCapture(self):
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            print('Cannot open camera')
        else:
            print('Camera initialized')
            return cap

    def initUndistort(self):
        DIM = (1920, 1080)
        scale = 0.9
        if self.direction == 'Left':
            K = np.array([[1278.3329591423956, 0.0, 947.1793094545336], [0.0, 1272.263911500477, 545.6556001724595],
                          [0.0, 0.0, 1.0]])
            D = np.array([[-0.08072115075733356], [-0.06195802439215523], [-0.6752344267487737], [1.384812125687627]])
        else:
            K = np.array(
                [[1173.9872439859596, 0.0, 931.2708588716505], [0.0, 1175.0690115602886, 604.2283683612116],
                 [0.0, 0.0, 1.0]])
            D = np.array([[-0.10552262366830106], [0.1933396850327919], [-0.39591591343946037], [0.27896878887047616]])
        K_new = K.copy()
        K_new[(0, 1), (0, 1)] = scale * K_new[(0, 1), (0, 1)]
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, DIM, cv2.CV_16SC2)

    def undistort(self, frame):
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def calibrate(self):
        CHECKERBOARD = (6, 8)

        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        _img_shape = None
        objpoints = []
        imgpoints = []

        i = 1
        while True:
            print(i)
            i += 1
            ret_f, frame = self.cap.read()
            if ret_f is False:
                break
            if i % 20 == 0:
                img = frame
                if _img_shape == None:
                    _img_shape = img.shape[:2]
                else:
                    assert _img_shape == img.shape[:2], "All images must share the same size."

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret:
                    objpoints.append(objp)
                    cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
                    imgpoints.append(corners)
        self.cap.release()
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        DIM = _img_shape[::-1]
        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")
        return DIM, K, D

    def display(self):
        iterations = 0
        max_nums = 10000
        while True:
            iterations += 1
            if iterations >= max_nums:
                break
            ret, frame = self.cap.read()
            if ret:
                if iterations % 500 == 0:
                    print(iterations)
                    frame = self.undistort(frame)
                    # frame = self.exposure(frame)
                    frame = self.binarize(frame)
                    frame = frame[150:950, 400:1500]
                    # img, contours = self.find_contours(frame)
                    # biggest_3 = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                    # image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    # cv2.drawContours(image, biggest_3[1:], -1, (0, 255, 0), 2)
                    cv2.imwrite(f'{iterations // 500}-map.png', frame)
                    # self.show([frame], True)
                    # biggest = sorted()
            else:
                print('Capture Error')
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print('Video Stopped')
        cv2.destroyAllWindows()
        self.cap.release()

    def exposure(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # threshold grayscale image to extract glare
        mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]

        # Optionally add some morphology close and open, if desired
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # use mask with input to do inpainting
        result = cv2.inpaint(img, mask, 21, cv2.INPAINT_TELEA)
        return result

    def show(self, images, video=False):
        size = (1080, 720)  # изменим размер до адекватного
        for i, image in enumerate(images):  # перебираем все изображения из списка
            window_name = f'window {i}'  # создаем имя окна
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, *size)  # изменяем размер окна
            cv2.imshow(window_name, image)  # выводим изображение
        if not video:
            cv2.waitKey(0)  # ждем закрытия окна
            cv2.destroyAllWindows()  # закрываем все окна

    def binarize(self, image, show_image=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 400, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if show_image:
            self.show([image, binary], self.isVideo)
        return binary

    def find_contours(self, image, show_image=False):
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        if show_image:
            self.show([image])
        return image, contours

    def crop(self, frame, x1, x2, y1, y2):
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        return frame[y1:y2, x1:x2]

    def crop_in_the_middle(self, frame):
        # Борьба с fisheye и с второй разметкой
        frame = frame[:, 100:, :]
        bin_frame = camera.binarize(frame)
        x, y = bin_frame.shape[1] // 2, bin_frame.shape[0] // 2

        np_row = bin_frame[y, :]
        np_col = bin_frame[:, x]

        mxx = -1
        mxy = -1
        mnx = bin_frame.shape[1]
        mny = bin_frame.shape[0]
        for i in range(bin_frame.shape[1]):
            if np_row[i] == 0:
                mnx = min(mnx, i)
                mxx = max(mxx, i)
        for i in range(bin_frame.shape[0]):
            if np_col[i] == 0:
                mny = min(mny, i)
                mxy = max(mxy, i)

        frame = frame[mny:mxy, mnx:mxx, :]
        return frame

    def warp_transform(self, frame):
        return cv2.warpPerspective(frame, self.M, (1700, 1080))


camera = TopCamera()
cap = cv2.VideoCapture('output_video.avi')
i = 0

while True:
    width = 50
    height = 50
    grid, start, goal = generate_grid(width, height)
    i+=1
    if i%20==0:
        ret, frame = cap.read()
        if not ret:
            print('End')
            break
        frame = camera.undistort(frame)
        frame = camera.warp_transform(frame)
        try:
            frame = camera.crop_in_the_middle(frame)
            frame = camera.binarize(frame)
            #cv2.imshow('window', frame)
            grid = image_to_grid(frame)
            start = (1, 1)

            print_grid(grid, start=start, goal=goal)
            print(f"\nFinding path from {start} to {goal}...\n")

            path = a_star_search(grid, start, goal)

            if path:
                print("Path found:")
                visualize_path(grid, start, goal, path)
            else:
                print("No path found.")
        except:
            print('ERROR', frame)
            pass
        finally:
            if cv2.waitKey(0) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break