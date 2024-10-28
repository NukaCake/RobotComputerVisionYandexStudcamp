import cv2
import numpy as np

"""Calibrating camera using chessboard method"""

# Define the chessboard size (number of inner corners per a chessboard row and column)
CHECKERBOARD = (10, 7)

# Termination criteria for corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the real chessboard dimensions
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Lists to store object points and image points from all frames
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Read your chessboard video for calibration
chessboard_video = cv2.VideoCapture('upcamera_chess.mp4')

while chessboard_video.isOpened():
    ret, frame = chessboard_video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret_corners:
        objpoints.append(objp)
        # Refine pixel coordinates for given 2d points.
        corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners_subpix)
        # Draw and display the corners
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners_subpix, ret_corners)
        cv2.imshow('Calibration', frame)
        cv2.waitKey(1)

chessboard_video.release()
cv2.destroyAllWindows()

# Calibration using fisheye model
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1,1,3), dtype=np.float64)] * N_OK
tvecs = [np.zeros((1,1,3), dtype=np.float64)] * N_OK

retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    criteria
)

print("Calibration complete.")
print("Camera matrix (K):\n", K)
print("Distortion coefficients (D):\n", D)

# Read the fisheye video to be undistorted
fisheye_video = cv2.VideoCapture('fisheye_video.mp4')

# Prepare for undistortion
dim = gray.shape[::-1]
balance = 0.0  # Adjust balance for field of view
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)

while fisheye_video.isOpened():
    ret, frame = fisheye_video.read()
    if not ret:
        break
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow('Undistorted Video', undistorted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fisheye_video.release()
cv2.destroyAllWindows()
