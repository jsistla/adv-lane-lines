import glob
import os
import pickle

import cv2
import numpy as np

# we store camera calibration pickle in the camera_cal directory
CAMERA_CALIBRATION_PICKLE_FILE = '../camera_cal/calibration_pickle.p'
UNDISTORTED_OUTPUT_IMAGES = '../output_images/camera_cal/'

class CameraCalibrate:
    def __init__(self, calibration_images, rows, col, use_existing_camera_coefficients=True):
                 
        """
        This class encapsulates camera calibration processpipe. When creating an instance of
        CameraCalibrator class, 
        :param calibration_images: image used for camera calibration
        :param rows: number of horizontal corners in calibration images
        :param col:  number of vertical corners in calibration images
        """
        self.calibration_images = calibration_images
        self.rows = rows
        self.col = col
        self.object_points = []
        self.image_points = []

        if not os.path.exists(CAMERA_CALIBRATION_PICKLE_FILE):
            self._calibrate_camera()

    def _calibrate_camera(self):
        """
        Camera calibrate abstraction function
        :return:
            Camera calibration coefficients as a python dictionary
        """
        object_point = np.zeros((self.rows * self.col, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:self.rows, 0:self.col].T.reshape(-1, 2)

        for idx, file_name in enumerate(self.calibration_images):
            image = cv2.imread(file_name)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ret, corners = cv2.findChessboardCorners(gray_image,
                                                     (self.rows, self.col),
                                                     None)
            if ret:
                self.object_points.append(object_point)
                self.image_points.append(corners)

        image_size = (image.shape[1], image.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points,
                                                           self.image_points, image_size, None, None)
        calibrated_data = {'mtx': mtx, 'dist': dist}

        with open(CAMERA_CALIBRATION_PICKLE_FILE, 'wb') as f:
            pickle.dump(calibrated_data, file=f)

    def undistort(self, image):
        """
        :param image:
        :return:
        """

        if not os.path.exists(CAMERA_CALIBRATION_PICKLE_FILE):
            raise Exception('Camera calibration data file does not exist at ' +
                            CAMERA_CALIBRATION_PICKLE_FILE)

        with open(CAMERA_CALIBRATION_PICKLE_FILE, 'rb') as f:
            calibrated_data = pickle.load(file=f)

        # image = cv2.imread(image)
        return cv2.undistort(image, calibrated_data['mtx'], calibrated_data['dist'],
                             None, calibrated_data['mtx'])
"""
    def saveUndistorted():
        
        #:param image:
        #:return:
        
        calibration_images = glob.glob('../camera_cal/calibration*.jpg')
        for image in calibration_images:
            start = image.rindex('/') + 1
            end = image.rindex('.')
            dest_path = UNDISTORTED_OUTPUT_IMAGES + image[start:end] + '.jpg'
    
            image = cv2.imread(image)
            undistorted_image = undistort(image)
            # Save undistorted image
            cv2.imwrite(dest_path, undistorted_image)
"""
class Transformation:
    def __init__(self, src_points, dest_points):

        self.src_points = src_points
        self.dest_points = dest_points

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dest_points)
        self.M_inverse = cv2.getPerspectiveTransform(self.dest_points, self.src_points)

    def transform(self, image):

        size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.M, size, flags=cv2.INTER_LINEAR)

    def inverse_transform(self, src_image):

        size = (src_image.shape[1], src_image.shape[0])
        return cv2.warpPerspective(src_image, self.M_inverse, size, flags=cv2.INTER_LINEAR)
"""
    def draw_lines(img, vertices):
        pts= np.int32([vertices])
        cv2.polylines(img, pts, True, (0,0,255))

    def perspective_unwarp(img, Minv):
        img_size = (img.shape[0], img.shape[1])
        unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
        return unwarped

    def equalize_lines(self, alpha=0.9):
        mean = 0.5 * (self.left_line.coeff_history[:, 0] + self.right_line.coeff_history[:, 0])
        self.left_line.coeff_history[:, 0] = alpha * self.left_line.coeff_history[:, 0] + \
                                             (1-alpha)*(mean - np.array([0,0, 1.8288], dtype=np.uint8))
        self.right_line.coeff_history[:, 0] = alpha * self.right_line.coeff_history[:, 0] + \
                                              (1-alpha)*(mean + np.array([0,0, 1.8288], dtype=np.uint8))
"""
def denoise(image, threshold=4):
    """
    This method is used to reduce the noise of binary images.
    :param image:
        binary image (0 or 1)
    :param threshold:
        min number of neighbours with value
    :return:

    """
    #k = np.ones((5,5),np.float32)/25
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(image, ddepth=-1, kernel=k)
    image[nb_neighbours < threshold] = 0
    return image

def binarize(image, gray_thresh=(20, 255), s_thresh=(170, 255), l_thresh=(30, 255), sobel_kernel=3):
    """
    This method extracts lane line pixels from road an image. Then create a minarized image where
    lane lines are marked in white color and rest of the image is marked in black color.
    :param image:
        Source image
    :param gray_thresh:
        Minimum and maximum gray color threshold
    :param s_thresh:
        This tuple contains the minimum and maximum S color threshold in HLS color scheme
    :param l_thresh:
        Minimum and maximum L color (after converting image to HLS color scheme)
        threshold allowed in the source image
    :param sobel_kernel:
        Size of the kernel use by the Sobel operation.
    :return:
        The binarized image where lane line pixels are marked in while color and rest of the image
        is marked in block color.
    """

    # first we take a copy of the source iamge
    image_copy = np.copy(image)

    # convert RGB image to HLS color space.
    # HLS more reliable when it comes to find out lane lines
    hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]

    # Next, we apply Sobel operator in X direction and calculate scaled derivatives.
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Next, we generate a binary image based on gray_thresh values.
    thresh_min = gray_thresh[0]
    thresh_max = gray_thresh[1]
    sobel_x_binary = np.zeros_like(scaled_sobel)
    sobel_x_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Next, we generated a binary image using S component of our HLS color scheme and
    # provided S threshold
    s_binary = np.zeros_like(s_channel)
    s_thresh_min = s_thresh[0]
    s_thresh_max = s_thresh[1]
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Next, we generated a binary image using S component of our HLS color scheme and
    # provided S threshold
    l_binary = np.zeros_like(l_channel)
    l_thresh_min = l_thresh[0]
    l_thresh_max = l_thresh[1]
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # finally, return the combined binary image
    binary = np.zeros_like(sobel_x_binary)
    binary[((l_binary == 1) & (s_binary == 1) | (sobel_x_binary == 1))] = 1
    binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')

    return denoise(binary)

class LaneLines():
    def __init__(self):
        """"""

        self.detected = False

        self.left_fit = None
        self.right_fit = None

        self.MAX_BUFFER_SIZE = 12

        self.buffer_index = 0
        self.iter_counter = 0

        self.buffer_left = np.zeros((self.MAX_BUFFER_SIZE, 720))
        self.buffer_right = np.zeros((self.MAX_BUFFER_SIZE, 720))

        self.perspective = self._build_perspective_transformer()
        self.calibrator = self._build_camera_calibrator()

    @staticmethod
    def _build_perspective_transformer():
        """
        :return:
        """
        corners = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])
        new_top_left = np.array([corners[0, 0], 0])
        new_top_right = np.array([corners[3, 0], 0])
        offset = [50, 0]

        src = np.float32([corners[0], corners[1], corners[2], corners[3]])
        dst = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])

        perspective = Transformation(src, dst)
        return perspective

    @staticmethod
    def _build_camera_calibrator():
        """
        :return:
        """
        calibration_images = glob.glob('../camera_cal/calibration*.jpg')
        calibrator = CameraCalibrate(calibration_images,
                                      9, 6, use_existing_camera_coefficients=True)
        return calibrator

    def base_lane_finder(self, binary_warped):
        """
        :param binary_warped:
        :return:
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :, 0], axis=0)

        # get midpoint of the histogram
        midpoint = np.int(histogram.shape[0] / 2)

        # get left and right halves of the histogram
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # based on number of events, we calculate hight of a window
        nwindows = 9
        window_height = np.int(binary_warped.shape[0] / nwindows)

        # Extracts x and y coordinates of non-zero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set current x coordinated for left and right
        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = 75
        min_num_pixels = 35

        # save pixel ids in these two lists
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            left_indice = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            right_indice = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(left_indice)
            right_lane_inds.append(right_indice)

            if len(left_indice) > min_num_pixels:
                leftx_current = np.int(np.mean(nonzerox[left_indice]))
            if len(right_indice) > min_num_pixels:
                rightx_current = np.int(np.mean(nonzerox[right_indice]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        fit_leftx = self.left_fit[0] * fity ** 2 + self.left_fit[1] * fity + self.left_fit[2]
        fit_rightx = self.right_fit[0] * fity ** 2 + self.right_fit[1] * fity + self.right_fit[2]

        self.detected = True

        return fit_leftx, fit_rightx

    def advance_lane_finder(self, binary_warped):
        """
        :param binary_warped:
        :return:
        """
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 75

        left_lane_inds = (
            (nonzerox > (
                self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - margin)) & (
                nonzerox < (
                    self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (
                self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - margin)) & (
                nonzerox < (
                    self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        return left_fitx, right_fitx

    def left_right_curve_road(self, image_size, left_x, right_x):
        """
        This method calculates left and right road curvature and off of the vehicle from the center
        of the lane
        :param image_size:
            Size of the image
        :param left_x:
            X coordinated of left lane pixels
        :param right_x:
            X coordinated of right lane pixels
        :return:
            Left and right curvatures of the lane and off of the vehicle from the center of the lane
        """
        # first we calculate the intercept points at the bottom of our image
        left_c = self.left_fit[0] * image_size[0] ** 2 + self.left_fit[1] * image_size[0] + self.left_fit[2]
        right_c = self.right_fit[0] * image_size[0] ** 2 + self.right_fit[1] * image_size[0] + self.right_fit[2]

        # Next take the difference in pixels between left and right interceptor points
        #width = right_c - left_c
        #assert width > 0, 'Road width in pixel can not be negative'

        # Since average highway lane line width in US is about 3.7m
        # Source: https://en.wikipedia.org/wiki/Lane#Lane_width
        # we calculate length per pixel in meters

        xm_per_pix = 3.7 / 720
        ym_per_pix = 30 / 700

        # Recalculate road curvature in X-Y space
        ploty = np.linspace(0, 719, num=720)
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_x * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_x * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curve = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
                        np.absolute(2 * left_fit_cr[0])

        right_curve = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                         np.absolute(2 * right_fit_cr[0])

        # Next, we can lane deviation
        cal_center = (left_c + right_c) / 2.0
        lane_deviation = (cal_center - image_size[1] / 2.0) * xm_per_pix

        return left_curve, right_curve, lane_deviation

    @staticmethod
    def fill_lane_lines(image, fit_left_x, fit_right_x):
        """
        This utility method highlights correct lane section on the road
        :param image:
            On top of this image, my lane will be highlighted
        :param fit_left_x:
            X coordinated of the left second order polynomial
        :param fit_right_x:
            X coordinated of the right second order polynomial
        :return:
            The input image with highlighted lane line.
        """
        copy_image = np.zeros_like(image)
        fit_y = np.linspace(0, copy_image.shape[0] - 1, copy_image.shape[0])

        pts_left = np.array([np.transpose(np.vstack([fit_left_x, fit_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_right_x, fit_y])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(copy_image, np.int_([pts]), (0, 255, 0))

        return copy_image

    def merge_images(self, binary_img, src_image):
        """
        This utility method merges merges two images
        :param binary_img:
        Binary image with highlighted lane segment.
        :param src_image:
        The original image on top of it we are going to highlight lane segment.
        :return:
        The Original image with highlighted lane segment.
        """
        copy_binary = np.copy(binary_img)
        copy_src_img = np.copy(src_image)

        copy_binary_pers = self.perspective.inverse_transform(copy_binary)
        result = cv2.addWeighted(copy_src_img, 1, copy_binary_pers, 0.3, 0)

        return result

    def processpipe(self, image):
        """
        This method takes an image as an input and produces an image with
        1. Highlighted lane line
        2. Left and right lane curvatures (in meters)
        3. Vehicle offset of the center of the lane (in meters)
        :param image:
            Source image
        :return:
            Annotated image with lane line details
        """
        image = np.copy(image)
        undistorted_image = self.calibrator.undistort(image)
        warped_image = self.perspective.transform(undistorted_image)
        binary_image = binarize(warped_image)

        if self.detected:
            fit_leftx, fit_rightx = self.advance_lane_finder(binary_image)
        else:
            fit_leftx, fit_rightx = self.base_lane_finder(binary_image)

        self.buffer_left[self.buffer_index] = fit_leftx
        self.buffer_right[self.buffer_index] = fit_rightx

        self.buffer_index += 1
        self.buffer_index %= self.MAX_BUFFER_SIZE

        if self.iter_counter < self.MAX_BUFFER_SIZE:
            self.iter_counter += 1
            ave_left = np.sum(self.buffer_left, axis=0) / self.iter_counter
            ave_right = np.sum(self.buffer_right, axis=0) / self.iter_counter
        else:
            ave_left = np.average(self.buffer_left, axis=0)
            ave_right = np.average(self.buffer_right, axis=0)

        left_curvature, right_curvature, calculated_deviation = self.left_right_curve_road(image.shape, ave_left,
                                                                                            ave_right)
        curvature_text = 'Left Curvature: {:.2f} m    Right Curvature: {:.2f} m'.format(left_curvature, right_curvature)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, curvature_text, (100, 50), font, 1, (221, 28, 119), 2)

        deviation_info = 'Lane Deviation: {:.3f} m'.format(calculated_deviation)
        cv2.putText(image, deviation_info, (100, 90), font, 1, (221, 28, 119), 2)

        filled_image = self.fill_lane_lines(binary_image, ave_left, ave_right)

        merged_image = self.merge_images(filled_image, image)

        return merged_image


if __name__ == '__main__':
    from moviepy.editor import VideoFileClip

    line = LaneLines()
    output_file = '../processpipeed_project_video.mp4'
    input_file = '../project_video.mp4'
    clip = VideoFileClip(input_file)
    out_clip = clip.fl_image(line.processpipe)
    out_clip.write_videofile(output_file, audio=False)