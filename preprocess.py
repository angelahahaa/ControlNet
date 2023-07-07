import cv2
from annotator.util import HWC3, resize_image
import numpy as np


def preprocess_seg(input_image, image_resolution, detector, detect_resolution):
    input_image = HWC3(input_image)
    detected_map = detector(resize_image(input_image, detect_resolution))
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    return img, detected_map

def preprocess_scribble(input_image, image_resolution):
    img = resize_image(HWC3(input_image), image_resolution)
    H, W, C = img.shape
    
    detected_map = np.zeros_like(img, dtype=np.uint8)
    detected_map[np.min(img, axis=2) < 127] = 255
    return img, detected_map

def preprocess_scribble_interactive(input_image, image_resolution):
    img = resize_image(HWC3(input_image['mask'][:, :, 0]), image_resolution)
    H, W, C = img.shape

    detected_map = np.zeros_like(img, dtype=np.uint8)
    detected_map[np.min(img, axis=2) > 127] = 255
    return img, detected_map

def preprocess_depth(input_image, image_resolution, detector, detect_resolution):
    input_image = HWC3(input_image)
    detected_map, _ = detector(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    return img, detected_map

def preprocess_fake_scribble(input_image, image_resolution, detector, detect_resolution):
    from annotator.hed import nms
    input_image = HWC3(input_image)
    detected_map = detector(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    detected_map = nms(detected_map, 127, 3.0)
    detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
    detected_map[detected_map > 4] = 255
    detected_map[detected_map < 255] = 0
    return img, detected_map

def preprocess_normal(input_image, image_resolution, detector, detect_resolution, bg_threshold):
    input_image = HWC3(input_image)
    _, detected_map = detector(resize_image(input_image, detect_resolution), bg_th=bg_threshold)
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    return img, detected_map

def preprocess_hough(input_image, image_resolution, detector, detect_resolution, value_threshold, distance_threshold):
    input_image = HWC3(input_image)
    detected_map = detector(resize_image(input_image, detect_resolution), value_threshold, distance_threshold)
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    return img, detected_map

def preprocess_hed(input_image, image_resolution, detector, detect_resolution):
    input_image = HWC3(input_image)
    detected_map = detector(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    return img, detected_map

def preprocess_pose(input_image, image_resolution, detector, detect_resolution):
    input_image = HWC3(input_image)
    detected_map, _ = detector(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    return img, detected_map

def preprocess_canny(input_image, image_resolution, detector, low_threshold, high_threshold):
    img = resize_image(HWC3(input_image), image_resolution)
    H, W, C = img.shape
    
    detected_map = detector(img, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)
    return img, detected_map
