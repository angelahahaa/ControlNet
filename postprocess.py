import cv2
import numpy as np


def postprocess_seg(detected_map, results):
    return [detected_map] + results

def postprocess_scribble(detected_map, results):
    return [255 - detected_map] + results

def postprocess_depth(detected_map, results):
    return [detected_map] + results

def postprocess_fake_scribble(detected_map, results):
    return [255 - detected_map] + results

def postprocess_normal(detected_map, results):
    return [detected_map] + results

def postprocess_hough(detected_map, results):
    return [255 - cv2.dilate(detected_map, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)] + results

def postprocess_hed(detected_map, results):
    return [detected_map] + results

def postprocess_pose(detected_map, results):
    return [detected_map] + results

def postprocess_canny(detected_map, results):
    return [255 - detected_map] + results