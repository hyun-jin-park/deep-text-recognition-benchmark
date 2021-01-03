import numpy as np
import math 
import PIL
import json
import cv2 

def compute_rotation_angle(box_pos, theta_threshold=1):
    delta_x = box_pos[1][0] - box_pos[0][0]
    delta_y = box_pos[1][1] - box_pos[0][1]

    if delta_x == 0 and delta_y < 0:
        return -90
    elif delta_x == 0 and delta_y > 0:
        return 90

    theta = math.atan(delta_y/delta_x) * 180 / math.pi

    if abs(theta) < theta_threshold and delta_x > 0:
        return 0
    elif abs(theta) < theta_threshold and delta_x < 0:
        return -180

    if theta > 0 and delta_x > 0:
        theta = theta
    elif theta > 0 > delta_x:
        theta = theta - 180
    elif theta < 0 < delta_x:
        theta = theta
    else:
        theta = 180+theta

    return theta

def align_points(box):
    # Make it clockwise align
    # box = np.array([(int(float(x)), int(float(y))) for x, y in zip(items[1::2], items[2::2])])
    centroid = np.sum(box, axis=0) / 4
    theta = np.arctan2(box[:, 1] - centroid[1], box[:, 0] - centroid[0]) * 180 / np.pi
    indices = np.argsort(theta)
    aligned_box = box[indices]
    return aligned_box


if __name__=='__main__':
    json_path = '/home/embian/Unity_Dataset/2021_01_02/Tiwan/BoundingBox.json'
    image_path = '/home/embian/Unity_Dataset/2021_01_02/Tiwan/rgb/rgb_51.png'
    json_fp = open(json_path, 'r')
    json_entries = json.load(json_fp)

    left_top = [int(json_entries['info_list'][51]['segment_list'][0]['quad']['left_top']['x']), 
                int(json_entries['info_list'][51]['segment_list'][0]['quad']['left_top']['y'])]

    right_top = [int(json_entries['info_list'][51]['segment_list'][0]['quad']['right_top']['x']), 
                int(json_entries['info_list'][51]['segment_list'][0]['quad']['right_top']['y'])]

    left_bottom = [int(json_entries['info_list'][51]['segment_list'][0]['quad']['left_bottom']['x']), 
                int(json_entries['info_list'][51]['segment_list'][0]['quad']['left_bottom']['y'])]

    right_bottom= [int(json_entries['info_list'][51]['segment_list'][0]['quad']['right_bottom']['x']), 
                int(json_entries['info_list'][51]['segment_list'][0]['quad']['right_bottom']['y'])]

    card_bounds = np.array([left_top, right_top, right_bottom, left_bottom])
    im = cv2.imread(image_path)
    im = cv2.polylines(im, [card_bounds], True, (255, 0, 0), 1)
    rotation_angle = compute_rotation_angle(card_bounds)
    matrix = cv2.getRotationMatrix2D(tuple(card_bounds[0]), rotation_angle, 1)
    # dst = cv2.warpAffine(image, matrix, (width, height))
    height, width, _ = im.shape
    dst = cv2.warpAffine(im, matrix, (width, height))
    rotated_bounds = np.matmul(matrix, card_bounds)
    cv2.imwrite('temp2.jpg', dst)

