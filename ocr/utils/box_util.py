import numpy as np
import cv2
import math

def bbox_resize(boxes, scale):
    if isinstance(boxes, list):
        resized_boxes = np.array(boxes, dtype=float)
    else:
        resized_boxes = boxes.copy()
    scale_x, scale_y = scale
    resized_boxes[:, 0] *= scale_x
    resized_boxes[:, 1] *= scale_y
    resized_boxes[:, 2] *= scale_x
    resized_boxes[:, 3] *= scale_y
    return resized_boxes


def bbox_margin(bboxes, margin):
    """Add margins for boxes
    Arguments:
        bboxes {list or nparray} -- boxes of (xmin, ymin, xmax, ymax)
        margin {list} -- margin
    Returns:
        newboxes
    """
    assert len(margin) == 2, "wrong shape of margin"
    if isinstance(bboxes, list):
        new_bboxes = np.array(bboxes)
    else:
        new_bboxes = bboxes.copy()
    margin_left, margin_top = margin

    new_bboxes[:, 0] += margin_left
    new_bboxes[:, 1] += margin_top
    new_bboxes[:, 2] += margin_left
    new_bboxes[:, 3] += margin_top
    return new_bboxes.tolist()


def cal_gt_box(box, r=0.5):
    xmin, ymin, xmax, ymax = box
    w = xmax - xmin
    h = ymax - ymin
    xgmin = (xmin + xmax - w * r) / 2
    ygmin = (ymin + ymax - h * r) / 2
    xgmax = (xmin + xmax + w * r) / 2
    ygmax = (ymin + ymax + h * r) / 2
    return np.array([xgmin, ygmin, xgmax, ygmax])


def bbox_rotate90(bboxes, img_size):
    """rotate 90 degress

    Arguments:
        bboxes {np.array} -- array of (x1, y1, x2, y2)
    """
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    w, h = img_size
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    newboxes = np.vstack([y1, w - x2, y2, w - x1]).T
    return newboxes


# def bbox_rotate(bbox, angle, rows, cols, interpolation=cv2.INTER_LINEAR):
#     """Rotates a bounding box by angle degrees

#     Args:
#         bbox (tuple): A tuple (x_min, y_min, x_max, y_max).
#         angle (int): Angle of rotation in degrees
#         rows (int): Image rows.
#         cols (int): Image cols.
#         interpolation (int): interpolation method.

#         return a tuple (x_min, y_min, x_max, y_max)
#     """
#     scale = cols / float(rows)
#     x = np.array([bbox[0], bbox[2], bbox[2], bbox[0]])
#     y = np.array([bbox[1], bbox[1], bbox[3], bbox[3]])
#     x = x - 0.5
#     y = y - 0.5
#     angle = np.deg2rad(angle)
#     x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
#     y_t = -np.sin(angle) * x * scale + np.cos(angle) * y
#     x_t = x_t + 0.5
#     y_t = y_t + 0.5
#     return [min(x_t), min(y_t), max(x_t), max(y_t)]


def bbox_rotate(box, a):  # 旋转中心点，框的左上，框的w，h，旋转角
    # x1, y1, x2, y2 = box
    center_x1, center_y1 = 0, 0
    x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
    a = (math.pi * a) / 180  # 角度转弧度
    x1, y1 = x, y  # 旋转前左上
    x2, y2 = x + w, y  # 旋转前右上
    x3, y3 = x + w, y + h  # 旋转前右下
    x4, y4 = x, y + h  # 旋转前左下

    px1 = (x1 - center_x1) * math.cos(a) - (y1 - center_y1) * math.sin(a) + center_x1  # 旋转后左上
    py1 = (x1 - center_x1) * math.sin(a) + (y1 - center_y1) * math.cos(a) + center_y1
    px2 = (x2 - center_x1) * math.cos(a) - (y2 - center_y1) * math.sin(a) + center_x1  # 旋转后右上
    py2 = (x2 - center_x1) * math.sin(a) + (y2 - center_y1) * math.cos(a) + center_y1
    px3 = (x3 - center_x1) * math.cos(a) - (y3 - center_y1) * math.sin(a) + center_x1  # 旋转后右下
    py3 = (x3 - center_x1) * math.sin(a) + (y3 - center_y1) * math.cos(a) + center_y1
    px4 = (x4 - center_x1) * math.cos(a) - (y4 - center_y1) * math.sin(a) + center_x1  # 旋转后左下
    py4 = (x4 - center_x1) * math.sin(a) + (y4 - center_y1) * math.cos(a) + center_y1

    return [int(x) for x in [px1, py1, px3, py3]]


def bbox_xywh_rotate(box, a):  # 旋转中心点，框的左上，框的w，h，旋转角
    # x1, y1, x2, y2 = box
    center_x1, center_y1 = 0, 0
    x, y, w, h = box
    a = (math.pi * a) / 180  # 角度转弧度
    x1, y1 = x, y  # 旋转前左上
    x2, y2 = x + w, y  # 旋转前右上
    x3, y3 = x + w, y + h  # 旋转前右下
    x4, y4 = x, y + h  # 旋转前左下

    px1 = (x1 - center_x1) * math.cos(a) - (y1 - center_y1) * math.sin(a) + center_x1  # 旋转后左上
    py1 = (x1 - center_x1) * math.sin(a) + (y1 - center_y1) * math.cos(a) + center_y1
    px2 = (x2 - center_x1) * math.cos(a) - (y2 - center_y1) * math.sin(a) + center_x1  # 旋转后右上
    py2 = (x2 - center_x1) * math.sin(a) + (y2 - center_y1) * math.cos(a) + center_y1
    px3 = (x3 - center_x1) * math.cos(a) - (y3 - center_y1) * math.sin(a) + center_x1  # 旋转后右下
    py3 = (x3 - center_x1) * math.sin(a) + (y3 - center_y1) * math.cos(a) + center_y1
    px4 = (x4 - center_x1) * math.cos(a) - (y4 - center_y1) * math.sin(a) + center_x1  # 旋转后左下
    py4 = (x4 - center_x1) * math.sin(a) + (y4 - center_y1) * math.cos(a) + center_y1

    return [int(x) for x in [px1, py1, px3, py3]]


def bboxes_rotate(bboxes, angle, rows, cols, interpolation=cv2.INTER_LINEAR):
    """Rotates a bounding box by angle degrees

    Args:
        bboxes (np.array): An array of tuple (x_min, y_min, x_max, y_max).
        angle (int): Angle of rotation in degrees
        rows (int): Image rows.
        cols (int): Image cols.
        interpolation (int): interpolation method.

        return a tuple (x_min, y_min, x_max, y_max)
    """
    scale = cols / float(rows)
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    x = np.array([bboxes[:, 0], bboxes[:, 2], bboxes[:, 2], bboxes[:, 0]])
    y = np.array([bboxes[:, 1], bboxes[:, 1], bboxes[:, 3], bboxes[:, 3]])
    x = x - 0.5
    y = y - 0.5
    angle = np.deg2rad(angle)
    x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
    y_t = -np.sin(angle) * x * scale + np.cos(angle) * y
    x_t = x_t + 0.5
    y_t = y_t + 0.5
    return [min(x_t), min(y_t), max(x_t), max(y_t)]
