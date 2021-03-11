import cv2
import numpy as np
import torch.nn as nn


softmax = nn.Softmax2d()


def cal_region_scores(hm, pred_thres):
    binary_map = (hm[0] < pred_thres).astype(np.uint8)
    _, _, stats, center_points = cv2.connectedComponentsWithStats(
        binary_map, connectivity=4
    )
    regions = sorted_regions(stats[1:, :-1])
    cls_map = hm[1:]
    c, h, w = cls_map.shape
    cls_ids = []
    scores = []
    for region in regions:
        x, y, w, h = region
        region_map = cls_map[:, y : y + h, x : x + w]
        num = binary_map[y : y + h, x : x + w].sum()
        cls_scores = np.sum(region_map.reshape(c, -1), axis=1) / num
        cls_id = cls_scores.argmax()
        cls_ids.append(cls_id + 1)
        scores.append(cls_scores[cls_id])
    return cls_ids, scores


def sorted_regions(regions):
    indices = np.argsort(regions[:, 0])
    return regions[indices]


def post_process(outputs, pred_thres=0.1):
    assert outputs.shape[0] == 1, "batch size should be setted 1"
    hm = softmax(outputs).cpu().data.numpy()[0]  # class, h, w
    cls_ids, scores = cal_region_scores(hm, pred_thres)
    return cls_ids, scores
