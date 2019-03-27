#!/usr/bin/env python
# encoding=utf-8
#test
#不规则图形NMS
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形
import h5py
import scipy.io as sio
import os
for epoch in range(10,21):
    train_data = 'creat_data_cor_30'
    pic = '003003'
    #epoch = 11

    thresh = 0.25
    test_data_name = os.path.join('pos_result', train_data, pic, str(epoch) + '.mat')
    test_data_file = h5py.File(test_data_name, 'r')
    scores_b = test_data_file['score'][:]
    scores = scores_b.transpose(1, 0)
    pos_b = test_data_file['pos_correct_all'][:]
    pos = pos_b.transpose(2, 1, 0)
    pos = pos[:, :-1, :]
    s = scores[:, 0]
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        iou = np.zeros(len(order) - 1)
        a = pos[i]
        poly1 = Polygon(a).convex_hull
        for j in range(1, len(order)):

            b = pos[order[j]]
            poly2 = Polygon(b).convex_hull
            union_poly = np.concatenate((a, b))
            if not poly1.intersects(poly2):  # 如果两四边形不相交
                iou[j - 1] = 0
            else:
                try:
                    inter_area = poly1.intersection(poly2).area  # 相交面积
                    # print(inter_area)
                    # union_area = poly1.area + poly2.area - inter_area
                    union_area = MultiPoint(union_poly).convex_hull.area
                    # print(union_area)
                    if union_area == 0:
                        iou[j - 1] = 0
                    else:
                        iou[j - 1] = float(inter_area) / union_area
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured, iou set to 0')
                    iou = 0
        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]

    print(keep)
    scores = scores_b[:, keep]
    scores = scores.transpose()
    pos_nms = pos_b[:, :, keep]
    pos_nms = pos_nms.transpose(2, 1, 0)
    sio.savemat(os.path.join('pos_result', train_data, pic, str(epoch) + 'NMS.mat'), {"scores": scores, "pos_nms": pos_nms})
