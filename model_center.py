from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import torch

from models.networks.pose_dla_dcn import get_pose_net, load_model, pre_process, ctdet_decode, post_process, \
    merge_outputs

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import params
class CENTER_MODEL(object):
    def __init__(self, weight_path=""):
        self.scale = 1.0
        self.threshold = 0.25
        self.num_classes = params.num_classes
        self.K = params.max_per_image
        self.model = get_pose_net(num_layers=params.num_layers, heads=params.heads, head_conv=params.head_conv)
        self.model = load_model(self.model, weight_path)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def detect_obj(self, img):
        """

        :param img: PIL image greyscale 'L' mode
        :return:
        """
        image, meta = pre_process(img, self.scale)
        with torch.no_grad():
            if torch.cuda.is_available():
                image = image.cuda()
            start = time.time()
            output = self.model(image)[-1]
            print(time.time() - start)
            hm = output['hm'].sigmoid_()
            reg = output['reg']
            dets = ctdet_decode(hm, reg=reg, K=self.K)

        dets = post_process(dets, meta)
        dets = [dets]

        results = merge_outputs(dets)

        list_center = []
        # list_center_label = []
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[2] >= self.threshold:
                    x_center, y_center = max(int(bbox[0]), 0), max(0, int(bbox[1]))
                    # list_center_label.append([[x_center, y_center], j])  # x, y ,label_id
                    list_center.append([x_center, y_center])
        # print(list_center)
        # img_draw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # colors = {1:(137,22,140)}
        # for center in list_center_label:
        #     img_draw = cv2.circle(img_draw, (center[0][0], center[0][1]), radius=1, color=colors[center[1]], thickness=2)

        # plt.imshow(img_draw)
        # plt.show()

        if (len(list_center) == 4):
            points = self.order_points(np.array(list_center[:4]))
        else:
            print("Cannot detect 4 corners !!!, Number of conners detected was ", len(list_center))
            return None

        img_aligh = self.align(img, points)
        # plt.imshow(img_aligh)
        # plt.show()
        return img_aligh


    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def align(self, image, pts):
        pts = np.array(pts, dtype="float32")
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

if __name__ == "__main__":
    model = CENTER_MODEL(weight_path="weights/model_best.pth")
    paths = glob.glob("img_test/*")
    for path in paths:
    #img_path = "img_test/511.jpg"
        img = cv2.imread(path)
        model.detect_obj(img)


