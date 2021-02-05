import os, sys, time, traceback
from copy import deepcopy
import cv2
import numpy as np

class NMS(object):

    @staticmethod
    def get_iou(bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        Returns
        -------
        float
            in [0, 1]
        """
        left1,top1,right1,bottom1 = bb1
        left2,top2,right2,bottom2 = bb2

        # determine the coordinates of the intersection rectangle
        x_left = max(left1, left2)
        y_top = max(top1, top2)
        x_right = min(right1, right2)
        y_bottom = min(bottom1, bottom2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (right1 - left1) * (bottom1 - top1)
        bb2_area = (right2 - right1) * (bottom2 - top2)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = 0.0
        if(bb1_area + bb2_area - intersection_area) == 0.0:
            iou = 1.0
        else:
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

        return iou

    @staticmethod
    def filter(rects):
        results = []
        rids = []

        for rid,rect in enumerate(rects):
            rids.append(rid)

        del_rids = []
        scores = []  #  仅仅为了debug使用
        for i_rid in rids:
            for j_rid in rids:
                if j_rid <= i_rid:
                    continue

                bbx_i = rects[i_rid]
                bbx_j = rects[j_rid]

                iou = NMS.get_iou(bbx_i, bbx_j)

                score = i_rid,j_rid,iou
                if iou >= 0.9:
                    del_rids.append(j_rid)
                scores.append(score)

        for rid,rect in enumerate(rects):
            if int(rid) not in del_rids:
                results.append(rect)

        return results


class DetectNetV2PostProcess(object):

    def __init__(self, width, height, score=0.5, classes=[0]):
        '''
        Params:
            width,height(int) is the image-size that you want to get the BBX
            score(float) is the confidence
            classes(int-list) is the 3-classes(0 for person,1 for bag, 2 for face)
        '''
        self.image_width = width
        self.image_height = height

        self.model_h = 544
        self.model_w = 960
        self.stride = 16.0
        self.box_norm = 35.0

        self.grid_h = int(self.model_h / self.stride)
        self.grid_w = int(self.model_w / self.stride)
        self.grid_size = self.grid_h * self.grid_w

        self.grid_centers_w = []
        self.grid_centers_h = []

        for i in range(self.grid_h):
            value = (i * self.stride + 0.5) / self.box_norm
            self.grid_centers_h.append(value)

        for i in range(self.grid_w):
            value = (i * self.stride + 0.5) / self.box_norm
            self.grid_centers_w.append(value)

        '''
        min_confidence (float): min confidence to accept detection
        analysis_classes (list of int): indices of the classes to consider
        '''
        self.min_confidence = score
        self.analysis_classes = classes

    def applyBoxNorm(self, o1, o2, o3, o4, x, y):
        """
        Applies the GridNet box normalization
        Args:
            o1 (float): first argument of the result
            o2 (float): second argument of the result
            o3 (float): third argument of the result
            o4 (float): fourth argument of the result
            x: row index on the grid
            y: column index on the grid
        Returns:
            float: rescaled first argument
            float: rescaled second argument
            float: rescaled third argument
            float: rescaled fourth argument
        """
        o1 = (o1 - self.grid_centers_w[x]) * -self.box_norm
        o2 = (o2 - self.grid_centers_h[y]) * -self.box_norm
        o3 = (o3 + self.grid_centers_w[x]) * self.box_norm
        o4 = (o4 + self.grid_centers_h[y]) * self.box_norm
        return o1, o2, o3, o4

    def change_model_size_to_real(self, model_size, type):
        real_size = 0
        if type == 'x':
            real_size = (model_size / float(self.model_w)) * self.image_width
        elif type == 'y':
            real_size = (model_size / float(self.model_h)) * self.image_height
        real_size = int(real_size)
        return real_size

    def start(self, buffer_bbox, buffer_scores):
        """
        Postprocesses the inference output
        Args:
            outputs (list of float): inference output
        Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
        """

        bbs = []
        for c in range(3):
            if c not in self.analysis_classes:
                continue

            x1_idx = (c * 4 * self.grid_size)
            y1_idx = x1_idx + self.grid_size
            x2_idx = y1_idx + self.grid_size
            y2_idx = x2_idx + self.grid_size

            boxes = buffer_bbox
            for h in range(self.grid_h):
                for w in range(self.grid_w):
                    i = w + h * self.grid_w
                    score = buffer_scores[c * self.grid_size + i]
                    if score >= self.min_confidence:
                        o1 = boxes[x1_idx + w + h * self.grid_w]
                        o2 = boxes[y1_idx + w + h * self.grid_w]
                        o3 = boxes[x2_idx + w + h * self.grid_w]
                        o4 = boxes[y2_idx + w + h * self.grid_w]

                        o1, o2, o3, o4 = self.applyBoxNorm(o1, o2, o3, o4, w, h)

                        xmin_model = int(o1)
                        ymin_model = int(o2)
                        xmax_model = int(o3)
                        ymax_model = int(o4)

                        xmin_image = self.change_model_size_to_real(xmin_model, 'x')
                        ymin_image = self.change_model_size_to_real(ymin_model, 'y')
                        xmax_image = self.change_model_size_to_real(xmax_model, 'x')
                        ymax_image = self.change_model_size_to_real(ymax_model, 'y')

                        rect = (xmin_image, ymin_image, xmax_image, ymax_image)

                        bbs.append(rect)
        return bbs