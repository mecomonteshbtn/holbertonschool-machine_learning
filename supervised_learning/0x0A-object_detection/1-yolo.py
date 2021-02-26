#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 08:00:42 2021

@author: Robinson Montes
"""
import tensorflow.keras as K
import numpy as np


class Yolo():
    """
    Class Yolo that uses the Yolo v3 algorithm to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor of the class

        Arguments:
         - model_path is the path to where a Darknet Keras model is stored
         - classes_path is the path to where the list of class names used for
            the Darknet model, listed in order of index, can be found
         - class_t is a float representing the box score threshold for
            the initial filtering step
         - nms_t is a float representing the IOU threshold for
            non-max suppression
         - anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
            * outputs is the number of outputs (predictions) made by
                the Darknet model
            * anchor_boxes is the number of anchor boxes used for
                each prediction
            * 2 => [anchor_box_width, anchor_box_height]

        Public instance attributes:
         - model: the Darknet Keras model
         - class_names: a list of the class names for the model
         - class_t: the box score threshold for the initial filtering step
         - nms_t: the IOU threshold for non-max suppression
         - anchors: the anchor boxes
        """

        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """
        Function that calculates sigmoid
        """
        return 1 / (1 + np.exp(-x))

    # Public method
    def process_outputs(self, outputs, image_size):
        """
        Public method to process the outputs

        Arguments:
         - outputs is a list of numpy.ndarrays containing the predictions
            from the Darknet model for a single image:
            Each output will have the shape
            (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
             * grid_height & grid_width => the height and width of
                the grid used for the output
             * anchor_boxes => the number of anchor boxes used
             * 4 => (t_x, t_y, t_w, t_h)
             * 1 => box_confidence
             * classes => class probabilities for all classes
         - image_size is a numpy.ndarray containing the image’s original size
            [image_size[0], image_size[1]]

        Returns:
         A tuple of (boxes, box_confidences, box_class_probs):
         - boxes: a list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output:
            * 4 => (x1, y1, x2, y2)
            * (x1, y1, x2, y2) should represent the boundary box
                relative to original image
         - box_confidences: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1)
            containing the box confidences for each output, respectively
         - box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes)
            containing the box’s class probabilities
            for each output, respectively
        """

        img_height = image_size[0]
        img_width = image_size[1]

        boxes = []
        box_confidences = []
        box_class_probs = []
        for output in outputs:
            # Create the list with np.ndarray
            boxes.append(output[..., 0:4])
            # Calculate confidences for each output
            box_confidences.append(self.sigmoid(output[..., 4, np.newaxis]))
            # Calculate class probability for each output
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        for i, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape

            c = np.zeros((grid_height, grid_width, anchor_boxes), dtype=int)

            # Cy matrix
            idx_y = np.arange(grid_height)
            idx_y = idx_y.reshape(grid_height, 1, 1)
            Cy = c + idx_y

            # Cx matrix
            idx_x = np.arange(grid_width)
            idx_x = idx_x.reshape(1, grid_width, 1)
            Cx = c + idx_x

            # Center coordinates output and normalized
            tx = (box[..., 0])
            ty = (box[..., 1])
            tx_n = self.sigmoid(tx)
            ty_n = self.sigmoid(ty)

            # Calculate bx & by and normalize it
            bx = tx_n + Cx
            by = ty_n + Cy
            bx /= grid_width
            by /= grid_height

            # Calculate tw & th
            tw = (box[..., 2])
            th = (box[..., 3])
            tw_t = np.exp(tw)
            th_t = np.exp(th)

            # Anchors box dimension
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # Calculate bw & bh and normalize
            bw = pw * tw_t
            bh = ph * th_t
            # input size
            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value
            bw /= input_width
            bh /= input_height

            # Corner coordinates
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            # Adjust scale
            box[..., 0] = x1 * img_width
            box[..., 1] = y1 * img_height
            box[..., 2] = x2 * img_width
            box[..., 3] = y2 * img_height

        return boxes, box_confidences, box_class_probs
