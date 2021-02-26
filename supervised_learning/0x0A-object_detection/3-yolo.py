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
            the Darknet model, listed in order of idx, can be found
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

    # Public method
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Public method to filter the boxes

        Arguments:
         - boxes: a list of numpy.ndarrays of shape
             (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output
         - box_confidences: a list of numpy.ndarrays of shape
             (grid_height, grid_width, anchor_boxes, 1)
            containing the processed box confidences for each output
         - box_class_probs: a list of numpy.ndarrays of shape
             (grid_height, grid_width, anchor_boxes, classes)
            containing the processed box class probabilities for each output
        Returns:
         A tuple of (filtered_boxes, box_classes, box_scores):
         * filtered_boxes: a numpy.ndarray of shape (?, 4) containing
            all of the filtered bounding boxes:
         * box_classes: a numpy.ndarray of shape (?,) containing
            the class number that each box in filtered_boxes predicts,
            respectively
         * box_scores: a numpy.ndarray of shape (?) containing
            the box scores for each box in filtered_boxes, respectively
        """

        scores = []

        for box_conf, box_class_prob in zip(box_confidences, box_class_probs):
            scores.append(box_conf * box_class_prob)

        # box_scores
        box_scores = [score.max(axis=-1) for score in scores]
        box_scores = [box.reshape(-1) for box in box_scores]
        box_scores = np.concatenate(box_scores)
        filtering_mask = np.where(box_scores < self.class_t)
        box_scores = np.delete(box_scores, filtering_mask)

        # box_classes
        box_classes = [box.argmax(axis=-1) for box in scores]
        box_classes = [box.reshape(-1) for box in box_classes]
        box_classes = np.concatenate(box_classes)
        box_classes = np.delete(box_classes, filtering_mask)

        # filtered_boxes
        filtered_boxes_list = [box.reshape(-1, 4) for box in boxes]
        filtered_boxes_box = np.concatenate(filtered_boxes_list, axis=0)
        filtered_boxes = np.delete(filtered_boxes_box, filtering_mask, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    @staticmethod
    def iou(b1, b2):
        """
        Method to calculate intersection over union
        (x1, y1, x2, y2)
        """
        xi1 = max(b1[0], b2[0])
        yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2])
        yi2 = min(b1[3], b2[3])

        intersection = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

        b1_area = (b1[3] - b1[1]) * (b1[2] - b1[0])
        b2_area = (b2[3] - b2[1]) * (b2[2] - b2[0])

        union = b1_area + b2_area - intersection

        return intersection / union

    # Public method
    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Public method to select the boxes to keep by its score idx

        Arguments:
         - filtered_boxes: a numpy.ndarray of shape (?, 4)
            containing all of the filtered bounding boxes:
         - box_classes: a numpy.ndarray of shape (?,)
            containing the class number for the class that filtered_boxes
            predicts, respectively
         - box_scores: a numpy.ndarray of shape (?)
            containing the box scores for each box in filtered_boxes,
            respectively

        Returns:
         A tuple of
           (box_predictions, predicted_box_classes, predicted_box_scores):
         - box_predictions: a numpy.ndarray of shape (?, 4) containing all of
            the predicted bounding boxes ordered by class and box score
         - predicted_box_classes: a numpy.ndarray of shape (?,)
            containing the class number for box_predictions
            ordered by class and box score, respectively
         - predicted_box_scores: a numpy.ndarray of shape (?)
            containing the box scores for box_predictions
            ordered by class and box score, respectively
        """

        idx = np.lexsort((-box_scores, box_classes))

        box_predictions = np.array([filtered_boxes[i] for i in idx])
        predicted_box_classes = np.array([box_classes[i] for i in idx])
        predicted_box_scores = np.array([box_scores[i] for i in idx])

        _, c_class = np.unique(predicted_box_classes, return_counts=True)

        i = 0
        counter = 0

        for class_count in c_class:
            while i < counter + class_count:
                j = i + 1
                while j < counter + class_count:
                    aux = self.iou(box_predictions[i],
                                   box_predictions[j])
                    if aux > self.nms_t:
                        box_predictions = np.delete(box_predictions, j,
                                                    axis=0)
                        predicted_box_scores = np.delete(predicted_box_scores,
                                                         j, axis=0)
                        predicted_box_classes = (np.delete
                                                 (predicted_box_classes,
                                                  j, axis=0))
                        class_count -= 1
                    else:
                        j += 1
                i += 1
            counter += class_count

        return box_predictions, predicted_box_classes, predicted_box_scores
