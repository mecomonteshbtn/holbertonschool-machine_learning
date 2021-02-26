#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 08:00:42 2021

@author: Robinson Montes
"""
import cv2
import glob
import os
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

            # Anchors box dimeensionension
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

    @staticmethod
    def load_images(folder_path):
        """
        Static method to load the images form a path

        Arguments:
         - folder_path: a string representing the path to the folder
            holding all the images to load

        Returns:
         A tuple of (images, image_paths):
             * images: a list of images as numpy.ndarrays
             * image_paths: a list of paths to the individual images in images
        """

        path = folder_path + '/*'
        image_paths = glob.glob(path, recursive=False)

        images = [cv2.imread(image) for image in image_paths]

        return images, image_paths

    # Public method
    def preprocess_images(self, images):
        """
        Public method to process images

        Arguments:
         - images: a list of images as numpy.ndarrays

        Returns:
         A tuple of (pimages, image_shapes):
            - pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                containing all of the preprocessed images
                * ni: the number of images that were preprocessed
                * input_h: the input height for the Darknet model
                * input_w: the input width for the Darknet model
                * 3: number of color channels
            - image_shapes: a numpy.ndarray of shape (ni, 2)
                containing the original height and width of the images
                * 2 => (image_height, image_width)
        """

        input_w = self.model.input.shape[1].value
        input_h = self.model.input.shape[2].value

        lpimages = []
        limage_shapes = []

        for img in images:
            # save original image size
            img_shape = img.shape[0], img.shape[1]
            limage_shapes.append(img_shape)

            # Resize the images with inter-cubic interpolation
            dimension = (input_w, input_h)
            resized = cv2.resize(img, dimension,
                                 interpolation=cv2.INTER_CUBIC)

            # Rescale all images to have pixel values in the range [0, 1]
            pimage = resized / 255
            lpimages.append(pimage)

        pimages = np.array(lpimages)
        image_shapes = np.array(limage_shapes)

        return pimages, image_shapes

    # Public method
    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Public method that displays a image with all boundary boxes,
        class names and box scores.

        Arguments:
         - image: a numpy.ndarray containing an unprocessed image
         - boxes: a numpy.ndarray containing the boundary boxes for the image
         - box_classes: a numpy.ndarray containing the class indices
            for each box
         - box_scores: a numpy.ndarray containing the box scores for each box
         - file_name: the file path where the original image is stored
            If the s key is pressed:
                The image should be saved in the directory detections,
                located in the current directory
            If any key besides s is pressed, the image window should be
            closed without saving
        """

        box_scores_r = np.around(box_scores, decimals=2)
        for i, box in enumerate(boxes):
            x, y, w, h = box
            txt_class = self.class_names[box_classes[i]]
            txt_score = str(box_scores_r[i])
            start_r = (int(x), int(h))
            start_text = (int(x) + 1, int(y) - 5)
            end_r = (int(w), int(y))

            cv2.rectangle(image, start_r, end_r, (255, 0, 0), 2)
            title = txt_class + ' ' + txt_score
            cv2.putText(image, title, start_text, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            if not os.path.exists('./detections'):
                os.makedirs('./detections')
            cv2.imwrite('./detections/' + file_name, image)
        cv2.destroyAllWindows()

    # Public method
    def predict(self, folder_path):
        """
        Public method to object detection prediction

        Arguments:
        - folder_path: a string representing the path to the folder
            holding all the images to predict

        Returns:
         A tuple of (predictions, image_paths):
            - predictions: a list of tuples for each image of
                (boxes, box_classes, box_scores)
            - image_paths: a list of image paths corresponding to each
                prediction in predictions
        """

        predictions = []
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        out = self.model.predict(pimages)

        for i, img in enumerate(images):
            outputs = [out[0][i, :, :, :, :],
                       out[1][i, :, :, :, :],
                       out[2][i, :, :, :, :]]

            img_dim = np.array([img.shape[0], img.shape[1]])
            bx, bx_conf, bx_cls_prob = self.process_outputs(outputs, img_dim)
            boxes, box_classes, box_scores = self.filter_boxes(bx, bx_conf,
                                                               bx_cls_prob)
            boxes, box_classes, box_scores = self.non_max_suppression(
                boxes, box_classes, box_scores)
            name = image_paths[i].split('/')[-1]
            self.show_boxes(img, boxes, box_classes, box_scores, name)
            predictions.append((boxes, box_classes, box_scores))

        return (predictions, image_paths)
