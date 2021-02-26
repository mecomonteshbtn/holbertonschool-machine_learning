# 0x0A. Object Detection

---
## Resources
*   [OpenCV](/rltoken/tkZxgeNLM20s4tjkHKLRwg "OpenCV")
*   [Getting Started with Images](/rltoken/mfnly92V8O3ZcPNUGpxsCw "Getting Started with Images")
*   [Object detection](/rltoken/ud_WF7iERPUIpi02YPE-Sw "Object detection")
*   [Object Localization](/rltoken/rn1h3S4bvjBz76SHLseHlw "Object Localization")
*   [Landmark Detection](/rltoken/6LpPD_Wa5EbToJFVdHUtCw "Landmark Detection")
*   [Object Detection](/rltoken/szw-X_x6oYVCz7DPpobAVg "Object Detection")
*   [Convolutional Implementation Sliding Windows](/rltoken/57_yh-oViUjMXK1zyn1d1g "Convolutional Implementation Sliding Windows")
*   [Intersection Over Union](/rltoken/UVplBgsNtqQZ4IfyUoGRbw "Intersection Over Union")
*   [Nonmax Suppression](/rltoken/5wf8fZq4Rt3eA6SNljI7Cw "Nonmax Suppression")
*   [Anchor Boxes](/rltoken/cSdJOW8Jv5mwlKCjsTr_yQ "Anchor Boxes")
*   [YOLO Algorithm](/rltoken/tnaTT6cX_ONkVKxm5uix1w "YOLO Algorithm")
*   [You Only Look Once: Unified, Real-Time Object Detection (CVPR 2016)](/rltoken/COCAXswObaL3IhdkOEA6lA "You Only Look Once: Unified, Real-Time Object Detection (CVPR 2016)")
*   [Real-time Object Detection with YOLO, YOLOv2 and now YOLOv3](/rltoken/s2jZaJ9HlLUyHLycBgYe4A "Real-time Object Detection with YOLO, YOLOv2 and now YOLOv3")
*   [What’s new in YOLO v3?](/rltoken/3TfvBlw8pGWyKKvsTw5gBg "What’s new in YOLO v3?")
*   [What do we learn from single shot object detectors (SSD, YOLOv3), FPN & Focal loss (RetinaNet)?](/rltoken/HVUKK7L6pUa5Ufb9nlu7tQ "What do we learn from single shot object detectors (SSD, YOLOv3), FPN & Focal loss (RetinaNet)?")
*   [mAP (mean Average Precision) for Object Detection](/rltoken/P44Kt078IXqpqebui_P9oA "mAP (mean Average Precision) for Object Detection")

---
## References
*   [cv2.imshow](/rltoken/vjPLuWAd6dBACK3A6fUT9w "cv2.imshow")
*   [cv2.imread](/rltoken/OcVoBXU5-pRzMuur8XmdlQ "cv2.imread")
*   [cv2.line](/rltoken/A0MC22w8VgTscZ9ISNA2Sw "cv2.line")
*   [cv2.putText](/rltoken/_gAdffr2iGn7kgHI2tyrNQ "cv2.putText")
*   [cv2.rectangle](/rltoken/Uxs3Amd8aNVmfhIzvAcPKA "cv2.rectangle")
*   [cv2.resize](/rltoken/GI8KVojF8QF9uxtNZ0wnyg "cv2.resize")
*   [glob](/rltoken/Zb1K6nDzDnffkoJEpOsm_Q "glob")
*   [You Only Look Once: Unified, Real-Time Object Detection](/rltoken/1U5BP38mwuW1sJdBmn8Thw "You Only Look Once: Unified, Real-Time Object Detection")
*   [YOLO9000: Better, Faster, Stronger](/rltoken/Otc9Uj5YguhQnPdcBo-krw "YOLO9000: Better, Faster, Stronger")
*   [YOLOv3: An Incremental Improvement](/rltoken/gRWmLovs6TJGPUHG-zTBsQ "YOLOv3: An Incremental Improvement")

---
## Advanced:
*   [Image segmentation](/rltoken/LkvWawxhxQ4hikicfEA7yw "Image segmentation")
*   [Region Proposals](/rltoken/nkx8Gnf9PWYon4qP53BqDg "Region Proposals")
*   [Understanding Feature Pyramid Networks for object detection (FPN)](/rltoken/7jwKRJifXqn7uixcviFxcQ "Understanding Feature Pyramid Networks for object detection (FPN)")
*   [What do we learn from region based object detectors (Faster R-CNN, R-FCN, FPN)?](/rltoken/2rao1xi0BO5DN4lobKXDGw "What do we learn from region based object detectors (Faster R-CNN, R-FCN, FPN)?")
*   [Object Detection for Dummies Part 3: R-CNN Family](/rltoken/eGfiNWIn94y7a3sKjLbeTw "Object Detection for Dummies Part 3: R-CNN Family")
*   [Image segmentation with Mask R-CNN](/rltoken/GIs2XhOLR2KQj05dVPc74A "Image segmentation with Mask R-CNN")

---
## Learning Objectives
At the end of this project, you are expected to be able to [explain to anyone](/rltoken/uA-2ZfuWIfA7EDqMCnku1A "explain to anyone"), **without the help of Google**:

---
### Description 
In this project we try to solve this questions:
- What is OpenCV and how do you use it?
- What is object detection?
- What is the Sliding Windows algorithm?
- What is a single-shot detector?
- What is the YOLO algorithm?
- What is IOU and how do you calculate it?
- What is non-max suppression?
- What are anchor boxes?
- What is mAP and how do you calculate it?

---
## Download and Use OpenCV 4.1.x
```
    alexa@ubuntu-xenial:~$ pip install --user opencv-python
    alexa@ubuntu-xenial:~$ python3
    >>> import cv2
    >>> cv2.__version__
    '4.1.0'
```
---
## Test Files
*   [yolo.h5](/rltoken/FlaRJJaIx3f-urZ0F8JZ9A "yolo.h5")
*   [coco_classes.txt](/rltoken/XERmxYdTXUuVlYTkxsJREw "coco_classes.txt")
*   [yolo_images.zip](/rltoken/9XrEpGalzYbbZV08qYdn1A "yolo_images.zip")

---
### Files
| File | Description |
| ------ | ------ |
| [0-yolo.py](0-yolo.py) | Class Yolo that uses the Yolo v3 algorithm to perform object detection. |
| [1-yolo.py](1-yolo.py) | Update the class Yolo, based on 0-yolo.py, adding the public method process_outputs. |
| [2-yolo.py](2-yolo.py) | Update the class Yolo, based on 1-yolo.py, adding the public method filter_boxes. |
| [3-yolo.py](3-yolo.py) | Update the class Yolo, based on 2-yolo.py, adding the public method non_max_suppression. |
| [4-yolo.py](4-yolo.py) | Update the class Yolo, based on 3-yolo.py, adding the static method load_images. |
| [5-yolo.py](5-yolo.py) | Update the class Yolo, based on 4-yolo.py, adding the public method preprocess_images. |
| [6-yolo.py](6-yolo.py) | Update the class Yolo, based on 5-yolo.py, adding the public method show_boxes. |
| [7-yolo.py](7-yolo.py) | Update the class Yolo, based on 6-yolo.py, adding the public method predict. |


### Build with
- Python (python 3.6)
- Numpy (numpy 1.15)
- Tensorflow (tensorflow 1.15)
- Ubuntu 20.04 LTS 

---
### [0. Initialize Yolo](./0-yolo.py)
Write a class `Yolo` that uses the Yolo v3 algorithm to perform object detection:
*   class constructor: `def __init__(self, model_path, classes_path, class_t, nms_t, anchors):`
    *   `model_path` is the path to where a Darknet Keras model is stored
    *   `classes_path` is the path to where the list of class names used for the Darknet model, listed in order of index, can be found
    *   `class_t` is a float representing the box score threshold for the initial filtering step
    *   `nms_t` is a float representing the IOU threshold for non-max suppression
    *   `anchors` is a `numpy.ndarray` of shape `(outputs, anchor_boxes, 2)` containing all of the anchor boxes:
        *   `outputs` is the number of outputs (predictions) made by the Darknet model
        *   `anchor_boxes` is the number of anchor boxes used for each prediction
        *   `2` => `[anchor_box_width, anchor_box_height]`
*   Public instance attributes:
    *   `model`: the Darknet Keras model
    *   `class_names`: a list of the class names for the model
    *   `class_t`: the box score threshold for the initial filtering step
    *   `nms_t`: the IOU threshold for non-max suppression
    *   `anchors`: the anchor boxes
```
    root@alexa-ml2:~/0x0A-object_detection# cat 0-main.py 
    #!/usr/bin/env python3

    if __name__ == '__main__':
        import numpy as np
        Yolo = __import__('0-yolo').Yolo

        np.random.seed(0)
        anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[10, 13], [16, 30], [33, 23]]])
        yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
        yolo.model.summary()
        print('Class names:', yolo.class_names)
        print('Class threshold:', yolo.class_t)
        print('NMS threshold:', yolo.nms_t)
        print('Anchor boxes:', yolo.anchors)
    root@alexa-ml2:~/0x0A-object_detection# ./0-main.py 
    WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, 416, 416, 3)  0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 416, 416, 32) 864         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 416, 416, 32) 128         conv2d[0][0]                     
    __________________________________________________________________________________________________

    ...

    reshape (Reshape)               (None, 13, 13, 3, 85 0           conv2d_58[0][0]                  
    __________________________________________________________________________________________________
    reshape_1 (Reshape)             (None, 26, 26, 3, 85 0           conv2d_66[0][0]                  
    __________________________________________________________________________________________________
    reshape_2 (Reshape)             (None, 52, 52, 3, 85 0           conv2d_74[0][0]                  
    ==================================================================================================
    Total params: 62,001,757
    Trainable params: 61,949,149
    Non-trainable params: 52,608
    __________________________________________________________________________________________________
    Class names: ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    Class threshold: 0.6
    NMS threshold: 0.5
    Anchor boxes: [[[116  90]
      [156 198]
      [373 326]]

     [[ 30  61]
      [ 62  45]
      [ 59 119]]

     [[ 10  13]
      [ 16  30]
      [ 33  23]]]
    root@alexa-ml2:~/0x0A-object_detection#
```

### [1. Process Outputs](./1-yolo.py)
Write a class `Yolo` (Based on `0-yolo.py`):
*   Add the public method `def process_outputs(self, outputs, image_size):`
    *   `outputs` is a list of `numpy.ndarray`s containing the predictions from the Darknet model for a single image:
        *   Each output will have the shape `(grid_height, grid_width, anchor_boxes, 4 + 1 + classes)`
            *   `grid_height` & `grid_width` => the height and width of the grid used for the output
            *   `anchor_boxes` => the number of anchor boxes used
            *   `4` => `(t_x, t_y, t_w, t_h)`
            *   `1` => `box_confidence`
            *   `classes` => class probabilities for all classes
    *   `image_size` is a `numpy.ndarray` containing the image’s original size `[image_height, image_width]`
    *   Returns a tuple of `(boxes, box_confidences, box_class_probs)`:
        *   `boxes`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, 4)` containing the processed boundary boxes for each output, respectively:
            *   `4` => `(x1, y1, x2, y2)`
            *   `(x1, y1, x2, y2)` should represent the boundary box relative to original image
        *   `box_confidences`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, 1)` containing the box confidences for each output, respectively
        *   `box_class_probs`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, classes)` containing the box’s class probabilities for each output, respectively

_HINT: The Darknet model is an input to the class for a reason. It may not always have the same number of outputs, input sizes, etc._
```
    root@alexa-ml2:~/0x0A-object_detection# cat 1-main.py 
    #!/usr/bin/env python3

    if __name__ == '__main__':
        import numpy as np
        Yolo = __import__('1-yolo').Yolo

        np.random.seed(0)
        anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[10, 13], [16, 30], [33, 23]]])
        yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
        output1 = np.random.randn(13, 13, 3, 85)
        output2 = np.random.randn(26, 26, 3, 85)
        output3 = np.random.randn(52, 52, 3, 85)
        boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
        print('Boxes:', boxes)
        print('Box confidences:', box_confidences)
        print('Box class probabilities:', box_class_probs)
    root@alexa-ml2:~/0x0A-object_detection# ./1-main.py 
    WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
    Boxes: [array([[[[-2.13743365e+02, -4.85478868e+02,  3.05682061e+02,
               5.31534670e+02],
             [-6.28222336e+01, -1.13713822e+01,  1.56452678e+02,
               7.01966357e+01],
             [-7.00753664e+02, -7.99011810e+01,  7.77777040e+02,
               1.24440730e+02]],

                    ...

            [[ 5.61142819e+02,  3.07685110e+02,  7.48124593e+02,
               6.41890468e+02],
             [ 5.80033260e+02,  2.88627445e+02,  7.62480440e+02,
               6.47683922e+02],
             [ 3.58437752e+02,  2.46899004e+02,  9.55713550e+02,
               7.08803200e+02]]]]), array([[[[-1.23432804e+01, -3.22999818e+02,  2.20307323e+01,
               3.46429534e+02],
             [-1.80604912e+01, -7.11557928e+00,  3.29117181e+01,
               3.76250584e+01],
             [-1.60588925e+02, -1.42683911e+02,  1.73612755e+02,
               1.73506691e+02]],

                    ...

            [[ 6.68829175e+02,  4.87753124e+02,  6.82031480e+02,
               5.03135980e+02],
             [ 4.34438370e+02,  4.78337823e+02,  9.46316108e+02,
               5.09664332e+02],
             [ 6.22075514e+02,  4.82103466e+02,  7.35368626e+02,
               4.95580409e+02]]]]), array([[[[-2.69962831e+00, -7.76450289e-01,  1.24050569e+01,
               2.78936573e+00],
             [-4.22907727e+00, -1.06544368e+01,  1.17677684e+01,
               2.57860099e+01],
             [-7.89233952e+01, -2.81791298e+00,  8.90320793e+01,
               1.29705044e+01]],

                    ...

            [[ 6.76849818e+02,  4.46289490e+02,  7.00081017e+02,
               5.41566130e+02],
             [ 6.66112867e+02,  4.81296831e+02,  7.28887543e+02,
               5.01093787e+02],
             [ 6.29818872e+02,  4.58019128e+02,  7.50570597e+02,
               5.32394185e+02]]]])]
    Box confidences: [array([[[[0.86617546],
             [0.74162884],
             [0.26226237]],

                    ...

            [[0.75932849],
             [0.53997516],
             [0.54609635]]]]), array([[[[0.56739016],
             [0.88809651],
             [0.58774731]],

                    ...

            [[0.58788139],
             [0.3982885 ],
             [0.56051201]]]]), array([[[[0.73336558],
             [0.57887604],
             [0.59277021]],

                    ...

            [[0.19774838],
             [0.69947049],
             [0.36158221]]]])]
    Box class probabilities: [array([[[[0.27343225, 0.72113296, 0.46223277, ..., 0.6143566 ,
              0.177082  , 0.81581579],
             [0.40054928, 0.77249355, 0.55188134, ..., 0.17584277,
              0.76638851, 0.57857896],
             [0.66409448, 0.30929663, 0.33413323, ..., 0.53542882,
              0.42083943, 0.66630914]],

                    ...

            [[0.48321664, 0.22789979, 0.6691948 , ..., 0.53945861,
              0.30098012, 0.72069712],
             [0.45435778, 0.68714784, 0.60871616, ..., 0.51156461,
              0.14100029, 0.38924579],
             [0.14533179, 0.91481357, 0.46558833, ..., 0.29158615,
              0.26098354, 0.6078719 ]]]]), array([[[[0.16748018, 0.74425493, 0.77271059, ..., 0.82238314,
              0.45290514, 0.35835817],
             [0.35157962, 0.88467242, 0.18324688, ..., 0.63359015,
              0.40203054, 0.48139226],
             [0.45924026, 0.81674653, 0.68472278, ..., 0.45661086,
              0.13878263, 0.58812507]],

                    ...

            [[0.34600385, 0.2430844 , 0.82184407, ..., 0.64286074,
              0.2806551 , 0.10224861],
             [0.89692404, 0.22950708, 0.46779974, ..., 0.6787069 ,
              0.25042145, 0.63684789],
             [0.36853257, 0.67040213, 0.4840967 , ..., 0.4530742 ,
              0.20817072, 0.59335632]]]]), array([[[[0.5651767 , 0.5976206 , 0.38176215, ..., 0.27546563,
              0.31863509, 0.31522224],
             [0.34311079, 0.35702272, 0.52498233, ..., 0.67677487,
              0.13292343, 0.79922556],
             [0.31009487, 0.12745849, 0.45797284, ..., 0.20871353,
              0.60219055, 0.29796099]],

                    ...

            [[0.24181667, 0.17739203, 0.77300902, ..., 0.59600114,
              0.3446732 , 0.75570862],
             [0.53289017, 0.31652626, 0.11990411, ..., 0.45424862,
              0.2372848 , 0.26196449],
             [0.40272803, 0.24201719, 0.80157031, ..., 0.2338579 ,
              0.18169015, 0.36450041]]]])]
    root@alexa-ml2:~/0x0A-object_detection#
```

### [2. Filter Boxes](./2-yolo.py)
Write a class `Yolo` (Based on `1-yolo.py`):
*   Add the public method `def filter_boxes(self, boxes, box_confidences, box_class_probs):`
    *   `boxes`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, 4)` containing the processed boundary boxes for each output, respectively
    *   `box_confidences`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, 1)` containing the processed box confidences for each output, respectively
    *   `box_class_probs`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, classes)` containing the processed box class probabilities for each output, respectively
    *   Returns a tuple of `(filtered_boxes, box_classes, box_scores)`:
        *   `filtered_boxes`: a `numpy.ndarray` of shape `(?, 4)` containing all of the filtered bounding boxes:
        *   `box_classes`: a `numpy.ndarray` of shape `(?,)` containing the class number that each box in `filtered_boxes` predicts, respectively
        *   `box_scores`: a `numpy.ndarray` of shape `(?)` containing the box scores for each box in `filtered_boxes`, respectively
```
    root@alexa-ml2:~/0x0A-object_detection# cat 2-main.py 
    #!/usr/bin/env python3

    if __name__ == '__main__':
        import numpy as np
        Yolo = __import__('2-yolo').Yolo

        np.random.seed(0)
        anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[10, 13], [16, 30], [33, 23]]])
        yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
        output1 = np.random.randn(13, 13, 3, 85)
        output2 = np.random.randn(26, 26, 3, 85)
        output3 = np.random.randn(52, 52, 3, 85)
        boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
        boxes, box_classes, box_scores = yolo.filter_boxes(boxes, box_confidences, box_class_probs)
        print('Boxes:', boxes)
        print('Box classes:', box_classes)
        print('Box scores:', box_scores)
    root@alexa-ml2:~/0x0A-object_detection# ./2-main.py
    WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
    Boxes: [[-213.74336488 -485.47886784  305.68206077  531.53467019]
     [ -62.82223363  -11.37138215  156.45267787   70.19663572]
     [ 190.62733946    7.65943712  319.201764     43.75737906]
     ...
     [ 647.78041714  491.58472667  662.00736941  502.60750466]
     [ 586.27543101  487.95333873  715.85860922  499.39422783]
     [ 666.1128673   481.29683099  728.88754319  501.09378706]]
    Box classes: [19 54 29 ... 63 25 46]
    Box scores: [0.7850503  0.67898563 0.81301861 ... 0.8012832  0.61427808 0.64562072]
    root@alexa-ml2:~/0x0A-object_detection# 
```

### [3. Non-max Suppression](./3-yolo.py)
Write a class `Yolo` (Based on `2-yolo.py`):
*   Add the public method `def non_max_suppression(self, filtered_boxes, box_classes, box_scores):`
    *   `filtered_boxes`: a `numpy.ndarray` of shape `(?, 4)` containing all of the filtered bounding boxes:
    *   `box_classes`: a `numpy.ndarray` of shape `(?,)` containing the class number for the class that `filtered_boxes` predicts, respectively
    *   `box_scores`: a `numpy.ndarray` of shape `(?)` containing the box scores for each box in `filtered_boxes`, respectively
    *   Returns a tuple of `(box_predictions, predicted_box_classes, predicted_box_scores)`:
        *   `box_predictions`: a `numpy.ndarray` of shape `(?, 4)` containing all of the predicted bounding boxes ordered by class and box score
        *   `predicted_box_classes`: a `numpy.ndarray` of shape `(?,)` containing the class number for `box_predictions` ordered by class and box score, respectively
        *   `predicted_box_scores`: a `numpy.ndarray` of shape `(?)` containing the box scores for `box_predictions` ordered by class and box score, respectively
```
    root@alexa-ml2:~/0x0A-object_detection# cat 3-main.py 
    #!/usr/bin/env python3

    if __name__ == '__main__':
        import numpy as np
        Yolo = __import__('3-yolo').Yolo

        np.random.seed(0)
        anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[10, 13], [16, 30], [33, 23]]])
        yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
        output1 = np.random.randn(13, 13, 3, 85)
        output2 = np.random.randn(26, 26, 3, 85)
        output3 = np.random.randn(52, 52, 3, 85)
        boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
        boxes, box_classes, box_scores = yolo.filter_boxes(boxes, box_confidences, box_class_probs)
        boxes, box_classes, box_scores = yolo.non_max_suppression(boxes, box_classes, box_scores)
        print('Boxes:', boxes)
        print('Box classes:', box_classes)
        print('Box scores:', box_scores)
    root@alexa-ml2:~/0x0A-object_detection# ./3-main.py
    WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
    Boxes: [[ 483.49145347  128.010205    552.78146847  147.87465464]
     [ -38.91328475  332.66704009  102.94594841  363.78584864]
     [  64.10861893  329.13266621  111.87941603  358.37523958]
     ...
     [ -30.48917643  444.88068667   52.18360562  467.77199113]
     [-111.73190894  428.19222574  175.04566042  483.40040996]
     [ 130.0729606   467.20024928  172.42160784  515.90336094]]
    Box classes: [ 0  0  0 ... 79 79 79]
    Box scores: [0.80673525 0.80405611 0.78972362 ... 0.6329012  0.61789273 0.61758194]
    root@alexa-ml2:~/0x0A-object_detection#
```

### [4. Load images](./4-yolo.py)
Write a class `Yolo` (Based on `3-yolo.py`):
*   Add the static method `def load_images(folder_path):`
    *   `folder_path`: a string representing the path to the folder holding all the images to load
    *   Returns a tuple of `(images, image_paths)`:
        *   `images`: a list of images as `numpy.ndarray`s
        *   `image_paths`: a list of paths to the individual images in `images`
```
    root@alexa-ml2:~/0x0A-object_detection# cat 4-main.py 
    #!/usr/bin/env python3

    if __name__ == '__main__':
        import cv2
        import numpy as np
        Yolo = __import__('4-yolo').Yolo

        np.random.seed(0)
        anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[10, 13], [16, 30], [33, 23]]])
        yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
        images, image_paths = yolo.load_images('../data/yolo')
        i = np.random.randint(0, len(images))
        cv2.imshow(image_paths[i], images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    root@alexa-ml2:~/0x0A-object_detection# ./4-main.py
    WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
```
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/3/aa9478ba935167ab5266.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200629%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200629T125202Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4ffcdce94ba10297070d84b43aa644a20672bff648900e31f301445aea4096e1)


### [5. Preprocess images](./5-yolo.py)
Write a class `Yolo` (Based on `4-yolo.py`):
*   Add the public method `def preprocess_images(self, images):`
    *   `images`: a list of images as `numpy.ndarray`s
    *   Resize the images with inter-cubic interpolation
    *   Rescale all images to have pixel values in the range `[0, 1]`
    *   Returns a tuple of `(pimages, image_shapes)`:
        *   `pimages`: a `numpy.ndarray` of shape `(ni, input_h, input_w, 3)` containing all of the preprocessed images
            *   `ni`: the number of images that were preprocessed
            *   `input_h`: the input height for the Darknet model _Note: this can vary by model_
            *   `input_w`: the input width for the Darknet model _Note: this can vary by model_
            *   `3`: number of color channels
        *   `image_shapes`: a `numpy.ndarray` of shape `(ni, 2)` containing the original height and width of the images
            *   `2` => `(image_height, image_width)`
```
    root@alexa-ml2:~/0x0A-object_detection# cat 5-main.py 
    #!/usr/bin/env python3

    if __name__ == '__main__':
        import cv2
        import numpy as np
        Yolo = __import__('5-yolo').Yolo

        np.random.seed(0)
        anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[10, 13], [16, 30], [33, 23]]])
        yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
        images, image_paths = yolo.load_images('../data/yolo')
        pimages, image_shapes = yolo.preprocess_images(images)
        print(type(pimages), pimages.shape)
        print(type(image_shapes), image_shapes.shape)
        i = np.random.randint(0, len(images))
        print(images[i].shape, ':', image_shapes[i])
        cv2.imshow(image_paths[i], pimages[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    root@alexa-ml2:~/0x0A-object_detection# ./5-main.py 
    WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
    <class 'numpy.ndarray'> (6, 416, 416, 3)
    <class 'numpy.ndarray'> (6, 2)
    (576, 768, 3) : [576 768]
```
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/3/a9c7e8c7ce308954fa1d.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200629%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200629T125202Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=8a8491c77c75040f85abf176abcd31768201149cd80f881bc3fe4c37d73d3454)


### [6. Show boxes](./6-yolo.py)
Write a class `Yolo` (Based on `5-yolo.py`):
*   Add the public method `def show_boxes(self, image, boxes, box_classes, box_scores, file_name):`
    *   `image`: a `numpy.ndarray` containing an unprocessed image
    *   `boxes`: a `numpy.ndarray` containing the boundary boxes for the image
    *   `box_classes`: a `numpy.ndarray` containing the class indices for each box
    *   `box_scores`: a `numpy.ndarray` containing the box scores for each box
    *   `file_name`: the file path where the original image is stored
    *   Displays the image with all boundary boxes, class names, and box scores _(see example below)_
        *   Boxes should be drawn as with a blue line of thickness 2
        *   Class names and box scores should be drawn above each box in red
            *   Box scores should be rounded to 2 decimal places
            *   Text should be written 5 pixels above the top left corner of the box
            *   Text should be written in `FONT_HERSHEY_SIMPLEX`
            *   Font scale should be 0.5
            *   Line thickness should be 1
            *   You should use `LINE_AA` as the line type
        *   The window name should be the same as `file_name`
        *   If the `s` key is pressed:
            *   The image should be saved in the directory `detections`, located in the current directory
            *   If `detections` does not exist, create it
            *   The saved image should have the file name `file_name`
            *   The image window should be closed
        *   If any key besides `s` is pressed, the image window should be closed without saving
```
    root@alexa-ml2:~/0x0A-object_detection# cat 6-main.py
    #!/usr/bin/env python3

    if __name__ == '__main__':
        import numpy as np
        Yolo = __import__('6-yolo').Yolo

        np.random.seed(0)
        anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[10, 13], [16, 30], [33, 23]]])
        yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
        images, image_paths = yolo.load_images('../data/yolo')
        boxes = np.array([[119.22100287, 118.62197718, 567.75985556, 440.44121152],
                          [468.53530752, 84.48338278, 696.04923556, 167.98947829],
                          [124.2043716, 220.43365057, 319.4254314 , 542.13706101]])
        box_scores = np.array([0.99537075, 0.91536146, 0.9988506])
        box_classes = np.array([1, 7, 16])
        ind = 0
        for i, name in enumerate(image_paths):
            if "dog.jpg" in name:
                ind = i
                break
        yolo.show_boxes(images[i], boxes, box_classes, box_scores, "dog.jpg")
    root@alexa-ml2:~/0x0A-object_detection# ./6-main.py
    WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/7/e49512524f47b4138d85.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200629%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200629T125202Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=ce520b1e8d7f14ea242538a0d8aadbcfe37627ce39ca34f6609fa6c0d24ebcdb)

_press the `s` button_

    root@alexa-ml2:~/0x0A-object_detection# ls detections
    dog.jpg
    root@alexa-ml2-1:~/0x0A-object_detection# 
```

### [7. Predict](./7-yolo.py)
Write a class `Yolo` (Based on `6-yolo.py`):
*   Add the public method `def predict(self, folder_path):`
    *   `folder_path`: a string representing the path to the folder holding all the images to predict
    *   All image windows should be named after the corresponding image filename without its full path_(see examples below)_
    *   Displays all images using the `show_boxes` method
    *   Returns: a tuple of `(predictions, image_paths)`:
        *   `predictions`: a list of tuples for each image of `(boxes, box_classes, box_scores)`
        *   `image_paths`: a list of image paths corresponding to each prediction in `predictions`
```
    root@alexa-ml2:~/0x0A-object_detection# cat 7-main.py 
    #!/usr/bin/env python3

    if __name__ == '__main__':
        import numpy as np
        Yolo = __import__('7-yolo').Yolo

        np.random.seed(0)
        anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[10, 13], [16, 30], [33, 23]]])
        yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
        predictions, image_paths = yolo.predict('../data/yolo')
        for i, name in enumerate(image_paths):
            if "dog.jpg" in name:
                ind = i
                break
        print(image_paths[ind])
        print(predictions[ind])
    root@alexa-ml2:~/0x0A-object_detection# ./7-main.py
    WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
```
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/7/e49512524f47b4138d85.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200629%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200629T125202Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=ce520b1e8d7f14ea242538a0d8aadbcfe37627ce39ca34f6609fa6c0d24ebcdb)

_press `s`_

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/7/c29e4d9c8824409119ea.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200629%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200629T125202Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=8786699c9e4794098156dc395d20184b3eda8af0c58c02c06827a554edfaddaf)

_press `s`_

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/7/4f4005adbd3b0fb3c278.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200629%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200629T125202Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=cff2175b5723a32705d2e02a3828c551f0772165fbd15e034f1db3dda7327144)

_press `s`_

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/7/0fa13e9c53b81b1c4fd3.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200629%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200629T125202Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=6e53ad61a1db601504e1f935ee5545a98cf8be43cdb4ac1c3141ffc762c96c3b)

_press `s`_

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/7/d39a47507bbe27c2a794.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200629%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200629T125202Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=9e3c36adbbcd752b8446ee077b0b2fcb784724655ce7e0e78780d53c2f01f6a0)

_press `s`_

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/7/cfdf1e5718c91185a8f6.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200629%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200629T125202Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=ffbfffff5c1aca49f3be2fc759fe7ab2d801e6b2c827b5447fe248ee063bdd55)

_press `s`_
```
    ../data/yolo/dog.jpg
    (array([[119.22100287, 118.62197718, 567.75985556, 440.44121152],
           [468.53530752,  84.48338278, 696.04923556, 167.98947829],
           [124.2043716 , 220.43365057, 319.4254314 , 542.13706101]]), array([ 1,  7, 16]), array([0.99537075, 0.91536146, 0.9988506 ], dtype=float32))
    root@alexa-ml2:~/0x0A-object_detection#  ls detections
    dog.jpg  eagle.jpg  giraffe.jpg  horses.jpg  person.jpg  takagaki.jpg
    root@alexa-ml2-1:~/0x0A-object_detection# 
```
---
## Authors

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
