import cv2
import numpy as np
import os
from os import listdir
import tensorflow as tf
import sys

sys.path.append("..")
from research.object_detection.utils import label_map_util
from research.object_detection.utils import visualization_utils as vis_util

MODEL_NAME='research/ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_IMAGES_DIR= 'Test_Images'
IMAGE_NAME = [os.path.join(PATH_TO_IMAGES_DIR,'inputImage{}.jpg'.format(i)) for i in range(1,2)]

CWD_PATH = os.getcwd()
print(CWD_PATH)
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

print(PATH_TO_CKPT)
PATH_TO_LABELS = os.path.join(CWD_PATH,'research/data', 'labelmap.pbtxt')
print(PATH_TO_LABELS)
PATH_TO_IMAGE =IMAGE_NAME
print(PATH_TO_IMAGE)

NUM_CLASSES = 12

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
category_index = label_map_util.create_category_index(categories)
class_names_mapping = {1: "Aeroplane", 2: "auto", 3: "boat", 4: "bike", 5: "bus", 6: "bicycle", 7: "Car", 8: "ship", 9: "Train",
            10: "scooty", 11: "Truck", 12: "Aeroplalne"}

# Load the Tensorflow model into memory.

detection_graph= tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    for image in IMAGE_NAME:

        image = cv2.imread(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        result = scores.flatten()
        res = []
        for idx in range(0,len(result)):
            if result[idx] > .40:
                res.append(idx)
        print(res)
        top_classes = classes.flatten()
        res_list = [top_classes[i] for i in res]
        print(res_list)

        class_final_names =  [ class_names_mapping[x] for x in res_list]
        print(class_final_names)

        top_scores = [e for l2 in scores for e in l2 if e > 0.30]
        print(top_scores)

        new_scores = scores.flatten()
        print(new_scores)
        new_boxes =boxes.reshape(300,4)
        print(new_boxes)
        max_boxes_to_draw = new_boxes.shape[0]
        print(max_boxes_to_draw)
        min_score_thresh = .30

        listofOutput = []
        for (name,score,i) in zip(class_final_names,top_scores,range(min(max_boxes_to_draw, new_boxes.shape[0]))):
            valdict = {}
            valdict['className'] = name
            valdict['score'] = str(score)
            if new_scores is None or new_scores [i] > min_score_thresh:
                val = list(new_boxes[i])
                valdict["yMin"] = str(val[0])
                valdict["xMin"] = str(val[1])
                valdict["yMax"] = str(val[2])
                valdict["xMax"] = str(val[3])
                listofOutput.append(valdict)
            print(valdict)
        # Draw the results of the detection
        # (aka 'visulaize the results')

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

        # All the results have been drawn on image. Now display the image.
        cv2.imshow('Object detector', image)
        #print(len(scores.flatten()))
        #print(len(classes.flatten()))

        # Press any key to close the image
        cv2.waitKey(0)

        # Clean up
        cv2.destroyAllWindows()












