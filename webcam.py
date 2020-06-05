import os
import sys
import numpy as np
from os import  listdir
import cv2
import tensorflow as tf
from  research.object_detection.utils import label_map_util
from research.object_detection.utils import visualization_utils as vis_util


MODEL_NAME= 'research/ssd_mobilenet_v1_coco_2017_11_17'
CWD_PATH= os.getcwd()
print(CWD_PATH)
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME, 'frozen_inference_graph.pb')
print(PATH_TO_CKPT)
PATH_TO_LABELS = os.path.join(CWD_PATH ,'research/data', 'labelmap.pbtxt' )
print(PATH_TO_LABELS)
NUM_CLASSES = 12

lable_map=label_map_util.load_labelmap(PATH_TO_LABELS)
category= label_map_util.convert_label_map_to_categories(lable_map,max_num_classes=NUM_CLASSES,
                                                         use_display_name=True)
category_index = label_map_util.create_category_index(category)

print("Load the Tensorflow model into memory.")

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT,'rb') as fid:
        Serialized_graph = fid.read()
        od_graph_def.ParseFromString(Serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)
    print(sess)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
print(image_tensor)
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
print(detection_boxes)
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
print(detection_scores)
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
print(detection_classes)
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
print(num_detections)


video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

while (True):
    ret,frame = video.read()
    frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis= 0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes,detection_scores,detection_classes,num_detections],
                                             feed_dict={image_tensor:frame_expanded})
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(frame, np.squeeze(boxes),np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8,min_score_thresh=0.60)


    cv2.imshow("Object_detection", frame)
    print(image_tensor)
    print(detection_boxes)
    print(detection_classes)
    print(num_detections)
    if cv2.waitKey(1) == ord('q'):
        break



video.release()
cv2.destroyAllWindows()



