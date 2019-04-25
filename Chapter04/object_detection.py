'''
NOTE:
put all the boxes in tuple --> cv2 only takes tuples
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--im_path', type=str, help='path to input image')
#parser.add_argument('--save_path', type=str, help='save path to output image')
args = parser.parse_args()
IM_PATH = args.im_path

def read_image(imPath):
	img = cv2.imread(imPath)
	return img



def save_bounding_boxes(img, bb, save_path, im_name):
	roi = img[bbox[1]:bbox[3], bbox[0]:bbox[2]] # from x1 to x2 and y1 to y2
	#print(roi.shape)
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	cv2.imwrite(os.path.join(save_path, im_name), roi)


# the path to checkpoint file
FROZEN_GRAPH_FILE = 'frozen_inference_graph.pb'  #path to frozen graph

# load the model

# making an empty graph
graph = tf.Graph()
with graph.as_default():

	serialGraph = tf.GraphDef()
	# we will create a serialized graph as the Protobuf (for which the extension of file is .pb)
	# needs to be read serially in a serial graph
	# we will transfer it later to the empty graph created

	with tf.gfile.GFile(FROZEN_GRAPH_FILE, 'rb') as f:
		serialRead = f.read()
		serialGraph.ParseFromString(serialRead)
		tf.import_graph_def(serialGraph, name = '')

sess = tf.Session(graph = graph)

# scores and num_detections is useless

for dirs in os.listdir(IM_PATH):
	if not dirs.startswith('.'):
		for im in os.listdir(os.path.join(IM_PATH, dirs)):
			if im.endswith('.jpeg'):

				image = read_image(os.path.join(IM_PATH, dirs, im))
				if image is None:
					print('image read as None')
				print('image name: ', im)

				# here we will bring in the tensors from the frozen graph we loaded,
				# which will take the input through feed_dict and output the bounding boxes

				imageTensor = graph.get_tensor_by_name('image_tensor:0')

				bboxs = graph.get_tensor_by_name('detection_boxes:0')


				classes = graph.get_tensor_by_name('detection_classes:0')

				(outBoxes, classes) = sess.run([bboxs, classes],feed_dict={imageTensor: np.expand_dims(image, axis=0)})


				# visualise
				cnt = 0
				imageHeight, imageWidth = image.shape[:2]
				boxes = np.squeeze(outBoxes)
				classes = np.squeeze(classes)
				boxes = np.stack((boxes[:,1] * imageWidth, boxes[:,0] * imageHeight,
								boxes[:,3] * imageWidth, boxes[:,2] * imageHeight),axis=1).astype(np.int)

				for i, bb in enumerate(boxes):
					#bbox = (x1, y1, x2, y2 )
					#print(bbox)
					'''
					save_bounding_boxes(image, bbox, 
						save_path = os.path.join(IM_PATH, dirs, 'detected' ),
						im_name = '_'.join([str(cnt), im]))
					cnt += 1
					'''
					print(classes[i])
					cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (255,255,0), thickness = 1)
				
				plt.figure(figsize = (10, 10))
				plt.imshow(image)
				plt.show()

				cv2.imwrite(os.path.join(IM_PATH, dirs, 'a_' + im), image)
				#cv2.waitKey()
