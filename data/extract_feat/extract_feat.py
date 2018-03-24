from tqdm import *
import numpy as np
import os
import json
import h5py
import sys
caffe_python = '/home/liqing/Desktop/program/caffe/python'
sys.path.insert(0, caffe_python)

import caffe
caffe.set_mode_gpu()
caffe.set_device(1)

model_dir = "/home/liqing/Desktop/program/ResNet-152/"
model_def = model_dir + "ResNet-152-deploy.prototxt"
model_weights = model_dir + "ResNet-152-model.caffemodel"
net = caffe.Net(model_def, model_weights, caffe.TEST)

image_size = 448
mu = np.array([ 104, 117, 123]) # [B, G, R]
transformer = caffe.io.Transformer({'data': (1, 3, image_size, image_size)})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


root_dir = os.environ['data_dir']
dataset_dir = root_dir + 'Images/'
all_imgs = os.listdir(dataset_dir)
all_imgs.sort()
json.dump(all_imgs, open('all_imgs.json','w'))
print(len(all_imgs))

#all_imgs = all_imgs[:10]

feat_layers = [('res5c', [2048, 14, 14])]
fid_h5 = h5py.File('VizWiz.h5', 'w')
for feat_layer, feat_shape in feat_layers:
	dset_feat = fid_h5.create_dataset(feat_layer, [len(all_imgs)] + feat_shape, 'f4')

for idx, img_info in enumerate(tqdm(all_imgs)):
	img_path = os.path.join(dataset_dir, img_info)
	image = caffe.io.load_image(img_path)
	image = caffe.io.resize_image(image, (image_size, image_size))
	net.blobs['data'].data[0] = transformer.preprocess('data', image)
	net.forward()
	for feat_layer, _ in feat_layers:
		tmp = net.blobs[feat_layer].data[0]
		fid_h5[feat_layer][idx] = tmp
fid_h5.close()


