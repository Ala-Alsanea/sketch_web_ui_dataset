#!/usr/bin/env python
# coding: utf-8

# In[1]:


import detectron2
import torch
from collections import namedtuple, OrderedDict
from object_detection.utils import dataset_util, label_map_util
import tensorflow.compat.v1 as tf
import io
from sklearn.model_selection import train_test_split
import random
import json
import imghdr
import shutil
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from matplotlib import pyplot as plt
from datetime import datetime
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from PIL import Image
import time
import cv2
import csv
import urllib
from pandas.plotting import register_matplotlib_converters
from matplotlib import rc
import matplotlib.pyplot as plt
from pylab import rcParams
import pandas as pd
import numpy as np
import os
import sys
import optparse

parser = optparse.OptionParser()

parser.add_option('-p', '--path', help='Pass the path image', default='.')

(opts, args) = parser.parse_args()  # instantiate parser

path = opts.path
new_dataset_path = f'{path}/new_dataset/'

print(opts.path)


# In[2]:


workspace = 'py'
if workspace != 'py':
    path = '.'

print(f'{workspace} - {path}')


# In[3]:


os.system(f'tar -zxvf {path}/dataset_roboflow.tar.gz -C {path}')
os.system(f'tar -zxvf {path}/dataset_sketch_it.tar.gz -C {path}')

# !tar -zxvf dataset_roboflow.tar.gz && tar -zxvf dataset_sketch_it.tar.gz


# In[4]:


new_dataset_path = f'{path}/new_dataset/'


# In[5]:


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# COMMON LIBRARIES

# from google.colab.patches import cv2_imshow
# %matplotlib inline

# DATA SET PREPARATION AND LOADING

# VISUALIZATION

# CONFIGURATION

# EVALUATION

# TRAINING

# from keras_retinanet import models/
# from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
# from keras_retinanet.utils.visualization import draw_box, draw_caption
# from keras_retinanet.utils.colors import label_color

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

register_matplotlib_converters()


RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# In[6]:


THRES_SCORE = 0.4


def draw_detections(image, box, scores=0, label=''):
    #   for box, score, label in zip(boxes[0], scores[0], labels[0]):
    #     if score < THRES_SCORE:
    #         continue

    #     color = label_color(labels)

    #     box = box.astype(int)

    draw_box(image, box, color=(0, 255, 0))

    caption = "{} {:.3f}".format(label, 0)
    draw_caption(image, box, caption)


def show_detected_objects(image_row, img_folder=''):

    img_path = img_folder+image_row.filename
    true_box = [
        image_row.xmin, image_row.ymin, image_row.xmax, image_row.ymax]
    image = read_image_bgr(img_path)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image)

#     boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis = 0))
#     print('scale',scale)
#     print('scores',scores)
#     print('labels',labels)
#     print('image',image)

#     boxes /= scale
#     print('boxes',boxes[0])

    draw_box(draw, true_box, color=(255, 0, 0))
    draw_detections(image, box=true_box, label=labels_to_name)

#     caption = "{} {:.3f}".format(image_row.class_name,0)
#     draw_caption(image,true_box , image_row.class_name)

    plt.axis('off')
    plt.imshow(draw)
    plt.show()


# In[7]:


roboflow = pd.read_csv(f'{path}/dataset_roboflow/new_train/annotations.csv')
sketch_it = pd.read_csv(f'{path}/dataset_sketch_it/images/annotations.csv')


# In[8]:


roboflow


# In[9]:


sketch_it


# In[10]:


new_dataset = roboflow.append(sketch_it, ignore_index=True)
new_dataset


# In[11]:


new_dataset.rename(columns={'image_name': 'filename'}, inplace=True)
new_dataset.rename(columns={'x_min': 'xmin'}, inplace=True)
new_dataset.rename(columns={'y_min': 'ymin'}, inplace=True)
new_dataset.rename(columns={'x_max': 'xmax'}, inplace=True)
new_dataset.rename(columns={'y_max': 'ymax'}, inplace=True)
new_dataset.rename(columns={'class_name': 'label'}, inplace=True)


roboflow.rename(columns={'image_name': 'filename'}, inplace=True)
roboflow.rename(columns={'x_min': 'xmin'}, inplace=True)
roboflow.rename(columns={'y_min': 'ymin'}, inplace=True)
roboflow.rename(columns={'x_max': 'xmax'}, inplace=True)
roboflow.rename(columns={'y_max': 'ymax'}, inplace=True)
roboflow.rename(columns={'class_name': 'label'}, inplace=True)

sketch_it.rename(columns={'image_name': 'filename'}, inplace=True)
sketch_it.rename(columns={'x_min': 'xmin'}, inplace=True)
sketch_it.rename(columns={'y_min': 'ymin'}, inplace=True)
sketch_it.rename(columns={'x_max': 'xmax'}, inplace=True)
sketch_it.rename(columns={'y_max': 'ymax'}, inplace=True)
sketch_it.rename(columns={'class_name': 'label'}, inplace=True)

new_dataset


# In[12]:


def copy2dri(df, destination, fromDir=[]):

    print(destination)

    for i in df.filename:

        #     create dir
        os.makedirs(destination, exist_ok=True)

        for Dir in fromDir:
            #             print(f'{Dir+i} -')

            if os.path.exists(Dir+i):
                pli_img = Image.open(Dir+i)
                cv_img = cv2.imread(Dir+i)

#                 print(f'{Dir+i} - {pli_img.format}')
        #         cv2.imwrite(destination+'/'+i, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                shutil.copy2(Dir+i, destination)

    df.to_csv(destination+'/annotations.csv', index=False)

    return 'done'


# # filter by accpted classes

# In[13]:


labels_to_name = pd.DataFrame(new_dataset['label'].unique()).sort_values(by=0)
labels_to_name


# In[14]:


new_dataset.label.value_counts()


# In[15]:


accpted_classes = pd.read_csv(f'{path}/accpted_classes.csv')
accpted_classes


# In[16]:


accpted_classes_list = accpted_classes.values.reshape(1, -1)[0]

classes = []

for i, item in enumerate(accpted_classes_list):
    classes.append({'name': item, 'id': i+1})


with open(f'{path}/label_map.pbtxt', 'w') as f:
    for item in classes:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(item['name']))
        f.write('\tid:{}\n'.format(item['id']))
        f.write('}\n')


# In[17]:


new_dataset = new_dataset[new_dataset.label.isin(
    list(accpted_classes.values.reshape(1, -1)[0]))]
new_dataset


# In[19]:


copy2dri(destination=new_dataset_path, df=new_dataset, fromDir=[
         f'{path}/dataset_roboflow/new_train/', f'{path}/dataset_sketch_it/images/'])


# In[20]:


new_dataset = pd.read_csv(f'{path}/new_dataset/annotations.csv')
new_dataset


# In[21]:


new_dataset[new_dataset['filename'] ==
            '8ae99b35-2354-4f8b-ade0-9fd98f09f4fe.png']


# # show all object

# In[26]:


# coco


def csv2coco(data, save_json_path):

    #     data = new_dataset.copy()

    images = []
    categories = []
    annotations = []

    category = {}
    category["supercategory"] = 'none'
    category["id"] = 0
    category["name"] = 'None'
    categories.append(category)

    data['fileid'] = data['filename'].astype('category').cat.codes
    data['categoryid'] = pd.Categorical(data['label'], ordered=True).codes
    data['categoryid'] = data['categoryid']+1
    data['annid'] = data.index

    def image(row):
        image = {}
        image["height"] = row.height
        image["width"] = row.width
        image["id"] = row.fileid
        image["file_name"] = row.filename
        return image

    def category(row):
        category = {}
        category["supercategory"] = 'None'
        category["id"] = row.categoryid
        category["name"] = row.label
        return category

    def annotation(row):
        annotation = {}
        area = (row.xmax - row.xmin)*(row.ymax - row.ymin)
        annotation["segmentation"] = []
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = row.fileid

        annotation["bbox"] = [row.xmin, row.ymin,
                              row.xmax - row.xmin, row.ymax-row.ymin]

        annotation["category_id"] = row.categoryid
        annotation["id"] = row.annid
        return annotation

    for row in data.itertuples():
        annotations.append(annotation(row))

    imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
    for row in imagedf.itertuples():
        images.append(image(row))

    catdf = data.drop_duplicates(
        subset=['categoryid']).sort_values(by='categoryid')
    for row in catdf.itertuples():
        categories.append(category(row))

    data_coco = {}
    data_coco["images"] = images
    data_coco["categories"] = categories
    data_coco["annotations"] = annotations

    json.dump(data_coco, open(save_json_path, "w"), indent=4)


# In[23]:


csv2coco(new_dataset.copy(), f'{path}/new_dataset/annotations.coco.json')


# In[24]:


DATA_SET_NAME = f'{path}/new_dataset/'
ANNOTATIONS_FILE_NAME = "annotations.coco.json"


# In[25]:


# TRAIN SET
DATA_SET_NAME = f"{DATA_SET_NAME}"
DATA_SET_IMAGES_DIR_PATH = os.path.join(DATA_SET_NAME)
DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_NAME, ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=DATA_SET_NAME,
    metadata={},
    json_file=DATA_SET_ANN_FILE_PATH,
    image_root=DATA_SET_IMAGES_DIR_PATH
)


# In[27]:


[
    data_set
    for data_set
    in MetadataCatalog.list()
    if data_set.startswith(DATA_SET_NAME)
]


# In[31]:


metadata = MetadataCatalog.get(DATA_SET_NAME)
dataset_train = DatasetCatalog.get(DATA_SET_NAME)

print(len(dataset_train))

dataset_entry = dataset_train[random.choice(range(0, len(dataset_train)))]
image = cv2.imread(dataset_entry["file_name"])
# image = cv2.imread('new_dataset/8ae99b35-2354-4f8b-ade0-9fd98f09f4fe.png')

visualizer = Visualizer(
    image[:, :, ::-1],
    metadata=metadata,
    scale=0.8,
    instance_mode=ColorMode.IMAGE_BW
)

out = visualizer.draw_dataset_dict(dataset_entry)
# cv2_imshow(out.get_image()[:, :, ::-1])

plt.figure(figsize=(15, 10))
plt.imshow(out.get_image()[:, :, ::-1])
plt.axis('off')
plt.show()


# # spilt into train and test

# In[32]:


new_dataset.label.value_counts()


# In[33]:


if os.path.exists('train'):
    shutil.rmtree('train')

if os.path.exists('test'):
    shutil.rmtree('test')

train_set = pd.DataFrame([])
test_set = pd.DataFrame([])
lbls = new_dataset.label.unique().tolist()

for lbl in lbls:

    train, test = train_test_split(
        new_dataset[new_dataset['label'] == lbl], test_size=0.2, shuffle=False)
    train_set = train_set.append(train, ignore_index=True)
    test_set = test_set.append(test, ignore_index=True)


# In[34]:


train_set


# In[35]:


test_set


# In[36]:


copy2dri(destination=f'{path}/train/images',
         df=train_set, fromDir=[new_dataset_path])


# In[37]:


copy2dri(destination=f'{path}/test/images',
         df=test_set, fromDir=[new_dataset_path])


# In[38]:


test_set = pd.read_csv(f'{path}/test/images/annotations.csv')
train_set = pd.read_csv(f'{path}/train/images/annotations.csv')


# # create coco from csv

# In[39]:


csv2coco(train_set.copy(), f'{path}/train/images/annotations.coco.json')


# In[40]:


csv2coco(test_set.copy(), f'{path}/test/images/annotations.coco.json')


# # create tf record

# In[41]:


# import tensorflow as tf


class_map = label_map_util.load_labelmap(f'{path}/label_map.pbtxt')
class_map_dict = label_map_util.get_label_map_dict(class_map)

# print(class_map_dict)


# TO-DO replace this with label map
def class_text_to_int(row_label):
    return class_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):

    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)

    width, height = image.size

#     print(encoded_jpg)

    filename = group.filename.encode('utf8')
#     print(image_name)
    image_format = image.format.encode()
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes_num = []

    for index, row in group.object.iterrows():
        #         print(row['x_min'])
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['label'].encode('utf8'))
        classes_num.append(class_text_to_int(row['label']))

#     print(image_format)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes_num),
    }))

    return tf_example


def csv_2_tfrecord(output_path, image_dir, csv_input):

    writer = tf.python_io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
#     print(grouped)

    # added
    file_errors = 0

    for group in grouped:
        try:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        except:

            # added
            file_errors += 1
            pass

    writer.close()

    # added
    print("FINISHED. There were %d errors" % file_errors)

    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


# call
csv_2_tfrecord(csv_input=f'{path}/train/images/annotations.csv',
               image_dir=f'{path}/train/images',
               output_path=f'{path}/train.record')

csv_2_tfrecord(csv_input=f'{path}/test/images/annotations.csv',
               image_dir=f'{path}/test/images',
               output_path=f'{path}/test.record')


# # to yolo format

# In[42]:


labels = {
    "train": os.path.join(f'{path}/train', "labels"),
    "test": os.path.join(f'{path}/test', "labels"),
}
for label in labels.values():
    os.makedirs(label, exist_ok=True)


# In[43]:


def csv_2_yolo(df, out_path, cocoAnnPath):
    f = open(cocoAnnPath)
    data = json.load(f)
    f.close()
    print(data['categories'])

    def get_img_ann(image_id):
        img_ann = []
        isFound = False
        for ann in data['annotations']:
            if ann['image_id'] == image_id:
                img_ann.append(ann)
                isFound = True
        if isFound:
            return img_ann
        else:
            return None

    def get_img(filename):
        for img in data['images']:
            if img['file_name'] == filename:
                return img

    for filename in df['filename']:
        # Extracting image
        img = get_img(filename)
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']

        # Get Annotations for this image
        img_ann = get_img_ann(img_id)

        if img_ann:
            # Opening file for current image
            file_object = open(f"{out_path}/{filename[:-4]}.txt", "a")

        for ann in img_ann:
            current_category = ann['category_id'] - \
                1  # As yolo format labels start from 0
            current_bbox = ann['bbox']
            x = current_bbox[0]
            y = current_bbox[1]
            w = current_bbox[2]
            h = current_bbox[3]

            # Finding midpoints
            x_centre = (x + (x+w))/2
            y_centre = (y + (y+h))/2

            # Normalization
            x_centre = x_centre / img_w
            y_centre = y_centre / img_h
            w = w / img_w
            h = h / img_h

            # Limiting upto fix number of decimal places
            x_centre = format(x_centre, '.6f')
            y_centre = format(y_centre, '.6f')
            w = format(w, '.6f')
            h = format(h, '.6f')

            # Writing current object
            file_object.write(
                f"{current_category} {x_centre} {y_centre} {w} {h}\n")

        file_object.close()
    print("done")


# In[44]:


train = pd.read_csv(f'{path}/train/images/annotations.csv')
test = pd.read_csv(f'{path}/test/images/annotations.csv')

csv_2_yolo(
    df=train, cocoAnnPath=f'{path}/train/images/annotations.coco.json', out_path=labels['train'])
csv_2_yolo(
    df=test, cocoAnnPath=f'{path}/test/images/annotations.coco.json', out_path=labels['test'])


# In[ ]:


# In[ ]:


# In[ ]:
