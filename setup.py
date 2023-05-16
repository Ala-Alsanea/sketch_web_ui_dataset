#!/usr/bin/env python
# coding: utf-8

# # setup env

# In[2]:


import os

os.system(
    "pip3 install --user 'git+https://github.com/facebookresearch/detectron2.git'")
os.system("pip3 install -r sketch_web_ui_dataset/requirements.txt")
os.system(
    "git clone https://github.com/tensorflow/models sketch_web_ui_dataset/models/.")
os.system("")
os.system("")
os.system("")
os.system("")
os.system("")

# linux
if os.name == 'posix':
    os.system("apt install protobuf-compiler")
    os.system("cd sketch_web_ui_dataset/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python3 -m pip install .")
    os.system("")
    os.system("")
    os.system("")
    os.system("")

# win
if os.name == 'nt':
    url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    os.system("pip3 install wget")
    import wget
    wget.download(url)
    os.makedirs('protoc', exist_ok=True)
#     os.system("move protoc-3.15.6-win64.zip ")
    os.system("tar -xvf protoc-3.15.6-win64.zip -C protoc")
    os.system("")
    os.system("")
    os.environ['PATH'] += os.pathsep + \
        os.path.abspath(os.path.join('protoc', 'bin'))
    os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install")
    os.system("cd Tensorflow/models/research/slim && pip install -e . ")
    os.system("")
    os.system("")


# val
VERIFICATION_SCRIPT = os.path.join("sketch_web_ui_dataset",
                                   'models', 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
# Verify Installation
os.system(f"python3 {VERIFICATION_SCRIPT}")


# In[ ]:
