import os
from shutil import copyfile

# You only need to change this line to your dataset download path
download_path = '/root/dataset/Market-1501'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/threshold'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#-----------------------------------------
#gallery
gallery_path = download_path + '/bounding_box_test'
gallery_save_path = download_path + '/threshold/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
#negative
train_path = download_path + '/bounding_box_train'
negative_save_path = download_path + '/threshold/negative'
if not os.path.isdir(negative_save_path):
    os.mkdir(negative_save_path)
for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = negative_save_path + '/' + ID[0]  #first image is used as negative sample
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
#positive
query_path = download_path + '/query'
positive_save_path = download_path + '/threshold/positive'
if not os.path.isdir(positive_save_path):
    os.mkdir(positive_save_path)
for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = query_path + '/' + name
        dst_path = positive_save_path + '/' + ID[0]  #first image is used as positive sample
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)