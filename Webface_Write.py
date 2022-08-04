"""**************************
将数据集写入到TfRecord文件
**************************"""
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import tqdm

data_dir = '/home/fxf/Datasets/CASIA-WebFace/CASIA-WebFace/'
tfrecord_train = '/home/fxf/Webface_TfRecord/Webface_train.tfrecords'
tfrecord_ver= '/home/fxf/Webface_TfRecord/Webface_ver.tfrecords'

face_class=os.listdir(data_dir)

class_num = 0
webface_filenames = []
wenface_labels = []
train_filenames = []
train_labels = []
ver_filenames = []
ver_labels = []
min =100

"""*********************************
检查数据集每个类别中图片最少的类别数
*********************************"""
# for face_name in face_class:   
#     face_list=os.listdir(data_dir+face_name)
#     temp=len(face_list)
#     if temp<min:
#         min=temp
#         print(face_name)
# print(min)
# exit()

"""**************************************
分配训练集和验证集，将每个类别的前五张
抽出来组成验证集。但是经过上面代码测试，
发现本数据集中还有部分类别的照片数不足五张。
**************************************"""
print('正在从数据集划分训练集和验证集：')
for face_name in tqdm.tqdm(face_class):
    face_list=os.listdir(data_dir+face_name)
    ver_filenames = ver_filenames+[data_dir+face_name+'/'+filename for filename in face_list[:5]]
    train_filenames = train_filenames+[data_dir+face_name+'/'+filename for filename in face_list[5:]]
    train_labels = train_labels+[class_num]*(len(face_list)-5)
    ver_labels = ver_labels + [class_num]
    class_num = class_num+1
print('数据集划分完成！')


"""**************************
#一共10757个类别，494414张照片
**************************"""
print('验证集：{}张'.format(len(ver_filenames)))
print('训练集：{}张'.format(len(train_filenames)))
print('类别数：{}'.format(class_num))


"""**************************
#构建训练集的TFRecord文件
**************************"""
print('正在构建训练集的TFRecord文件：')
with tf.io.TFRecordWriter(tfrecord_train) as writer:
    for filename, label in tqdm.tqdm(zip(train_filenames, train_labels)):
        image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
        feature = {                             # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
        writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件
print('训练集TFRecord文件构建完成！')


"""**************************
#构建验证集的TFRecord文件
**************************"""
print('正在构建验证集的TFRecord文件：')
with tf.io.TFRecordWriter(tfrecord_ver) as writer:
    for filename, label in zip(ver_filenames, ver_labels):
        image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
        feature = {                             # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
        writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件
print('测试集TFRecord文件构建完成！')


