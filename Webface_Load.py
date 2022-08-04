"""**************************
从TfRecord文件读取数据
**************************"""
import tensorflow as tf
import matplotlib.pyplot as plt
from modules.models import resnet
from modules import metrics
class_num = 10575
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'#设置所有可以使用的显卡，共计四块

train_file ='/home/fxf/resnet18_last_version/resnet18/data/webface_train.tfrecord'
ver_file = '/home/fxf/resnet18_last_version/resnet18/data/webface_val.tfrecord'


Feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/label': tf.io.FixedLenFeature([], tf.int64),
}


def _parse_example(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_dict = tf.io.parse_single_example(example_string,Feature_description)
    feature_dict['image/encoded'] = tf.io.decode_jpeg(feature_dict['image/encoded'], channels=3)  # 解码JPEG图片
    img = feature_dict['image/encoded']
    img = tf.image.resize(img, (96, 96))
    img = img/255
    return img, tf.stack([tf.cast(feature_dict['image/label'], 'int32')])



"""**************************
测试生成的 tf.data.Dataset 对象
**************************"""
# for image, label in dataset_ver:
#     plt.figure()
#     #plt.title(str(label))
#     plt.imshow(image.numpy())
#     plt.show()
classes_num = 10575
batch=64
epochs=20

"""**************************
加载TFRecord文件生成数据集
**************************"""
raw_dataset = tf.data.TFRecordDataset(train_file)  # 读取 TFRecord 文件
raw_dataset = raw_dataset.repeat()
train_datasets = raw_dataset.map(_parse_example)
train_datasets = train_datasets.shuffle(buffer_size=1024)
train_datasets = train_datasets.batch(batch)
train_datasets = train_datasets.prefetch(buffer_size=-1)  #开启预加载

raw_dataset2 = tf.data.TFRecordDataset(ver_file)  # 读取 TFRecord 文件
raw_dataset2 = raw_dataset2.repeat()
val_datasets = raw_dataset2.map(_parse_example)
val_datasets = val_datasets.batch(batch)


"""******************
创建网络模型设置优化器
******************"""
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.losses.SparseCategoricalCrossentropy()

model = resnet(10575)

model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=[metrics.MySparseAccuracy(batch)])


"""************************
    设置回调保存模型
************************"""
checkpoint_dir = "./ckpt1"
if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)

ckpt_path = tf.train.latest_checkpoint('./ckpt1')
if ckpt_path is not None:
            print("[*] load ckpt from {}".format(ckpt_path))
            model.load_weights(ckpt_path)
else:
            print("[*] training from scratch.")

callbacks = [tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + '/ckpt-loss{loss:.2f}' + 'resnet_accuracy{myaccuracy:.3f}',
            save_freq='epoch',
            save_weights_only=True,
            monitor='myaccuracy',
            verbose=1
        )]

steps_per_epoch = 408072 //batch
val_steps = 45341 // batch

"""***********************
        开始训练
***********************"""
print('Start training...')
model.fit(train_datasets, epochs=epochs,callbacks=callbacks, steps_per_epoch=steps_per_epoch,
          validation_data=val_datasets, validation_steps=val_steps)
print("training done!")