# TFRecord
>***How to build and load a tfrecord file***
## 简介
为了高效地读取数据，有效的一种做法是对数据进行序列化并将其存储在一组可线性读取的文件中，这尤其适用于通过网络进行流式传输的数据。TFRecord 格式是一种用于存储二进制记录序列的简单格式。
原始的数据集往往是一个大压缩包，里面按类别有许多文件夹，文件夹里存放着该类别数据的jpg格式照片，为了提高数据集的读取效率，实际上也是为了提高训练网络模型时的速度，首先要做的就是将此压缩包转为一个TFRecord的二进制文件，对于训练集和验证集皆是如此，之后所有的读取数据集操作都将通过TFRecord文件来进行。
在这个地方我们需要自己写两个函数，分别用来构建TFRecord文件和加载TFRecord文件。

## 构建TFRecord文件
 ```
 with tf.io.TFRecordWriter(tfrecord_train) as writer: 
    for filename, label in zip(train_filenames, train_labels):
        image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
        feature = {                             # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
        writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件
```
其中tfrecord_train是用来保存TFRecord文件的绝对路径。filename是照片的绝对路径构成的列表，lable是照片对于的类别列表，两个列表位置相互对应，这里的feature需要根据待保存的数据类型构建，这里因为要保存图片数据以及图片的类别，因此需要建立两项，然后数据类型要对应。之后建立实例example，最后执行序列化并写入文件，完成TFRecord文件的构建。

## 加载TFRecord文件
### 1. 定义Feature结构
  这里定义的Feature结构一定要与构建时的结构一样：
```
Feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/label': tf.io.FixedLenFeature([], tf.int64),
}
```
### 2. 创建tf.train.Example 解码函数
```
def _parse_example(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_dict = tf.io.parse_single_example(example_string,Feature_description)
    feature_dict['image/encoded'] = tf.io.decode_jpeg(feature_dict['image/encoded'], channels=3)  # 解码JPEG图片
    img = feature_dict['image/encoded']
    img = tf.image.resize(img, (96, 96))
    img = img/255
    return img, tf.stack([tf.cast(feature_dict['image/label'], 'int32')])
```
### 3. 加载TFRecord文件
```
raw_dataset = tf.data.TFRecordDataset(train_file)  # 读取 TFRecord 文件
raw_dataset = raw_dataset.repeat()
train_datasets = raw_dataset.map(_parse_example)
train_datasets = train_datasets.shuffle(buffer_size=1024)
train_datasets = train_datasets.batch(batch)
train_datasets = train_datasets.prefetch(buffer_size=-1)  #开启预加载
```
repeat()一定不能漏，不然后面训练会出错，只有一个epoch的数据，prefetch用以开启预加载。
