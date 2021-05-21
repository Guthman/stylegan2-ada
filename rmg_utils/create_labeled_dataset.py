import os
import numpy as np
import PIL.Image
import tensorflow as tf


def error(msg):
    print('Error: ' + msg)
    exit(1)


class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, resolution_log2=7, print_progress=True, progress_interval=10,
                 tfr_prefix=None):
        self.tfrecord_dir = tfrecord_dir
        if tfr_prefix is None:
            self.tfr_prefix = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        else:
            self.tfr_prefix = os.path.join(self.tfrecord_dir, tfr_prefix)
        self.expected_images = expected_images
        self.cur_images = 0
        self.shape = None
        # self.resolution_log2    = None
        self.resolution_log2 = resolution_log2
        self.tfr_writers = []
        self.print_progress = print_progress
        self.progress_interval = progress_interval

        if self.print_progress:
            name = '' if tfr_prefix is None else f' ({tfr_prefix})'
            print(f'Creating dataset "{tfrecord_dir}"{name}')
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self):  # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2 ** self.resolution_log2
            tfr_opt = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def create_tfr_writer(self, shape):
        self.shape = [shape[2], shape[0], shape[1]]
        assert self.shape[0] in [1, 3]
        assert self.shape[1] % (2 ** self.resolution_log2) == 0
        assert self.shape[2] % (2 ** self.resolution_log2) == 0
        tfr_opt = tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.NONE
        )
        tfr_file = self.tfr_prefix + "-r%02d.tfrecords" % (
            self.resolution_log2
        )
        self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    def add_image_raw(self, encoded_jpg):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print(
                "%d / %d\r" % (self.cur_images, self.expected_images),
                end="",
                flush=True,
            )
        for lod, tfr_writer in enumerate(self.tfr_writers):
            ex = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "shape": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=self.shape)
                        ),
                        "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg]))
                    }
                )
            )
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_from_image_folders(tfrecord_dir, image_dir, shuffle, ignore_labels):
    images = []
    labels = []
    print('Loading images from "%s"' % image_dir)

    label_count = 0
    for root, subdirs, files in os.walk(image_dir):
        for subdir in subdirs:
            folder_path = os.path.join(root, subdir)
            print('\t Loading images from "%s" as label %d' % (folder_path, label_count))
            # print('\t contains %d files' % len(os.listdir(folder_path)))

            if len(os.listdir(folder_path)):
                for file in os.listdir(folder_path):
                    images.append(os.path.join(folder_path, file))
                    labels.append(label_count)
                    # print(os.path.join(folder_path,file))

            label_count += 1

    assert ignore_labels in [0, 1]
    assert np.min(labels) == 0 and np.max(labels) == (label_count - 1)
    labels = np.array(labels)
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    img = np.asarray(PIL.Image.open(images[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir, len(images)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(images))
        for idx in range(order.size):
            # print('processing image: %s' % images[order[idx]])
            img = np.asarray(PIL.Image.open(images[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :]  # HW => CHW
            else:
                img = img.transpose([2, 0, 1])  # HWC => CHW
            tfr.add_image(img)

        if not ignore_labels:
            tfr.add_labels(onehot[order])


def create_from_image_folders_raw(tfrecord_dir, image_dir, shuffle, ignore_labels, resolution_log2=7):
    images = []
    labels = []
    print('Loading images from "%s"' % image_dir)

    label_count = 0
    for root, subdirs, files in os.walk(image_dir):
        for subdir in subdirs:
            folder_path = os.path.join(root, subdir)
            print('\t Loading images from "%s" as label %d' % (folder_path, label_count))
            # print('\t contains %d files' % len(os.listdir(folder_path)))

            if len(os.listdir(folder_path)):
                for file in os.listdir(folder_path):
                    images.append(os.path.join(folder_path, file))
                    labels.append(label_count)
                    # print(os.path.join(folder_path,file))

            label_count += 1

    assert ignore_labels in [0, 1]
    assert np.min(labels) == 0 and np.max(labels) == (label_count - 1)

    # Create label array
    labels = np.array(labels)
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    img = np.asarray(PIL.Image.open(images[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir, len(images), resolution_log2=resolution_log2) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(images))
        tfr.create_tfr_writer(img.shape)
        for idx in range(order.size):
            with tf.gfile.FastGFile(images[order[idx]], 'rb') as fid:
                try:
                    tfr.add_image_raw(fid.read())
                except Exception:
                    print('error when adding', images[order[idx]])
                    continue

        if not ignore_labels:
            tfr.add_labels(onehot[order])


tfrecord_output_dir = r'F:\temp\thesisdata\saatchi_micro_resized512_tfrecords2'
image_input_dir = r'F:\temp\thesisdata\saatchi_micro_resized512'
shuffle = True
ignore_labels = False

create_from_image_folders_raw(tfrecord_output_dir, image_input_dir, shuffle, ignore_labels, resolution_log2=8)
