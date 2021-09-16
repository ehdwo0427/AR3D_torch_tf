import cv2
import tensorflow as tf
import numpy as np
import os
import sys
import math

from config_tf import config
from config_tf import Path


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, phase, shuffle=True):
        super(DataLoader, self).__init__()
        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.clip_len = 16
        self.crop_size = 112
        self.phase = phase
        self.num_classes = config.num_classes

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(self.y)))}  # LDJ

        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in self.y], dtype=int)
        # self.label_array = self.y       #LDJ

        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        clip_dir_list = [self.x[k] for k in indexes]
        label_list = [self.label_array[k] for k in indexes]
        X = self.__get_batch(clip_dir_list)
        return X, tf.keras.utils.to_categorical(label_list, num_classes=self.num_classes)
        # return X, label_list

    def __get_batch(self, clip_dir_list):
        batch = np.empty((self.batch_size, self.clip_len, self.crop_size, self.crop_size, 3), np.dtype('float32'))
        for i, clip_dir in enumerate(clip_dir_list):
            buffer = self.__get_frames(clip_dir)
            buffer = self.__crop(buffer)  # LDJ
            buffer = self.__normalize(buffer)
            if self.phase == 'train':
                self.__random_flip(buffer)
            batch[i] = buffer
        return batch

    def __get_frames(self, clip_dir):
        # print(clip_dir)
        frames = sorted([os.path.join(clip_dir, img) for img in os.listdir(clip_dir)],
                        key=lambda x: int(x.split('/')[-1][:-4]))
        frames = self.__timing_jitter(frames)
        frame_count = len(frames)
        frame_height, frame_width = cv2.imread(frames[0]).shape[0:2]
        buffer = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
        return buffer

    def __timing_jitter(self, frames):
        """randomly select time index for temporal jittering"""

        time_index = np.random.randint(len(frames) - self.clip_len)
        frames = frames[time_index:time_index + self.clip_len]
        return frames

    def __crop(self, buffer):
        """crop frames to input size"""

        height_index = np.random.randint(buffer.shape[1] - self.crop_size)
        width_index = np.random.randint(buffer.shape[2] - self.crop_size)

        buffer = buffer[:, height_index:height_index + self.crop_size,
                 width_index:width_index + self.crop_size, :]

        return buffer

    def __normalize(self, buffer):
        """mean shift and nomalize data to [0,1]"""
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            frame = frame / 255.0
            buffer[i] = frame

        return buffer

    def __random_flip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


def load_data(dataset, split):
    labels = []
    fnames = []
    _, frames_dir = Path.db_dir(dataset)
    train_folder = os.path.join(frames_dir, split)
    for label in sorted(os.listdir(train_folder)):
        for fname in os.listdir(os.path.join(train_folder, label)):
            fnames.append(os.path.join(train_folder, label, fname))
            labels.append(label)

    return fnames, labels


##LDJ##
def load_data_2(dataset, split):
    labels = []
    fnames = []
    _, frames_dir = Path.db_dir(dataset)
    train_folder = os.path.join(frames_dir, split)
    for fname in os.listdir(os.path.join(train_folder)):
        fnames.append(os.path.join(train_folder, fname))

    return fnames, labels


def preprocess(dataset, fold, split):
    assert split in ["test", "train", "val"], "split must be one of (test, train, val)"

    split_dir, frames_dir = Path.db_dir(dataset)
    split_dir = split_dir + "/split" + str(fold) + "/" + split

    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)
    if not os.path.exists(os.path.join(frames_dir, split)):
        os.mkdir(os.path.join(frames_dir, split))

    count = 0
    for vid_class in os.listdir(split_dir):
        print("processing " + vid_class + "...   " + str(count + 1) + "/" + str(len(os.listdir(split_dir))))

        count += 1
        class_path = os.path.join(split_dir, vid_class)
        vid_files = [name for name in os.listdir(class_path)]

        for video in vid_files:
            extract_frames(video, vid_class, split_dir, frames_dir, split)


def extract_frames(vid, cls, vid_dir, frames_dir, split):
    resize_height = 128
    resize_width = 171

    video_filename = vid.split('.')[0]
    if not os.path.exists(os.path.join(frames_dir, split)):
        os.mkdir(os.path.join(frames_dir, split))
    if not os.path.exists(os.path.join(frames_dir, split, cls)):
        os.mkdir(os.path.join(frames_dir, split, cls))
    if not os.path.exists(os.path.join(frames_dir, split, cls, video_filename)):
        os.mkdir(os.path.join(frames_dir, split, cls, video_filename))
    cap = cv2.VideoCapture(os.path.join(vid_dir, split, cls, vid))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0
    i = 0
    retaining = True

    EXTRACT_FREQUENCY = 1

    while (count < frame_count and retaining):
        retaining, frame = cap.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            if (frame_height != resize_height) or (frame_width != resize_width):
                frame = cv2.resize(frame, (resize_width, resize_height))
            if i < 10:
                cv2.imwrite(filename=os.path.join(frames_dir, split, cls, video_filename, '0000{}.jpg'.format(str(i))),
                            img=frame)
            else:
                cv2.imwrite(filename=os.path.join(frames_dir, split, cls, video_filename, '000{}.jpg'.format(str(i))),
                            img=frame)
            i += 1
        count += 1

    cap.release()


if __name__ == "__main__":
    x_train, y_train = load_data(config.dataset, "train")
    # x_valid, y_valid = load_data(config.dataset, "valid")

    train_loader = DataLoader(x_train, y_train, 25, 'train', config.batch_size)

    # for i, (clips, label) in enumerate(loader):
    #     if i == 10:
    #         break
    #     else:
    #         first_clip_of_each_batch = clips[0,:,:,:,:]*255
    #         cv2.imwrite("test_" + str(i+1) + "_" + label[0] + ".jpg",first_clip_of_each_batch+np.array([[[90.0, 98.0, 102.0]]]))

    for i, (clips, label) in enumerate(train_loader):
        if i == 10:
            break
        else:
            first_clip_of_each_batch = clips[0, :, :, :, :] * 255
            for j in range(first_clip_of_each_batch.shape[0]):
                cv2.imwrite(label[0] + "_test_" + str(j) + "_" + ".jpg",
                            (first_clip_of_each_batch[j] + np.array([[[90.0, 98.0, 102.0]]])))

# preprocess("hmdb51", 1, "test")
