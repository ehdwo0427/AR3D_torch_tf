#from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import os
import sys
sys.path.append("../")
from config import config, Path

class VideoDataset():
    def __init__(self, dataset='ucf101', split='train', clip_len=config.frames_per_clips, preprocess=False):
        self.clip_dir, self.frames_dir = Path.db_dir(dataset)
        self.clip_len = clip_len
        self.crop_size = 112
        self.split = split
        folder = os.path.join(self.frames_dir, split)
        self.labels = []
        self.fnames = []
        self.resize_height = 128
        self.resize_width =171


        # TODO: replace bypass for kinetics400 below
        # if config.dataset != 'kinetics400':
        #     if not self.check_integrity():
        #         raise RuntimeError('Dataset not found or corrupted.' +
        #                         ' You need to download it from official website.')
        #
        #     if (not self.check_preprocess()) or preprocess:
        #         print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
        #         self.preprocess()
        # self.preprocess()
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                self.labels.append(label)
                
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(self.labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in self.labels], dtype=int)

        # if not os.path.exists(config.labels_path):
        #     with open(config.labels_path, 'w') as f:
        #         for id, label in enumerate(sorted(self.label2index)):
        #             f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)
        
    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        # TODO: Remove bypass for kinetics400 below
        if config.dataset == 'kinetics400':
            return False

        if not os.path.exists(self.clip_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in frames_dir

        if not os.path.exists(self.frames_dir):
            return False
        elif not os.path.exists(os.path.join(self.frames_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.frames_dir, 'train'))):
            for video in os.listdir(os.path.join(self.frames_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.frames_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.frames_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] == 128 or np.shape(image)[1] == 128:
                    return True
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.frames_dir):
            os.mkdir(self.frames_dir)
            os.mkdir(os.path.join(self.frames_dir, 'train'))
            os.mkdir(os.path.join(self.frames_dir, 'val'))
            os.mkdir(os.path.join(self.frames_dir, 'test'))
        
        # Split train/val/test sets
            for file in os.listdir(self.clip_dir):
                file_path = os.path.join(self.clip_dir, file)
                video_files = [name for name in os.listdir(file_path)]

                train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
                train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

                train_dir = os.path.join(self.frames_dir, 'train', file)
                val_dir = os.path.join(self.frames_dir, 'val', file)
                test_dir = os.path.join(self.frames_dir, 'test', file)

                if not os.path.exists(train_dir):
                    train_dir = train_dir.replace("\ ", "\\")
                    print("train " + train_dir)
                    os.mkdir(train_dir)
                if not os.path.exists(val_dir):
                    val_dir = val_dir.replace("\ ", "\\")
                    os.mkdir(val_dir)
                if not os.path.exists(test_dir):
                    test_dir = test_dir.replace("\ ", "\\ ")
                    os.mkdir(test_dir)

                # for video in video_files:
                #     self.process_video(video, file, val_dir)

                for video in train:
                    self.process_video(video, file, train_dir)

                for video in val:
                    self.process_video(video, file, val_dir)

                for video in test:
                    self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.clip_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            #added nomalize data to [0,1]
            frame = frame / 255.0
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        # train 1
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        ## added, modified
        # train 0
        # frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)],key=lambda x:int(x.split('/')[-1][:-4]))
        frame_count = len(frames)
        frame_height, frame_width = cv2.imread(frames[0]).shape[0:2]
        buffer = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering

        #time_index = np.random.randint(buffer.shape[0] - clip_len)
        #ldj
        time_index = buffer // clip_len
        frame_select = np.empty((clip_len, buffer[1], buffer[2]), np.dtype('float32'))
        for i in len(buffer):
            if i // time_index == 0:
                frame_select.append(buffer[i])

            if len(frame_select) == clip_len:
                break

        if frame_select < clip_len:
            while(True):
                frame_select.insert(int(len(frame_select) / 2), frame_select[int(len(frame_select) / 2)])
                if frame_select == clip_len:
                    break
        #ldj

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        # buffer = buffer[time_index:time_index + clip_len,
        #          height_index:height_index + crop_size,
        #          width_index:width_index + crop_size, :]
        #ldj
        buffer = frame_select
        buffer = buffer[:,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]
        #ldj
        # for i in range(len(buffer)):
        #     cv2.imwrite("test"+str(i)+".jpg",buffer[i])
        # raise RuntimeError('check!!')

        return buffer
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='ucf101', split='train', clip_len=config.frames_per_clips, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break