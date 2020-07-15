import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

INPUT_DIR = "/home/devi/Documents/scratchpad/dl_exp/dogs_versus_cats/data/train/"
IMAGE_DIM = (128, 128)
NUM_OF_CHANNELS = 3

class DataParser:
    def __init__(self, input_dir, image_dim, num_channels):
        self.image_dim = image_dim
        self.num_channels = num_channels
        filename_list = [os.path.join(input_dir, each) for each in os.listdir(input_dir)]
        self.train_filenames, self.valid_filenames = train_test_split(filename_list, test_size = 0.3)
        print("Number of train data samples: {}".format(len(self.train_filenames)))
        print("Number of valid data samples: {}".format(len(self.valid_filenames)))

    def parse_labels(self, filename_list):
        return [each_name.split("/")[-1].split(".")[0] for each_name in filename_list]
   
    def read_single_image(self, file_path):
        return cv2.imread(file_path)
    
    def preprocess_single_image(self, img):
        resized_img = cv2.resize(img, self.image_dim)
        return resized_img/255.0

    def read_image_data(self, file_list):
        img_data = np.empty((len(file_list), self.image_dim[0], self.image_dim[1], self.num_channels))
        for index, each_file_name in enumerate(file_list):
            img = self.read_single_image(each_file_name)
            img = self.preprocess_single_image(img)
            img_data[index] = img
        return img_data

    def train_data_loader(self, batch_size, current_batch):
        from_batch = current_batch*batch_size
        to_batch = current_batch*batch_size + batch_size
        if to_batch > len(self.train_filenames):
            to_batch = len(self.train_filenames)
        current_batch_file_names = self.train_filenames[from_batch:to_batch]

        train_data = self.read_image_data(current_batch_file_names)
        labels = self.parse_labels(current_batch_file_names)
        return train_data, labels

    def valid_data_loader(self, batch_size, current_batch):
        from_batch = current_batch*batch_size
        to_batch = current_batch*batch_size + batch_size
        if to_batch > len(self.valid_filenames):
            to_batch = len(self.valid_filenames)
        current_batch_file_names = self.valid_filenames[from_batch:to_batch]

        valid_data = self.read_image_data(current_batch_file_names)
        labels = self.parse_labels(current_batch_file_names)
        return valid_data, labels
        
if __name__ == "__main__":
    dp = DataParser(INPUT_DIR, IMAGE_DIM, NUM_OF_CHANNELS)
    batch_size = 200
    num_of_batches = (len(dp.valid_filenames) // batch_size) + 1
    for batch_number in range(num_of_batches):
        dp.valid_data_loader(batch_size, batch_number)
