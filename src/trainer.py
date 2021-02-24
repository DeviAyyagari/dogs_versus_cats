from model import get_model
from data_parser import DataParser

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

ROOT_DIR = "/home/devi/Documents/scratchpad/dl_exp/dogs_versus_cats/"
INPUT_DIR = ROOT_DIR + "data/train/"
MODEL_NAME = 'model_3layers_aug_dropout_SGD_lr_0.001_100_epochs'
IMAGE_DIM = (200, 200)
NUM_OF_CHANNELS = 3
BATCH_SIZE = 100

def plot_metrics(history):
    # summarize history for accuracy
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(ROOT_DIR + 'models/' + MODEL_NAME + '_acc.png')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(ROOT_DIR + 'models/' + MODEL_NAME + '_loss.png')

def train():
    dp = DataParser(INPUT_DIR, IMAGE_DIM, NUM_OF_CHANNELS)
    model = get_model(IMAGE_DIM[0], IMAGE_DIM[1], NUM_OF_CHANNELS)

    train_datagen = ImageDataGenerator(rescale=1./255,
                                        width_shift_range=0.1, 
                                        height_shift_range=0.1, 
                                        horizontal_flip=True)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_df = dp.create_keras_dataframe(dp.train_filenames)
    valid_df = dp.create_keras_dataframe(dp.valid_filenames)

    train_generator = train_datagen.flow_from_dataframe(
            dataframe = train_df,
            directory = INPUT_DIR,
            x_col = 'file_name',
            y_col = 'labels',
            target_size=IMAGE_DIM,
            batch_size=BATCH_SIZE,
            class_mode='binary')
    
    validation_generator = valid_datagen.flow_from_dataframe(
            dataframe = valid_df,
            directory = INPUT_DIR,
            x_col = 'file_name',
            y_col = 'labels',
            target_size=IMAGE_DIM,
            batch_size=BATCH_SIZE,
            class_mode='binary')
    
    checkpoint = ModelCheckpoint(ROOT_DIR + 'models/' + MODEL_NAME + '.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')  

    history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=100,
            verbose=1,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[checkpoint])
    plot_metrics(history)
    
if __name__ == "__main__":
    train()