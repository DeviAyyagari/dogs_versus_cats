from model import get_model
from data_parser import DataParser
from keras.preprocessing.image import ImageDataGenerator


INPUT_DIR = "/home/devi/Documents/scratchpad/dl_exp/dogs_versus_cats/data/train/"
IMAGE_DIM = (128, 128)
NUM_OF_CHANNELS = 3
BATCH_SIZE = 1

def train():
    dp = DataParser(INPUT_DIR, IMAGE_DIM, NUM_OF_CHANNELS)
    model = get_model(IMAGE_DIM[0], IMAGE_DIM[1], NUM_OF_CHANNELS)

    train_datagen = ImageDataGenerator(rescale=1./255)
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
    
    model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=20,
            verbose=1,
            validation_data=validation_generator,
            validation_steps=len(validation_generator))
    
if __name__ == "__main__":
    train()