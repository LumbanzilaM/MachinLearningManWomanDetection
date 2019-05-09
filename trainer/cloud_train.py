import numpy as np
import model_helper
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.python.lib.io import file_io
import keras
import argparse
import IO_utils
from zipfile import ZipFile
from keras.callbacks import ModelCheckpoint
import random
import os

LOCAL_TRAIN_FILE = "train.zip"
LOCAL_LABEL_FILE = "labels.txt"
LOCAL_MODEL_FILE = "model.h5"


def train(args):

    def get_dataset():
        images = []
        labels = []
        local_zip = IO_utils.load_data(args.training_dir, LOCAL_TRAIN_FILE)
        print("zip path = ", local_zip)
        with file_io.FileIO(args.training_dir, 'r') as f:
            with ZipFile(f, 'r') as archive:
                file_list = archive.infolist()
                random.shuffle(file_list)
                print("image number", len(file_list))
                for entry in file_list:
                    with archive.open(entry) as file:
                        try:
                            open_img = Image.open(file)
                            images.append(np.array(open_img))
                            label = np.zeros(num_classes)
                            label = define_label_one_hot(file.name, label)
                            labels.append(label)
                        except Exception as error:
                            print(error)
        images = np.array(images)
        labels = np.array(labels)
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        return x_train, x_test, y_train, y_test

    def define_label_one_hot(img, label):
        for c in class_names:
            if c in os.path.basename(img):
                label[class_names.index(c)] = 1
        return label

    def define_label(img, label):
        for c in class_names:
            if c in os.path.basename(img):
                label = class_names.index(c)
        return label

    with file_io.FileIO(args.label_file, 'r') as f:
        class_names = f.read().split(",")
    num_classes = len(class_names)
    input_shape = (args.img_size, args.img_size, args.channel)
    print("labels", class_names)
    print("num_class", num_classes)
    X_train, X_test, y_train, y_test = get_dataset()
    print("Xtrain", X_train.shape)
    print("Ytrain", y_train.shape)
    model = model_helper.init_cnn_model(input_shape, num_classes)
    model.summary()
    # checkpoint
    checkpoint = ModelCheckpoint(LOCAL_MODEL_FILE, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_test, y_test),
              callbacks=[checkpoint, model_helper.get_tensorboard_config("logs")], verbose=1)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Predict :", model.predict(X_test[:4]))
    model.save("last-" + LOCAL_MODEL_FILE)
    # predict first 4 images in the test set
    print("Predict :", model.predict(X_test[:4]))
    logs = file_io.list_directory("logs")
    print("logs = ", logs)
    IO_utils.save_file_in_cloud("last-" + LOCAL_MODEL_FILE, args.job_dir + "/" + 'last-' + args.job_name)
    IO_utils.save_file_in_cloud(LOCAL_MODEL_FILE, args.job_dir + "/" + args.job_name)
    for entry in logs:
        IO_utils.save_file_in_cloud("logs/" + entry, args.job_dir + "/logs/" + entry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--training-dir',
        help='GSC file or local path to the training dataset directory',
        required=True)
    parser.add_argument(
        '--job-dir',
        help='GSC file or local path to save the model',
        required=True)
    parser.add_argument(
        '--label-file',
        help='GSC file or local path to label file',
        required=True)
    parser.add_argument(
        '--job-name',
        help='model name',
        default='model.h5')
    parser.add_argument(
        '--img-size',
        help='train file size',
        default=128,
        type=int)
    parser.add_argument(
        '--channel',
        help='number of channel in images',
        default=3,
        type=int)
    parser.add_argument(
        '--epochs',
        help='number of epochs',
        default=100,
        type=int)
    parser.add_argument(
        '--batch-size',
        help='training batch size',
        default=200,
        type=int)
    args = parser.parse_args()
    train(args)

