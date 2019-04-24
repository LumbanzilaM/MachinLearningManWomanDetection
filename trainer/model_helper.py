import keras
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential

def init_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='SAME', input_shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='SAME', input_shape=input_shape),
        keras.layers.MaxPool2D((3, 3)),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


def init_dumb_cnn_model(input_shape, num_classes):
    # create model
    model = Sequential()
    # add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


def init_res_net_model(input_shape, num_class):
    keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
                                                             input_shape=input_shape, pooling=None, classes=num_class)



def get_tensorboard_config(path_to_log):
    return keras.callbacks.TensorBoard(
        log_dir=path_to_log,
        histogram_freq=0,
        write_graph=True,
        write_images=False)
