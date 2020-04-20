import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten


class BinaryNet:
    @staticmethod
    def build(width, height, depth):
        model = tf.keras.models.Sequential()
        inputShape = (height, width, depth)

        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=inputShape))
        model.add(AveragePooling2D())
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(AveragePooling2D())
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(AveragePooling2D())
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(units=256, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(units=128, activation='relu'))

        model.add(Dense(units=2, activation='softmax'))

        return model

if __name__ == '__main__':
    BinaryNet = BinaryNet
    Model = BinaryNet.build(64,64,3)
    Model.summary()

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 62, 62, 16)        448       
_________________________________________________________________
average_pooling2d (AveragePo (None, 31, 31, 16)        0         
_________________________________________________________________
dropout (Dropout)            (None, 31, 31, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 29, 32)        4640      
_________________________________________________________________
average_pooling2d_1 (Average (None, 14, 14, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 64)        18496     
_________________________________________________________________
average_pooling2d_2 (Average (None, 6, 6, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 6, 6, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 2304)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               590080    
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 258       
=================================================================
Total params: 646,818
Trainable params: 646,818
Non-trainable params: 0
_________________________________________________________________
"""
