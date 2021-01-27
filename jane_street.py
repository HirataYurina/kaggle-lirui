# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:jane_street.py
# software: PyCharm


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def create_resnet(input1, input2, num_labels):
    """the feature is splited into input1 and inputs

    Args:
        input1:        (none, 71)
        input2:        (none, 59)
        num_labels:     5[int]
        label_smooth:   use label smooth to regularize model

    Returns:
        model

    """
    x1 = layers.BatchNormalization()(input1)
    x1 = layers.Dropout(rate=0.4)(x1)
    x1 = layers.Dense(300)(x1)
    x1 = Mish()(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(rate=0.4)(x1)
    x1 = layers.Dense(300)(x1)
    x1 = Mish()(x1)

    x1_x2 = layers.Concatenate()([x1, input2])

    x3 = layers.BatchNormalization()(x1_x2)
    x3 = layers.Dropout(rate=0.4)(x3)
    x3 = layers.Dense(300)(x3)
    x3 = Mish()(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Dropout(rate=0.4)(x3)
    x3 = layers.Dense(300)(x3)
    x3 = Mish()(x3)

    x1_x3_average = layers.Average()([x1, x3])
    x4 = layers.BatchNormalization()(x1_x3_average)
    x4 = layers.Dense(300)(x4)
    x4 = Mish()(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Dropout(0.4)(x4)
    x4 = layers.Dense(300)(x4)
    x4 = Mish()(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Dropout(0.4)(x4)
    x4 = layers.Dense(num_labels, activation='sigmoid')(x4)

    return keras.Model([input1, input2], x4)


class Mish(layers.Layer):
    """mish activation is state of the art activation function"""

    def __init__(self):
        super(Mish, self).__init__()

    def call(self, inputs, **kwargs):
        return inputs * tf.tanh(tf.nn.softplus(inputs))


if __name__ == '__main__':
    input1_ = keras.Input(shape=(71,))
    input2_ = keras.Input(shape=(59,))
    model = create_resnet(input1_, input2_, num_labels=5)
    model.summary()

    # read train data

    # feature engineering
    # 1.discard some old data
    # 2.fill miss data by average of dataset

    # start training
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(label_smoothing=0.01))
    model.fit(train_x, train_y, batch_size=200000, epochs=200)
