import pandas as pd
from enum import Enum
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model

import matplotlib.pyplot as plt
from keras.callbacks import Callback
import itertools
from sklearn.metrics import confusion_matrix

# Callback class to visialize training progress
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# load data
raw_data = pd.read_csv('data_under_scene.csv', header=0)
dataset = raw_data.values
# X = dataset[:, 0:36].astype(float)
# Y = dataset[:, 36]
X = dataset[0:3161, 0:36].astype(float)
Y = dataset[0:3161, 36]

# encoder = LabelEncoder()
# encoder_Y = encoder.fit_transform(Y)
# print(encoder_Y[0], encoder_Y[900], encoder_Y[1800], encoder_Y[2700])
encoder_Y = [0]*743 + [1]*722 + [2]*688 + [3]*1008
#encoder_Y = [0] * 45446 + [1] * 48402 + [2] * 49310 + [3] * 78088 + [4] * 56730 + [5] * 56559 + [6] * 76461
#encoder_Y = [0] * 45445 + [1] * 48402 + [2] * 49310 + [3] * 78088 + [4] * 56730 + [5] * 56559 + [6] * 76461
# one hot
dummy_Y = np_utils.to_categorical(encoder_Y)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=9)

# build keras model
model = Sequential()
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=4, activation='softmax'))  # units = nums of classes

# training
his = LossHistory()
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[his])
model.summary()
his.loss_plot('epoch')
model.save('action_new.h5')

# evaluate and draw confusion matrix
print('Test:')
score, accuracy = model.evaluate(X_test,Y_test,batch_size=32)
print('Test Score:{:.3}'.format(score))
print('Test accuracy:{:.3}'.format(accuracy))


