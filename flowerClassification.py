import glob
import numpy as np
from sklearn.model_selection import train_test_split

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pylab as plt
from matplotlib.pyplot import imshow

def load_dataset():
    X_data = []
    Y_data = []
    classes = {'daisy':0, 'dandelion':1, 'rose':2, 'sunflower':3, 'tulip':4}
    for flower in classes:
        files = glob.glob ("flowers/" + flower + "/*")
        for myFile in files:
            #print(myFile)
            image = plt.imread (myFile)
            image = plt.resize(image, (120,80,3))
            X_data.append(image)
            Y_data.append(classes[flower])

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    Y_data = Y_data.reshape((Y_data.shape[0],1))

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2)

    Y_train = to_categorical(Y_train, 5)
    Y_train = np.array(Y_train)
    Y_test = to_categorical(Y_test, 5)
    Y_test = np.array(Y_test)

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

    return X_train, X_test, Y_train, Y_test

def createModel(input_shape):
    #input layer
    X_input = Input(input_shape)

    #layers
    X = Conv2D(32, (3,3), strides = (1,1), name='conv_1', padding='same')(X_input)
    X = Activation('relu')(X)
    X = Conv2D(32, (3,3), strides = (1,1), name='conv_12', padding='same')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name='max_pool_1')(X)
    X = Dropout(0.25)(X)

    X = Conv2D(64, (3,3), strides = (1,1), name='conv_2', padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D(64, (3,3), strides = (1,1), name='conv_22', padding='same')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name='max_pool_2')(X)
    X = Dropout(0.25)(X)

    #Flatten and FC layer
    X = Flatten()(X)
    X = Dense(1920, activation = 'relu')(X)
    X = Dense(5, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='CNN')
    return model
    
def main():
    X_train, X_test, Y_train, Y_test = load_dataset()
    
    model = createModel(X_train.shape[1:])
    model.compile( optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])
    model.fit( x = X_train, y = Y_train, epochs = 30, batch_size = 80, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=0)])
    predictions = model.evaluate( x = X_test, y = Y_test)
    print()
    print("Loss: ", predictions[0])
    print("Accuracy: ", predictions[1])
    model.summary()
    imshow(X_test[1])

if __name__=='__main__':
    main()
