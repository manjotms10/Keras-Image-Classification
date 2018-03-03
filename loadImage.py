import matplotlib.pylab as plt
import glob
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset():
    X_data = []
    Y_data = []
    classes = {'airplane':1, 'no_airplane':0}
    files = glob.glob ("data/train/airplanes/*")
    for myFile in files:
        print(myFile)
        image = plt.imread (myFile)
        image = plt.resize(image, (32,32,3))
        X_data.append(image)
        Y_data.append(1)

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    Y_data = Y_data.reshape((Y_data.shape[0],1))

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2)

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

load_dataset()
