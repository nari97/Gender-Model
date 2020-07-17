import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Softmax, Flatten, LSTM
from keras.layers.embeddings import Embedding
from sklearn.metrics import accuracy_score
from keras.models import load_model

def return_counts(row):
    vector = [0]*26
    row = row.replace(',',' ')
    for ch in row.strip():
        try:
            vector[ord(ch)-97]+=1
        except:
            print (ch, row)
            exit()
    return vector

def transform_data(data):
    t_data = np.zeros((len(data), 26))
    for i in range(len(data)):
        t_data[i] = return_counts(data[i])
    return t_data

def buildModel():
    data = pd.read_csv(r"C:\Users\nari9\OneDrive\Documents\Projects\GenderModel\Data\CleanedData.csv")
    X = transform_data(data['name'].to_list())
    
    Y = data['gender'].to_numpy()
    
    Y = to_categorical(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size = 0.6, random_state = 1, shuffle = True)

    print (X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    model = Sequential()
    
    model.add(Dense(100, input_dim = 26))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation = 'softmax'))
    
    '''
    model.add(Embedding(X.shape[0],4, input_length = 26))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    '''
    print (model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=10, batch_size=64)
    o_values = model.predict(X_test)
    print(accuracy_score(np.argmax(o_values, axis = 1), np.argmax(Y_test, axis = 1)))
    model.save('model.hdf5')


def testModel():

    model = load_model('model.hdf5')
    data = pd.read_csv(r"C:\Users\nari9\OneDrive\Documents\Projects\GenderModel\Data\somedata.csv")
    X = transform_data(data['name'].to_list())
    
    Y = data['gender'].to_numpy()
    val = model.predict(data)
    print(accuracy_score(np.argmax(val, axis = 1), np.argmax(Y, axis = 1)))




if __name__ == "__main__":
    #buildModel()
    testModel()