import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# import training text file
trainFile = open('train-io.txt')
txt = trainFile.read()

array = txt.split('\n')

# parsing current array into x and y values
X = []
Y = []

for line in txt.split('\n'):
    tempData = line.split(' ')
    if len(tempData) == 11:
        temp = [float(i) for i in tempData]
        X.append(temp[:10])
        Y.append(temp[10])
    
X = np.asarray(X)
Y = np.asarray(Y)

# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(10,), activation='relu'))
model.add(Dense(700, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, Y, epochs=250, batch_size=1000)

# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))

# import test file
testFile = open('test-i.txt')
testTxt = testFile.read()

testArray = testTxt.split('\n')

# parsing current array into x and y values
rows = len(testArray)
test_x = []

for i in range(rows):
    tempInput = testArray[i].split(' ')
    if len(tempInput) == 10:
        test_x.append([float(i) for i in tempInput])

test_x = np.asarray(test_x)

#print(test_x)

pred_y = model.predict(test_x)

y_pred_labels = [round(i[0]) for i in pred_y]

print("Prediction Results Imported as 'test-result-i.txt' in Directory")
print("******************************")

with open ('test-result-i.txt', 'w') as f:
    for pred in y_pred_labels:
        f.write(str(pred))
        f.write('\n')