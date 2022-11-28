import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

f = open('train-io.txt')
txt = f.read()

array = txt.split('\n')

#parsing current array into x and y values
rows = len(array)
X = []
Y = []

for i in range(rows):
    tempData = array[i].split(' ')
    for i in range(len(tempData)):
        tempData[i] = float(tempData[i])
    tempTarget = tempData.pop()
    X.append(tempData)
    Y.append(tempTarget)

# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(10,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, Y, epochs=200, batch_size=100)

# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))
