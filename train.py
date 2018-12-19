import misc
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
import keras.optimizers as kopt

data = misc.getData()

# Keras only admits numpy arrays as input
npdata = np.array([np.array(x) for x in data])

# Shuffling training and test data
np.random.shuffle(npdata)

# Training set is 80% of total data
xtrain = npdata[:800,:-1]
ytrain = npdata[:800, -1]

# Test set is the remaining 20%
xtest = npdata[800:,:-1]
ytest = npdata[800:, -1]

# One hot codification (2 = 0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
oh_ytrain = to_categorical(ytrain, num_classes=10)
oh_ytest = to_categorical(ytest, num_classes=10)

model = Sequential()

# 4 layers (different parameters give different results)
model.add(Dense(16, activation='relu', input_dim=4))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=kopt.SGD(lr=0.001), metrics=['accuracy'])

# Can modify epochs
model.fit(xtrain, oh_ytrain, epochs=100, batch_size=40)

result = model.evaluate(xtest, oh_ytest, batch_size=40)

print(result)

# model.save('my_model_3.h5')
