from keras.models import load_model
import numpy as np
import misc
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential

callback = [EarlyStopping("val_loss", patience=50)]

data = misc.getAlldirsRegressionData()

# model = load_model("all_dirs_models/all_models/model_relu_64_4_32_4_64.h5")

# Keras only admits numpy arrays as input
npdata = np.array([np.array(x) for x in data])

# Shuffling training and test data
np.random.shuffle(npdata)

# Training set is 80% of total data
xtrain = npdata[:3200, :-2]
ytrain = npdata[:3200, -2:]

# Test set is the remaining 20%
xtest = npdata[3200:, :-2]
ytest = npdata[3200:, -2:]

model = Sequential()
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(2, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

model.fit(xtrain, ytrain, epochs=2000, batch_size=40, verbose=1, validation_data=(xtest, ytest), callbacks=callback)

model.save("all_dirs_models/best_model_extended.h5")
