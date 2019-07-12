import misc
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import numpy as np
import os


data = misc.getNewRegressionData("7")

# Keras only admits numpy arrays as input
npdata = np.array([np.array(x) for x in data])

# Shuffling training and test data
np.random.shuffle(npdata)

# Training set is 80% of total data
xtrain = npdata[:336, :-2]
ytrain = npdata[:336, -2:]

# Test set is the remaining 20%
xtest = npdata[336:, :-2]
ytest = npdata[336:, -2:]

activation_function = 'relu'

# 4 layers (different parameters give different results)
approved_models = [f for f in os.listdir("./regression_models/tested/approved")]
approved_min_models = [f for f in os.listdir("./regression_models/tested/min_approved")]
for model in approved_min_models:
    if model not in approved_models:
        approved_models.append(model)

models = []
for model in approved_models:
    models.append(misc.getLayersFromModelName(model))

for layers in models:

    model = Sequential()
    if layers[0] != 0: model.add(Dense(layers[0], activation=activation_function, input_dim=4))
    if layers[1] != 0: model.add(Dense(layers[1], activation=activation_function))
    if layers[2] != 0: model.add(Dense(layers[2], activation=activation_function))
    if layers[3] != 0: model.add(Dense(layers[3], activation=activation_function))
    if layers[4] != 0: model.add(Dense(layers[4], activation=activation_function))
    model.add(Dense(2, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    # Can modify epochs
    model.fit(xtrain, ytrain, epochs=100, batch_size=20, verbose=0)

    result = model.evaluate(xtest, ytest, batch_size=20)

    if result[0] < 1.5:
        save_name = 'new_regression_models/pot7/fixed/new_model_relu_{}_{}_{}_{}_{}'.format(layers[0], layers[1],
                                                                                            layers[2], layers[3],
                                                                                            layers[4])
        model.save(save_name)
        print("Model", save_name, "saved.")

    K.clear_session()

