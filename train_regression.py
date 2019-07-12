import misc
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


data = misc.getRegressionData()

# Keras only admits numpy arrays as input
npdata = np.array([np.array(x) for x in data])

# Shuffling training and test data
np.random.shuffle(npdata)

# Training set is 80% of total data
xtrain = npdata[:800, :-2]
ytrain = npdata[:800, -2:]

# Test set is the remaining 20%
xtest = npdata[800:, :-2]
ytest = npdata[800:, -2:]

neuron_number = [4, 8, 16, 32, 64, 0]
activation_function = 'relu'

# 4 layers (different parameters give different results)
for first_layer in neuron_number[:-1]:
    for second_layer in neuron_number:
        for third_layer in neuron_number:
            for fourth_layer in neuron_number:
                for fifth_layer in neuron_number:
                    model = Sequential()
                    model.add(Dense(first_layer, activation=activation_function, input_dim=4))
                    if second_layer != 0: model.add(Dense(second_layer, activation=activation_function))
                    if third_layer != 0: model.add(Dense(third_layer, activation=activation_function))
                    if fourth_layer != 0: model.add(Dense(fourth_layer, activation=activation_function))
                    if fifth_layer != 0: model.add(Dense(fifth_layer, activation=activation_function))
                    model.add(Dense(2, activation='linear'))

                    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

                    # Can modify epochs
                    model.fit(xtrain, ytrain, epochs=100, batch_size=40)

                    result = model.evaluate(xtest, ytest, batch_size=40)

                    if result[0] < 1.5:
                        save_name = 'regression_models/model_relu_{}_{}_{}_{}_{}.h5'.format(first_layer, second_layer,
                                                                                         third_layer, fourth_layer,
                                                                                         fifth_layer)
                        model.save(save_name)
