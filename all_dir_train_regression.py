import misc
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras import backend as K


data = misc.getAlldirsRegressionData()

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

neuron_number = [4, 8, 16, 32, 64, 0]
activation_function = 'relu'

# model = Sequential()
# model.add(Dense(8, activation=activation_function, input_dim=4))
# model.add(Dense(4, activation=activation_function, input_dim=4))
# model.add(Dense(32, activation=activation_function, input_dim=4))
# model.add(Dense(16, activation=activation_function, input_dim=4))
# model.add(Dense(2, activation='linear'))
#
# model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
#
# model.fit(xtrain, ytrain, epochs=100, batch_size=40)
#
# result = model.evaluate(xtest, ytest, batch_size=40)
#
# model.save("all_dirs_models/model3.h5")

# 4 layers (different parameters give different results)
first_layer = 16
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
#
#                     Can modify epochs
                model.fit(xtrain, ytrain, epochs=100, batch_size=40, verbose=0)

                result = model.evaluate(xtest, ytest, batch_size=40, verbose=0)

                model_name = 'model_relu_{}_{}_{}_{}_{}.h5'.format(first_layer, second_layer,
                                                                third_layer, fourth_layer,
                                                                fifth_layer)

                print(model_name)

                if result[0] < 0.6:
                    save_name = 'all_dirs_models/' + model_name
                    model.save(save_name)

                K.clear_session()

