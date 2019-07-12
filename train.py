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

# One hot codification (2 = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
oh_ytrain = to_categorical(ytrain, num_classes=10)
oh_ytest = to_categorical(ytest, num_classes=10)



neuron_number = [4, 8, 16, 32, 64, 128, 0]
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
activation_functions = ['relu', 'softplus']

for activation_function in activation_functions:
    for learning_rate in learning_rates:
        for first_layer in neuron_number[:-1]:
            for second_layer in neuron_number:
                for third_layer in neuron_number:
                    for fourth_layer in neuron_number:
                        model = Sequential()
                        # 4 layers (different parameters give different results)
                        model.add(Dense(first_layer, activation=activation_function, input_dim=4))
                        if second_layer != 0:
                            model.add(Dense(second_layer, activation=activation_function))
                        if third_layer != 0:
                            model.add(Dense(third_layer, activation=activation_function))
                        if fourth_layer != 0:
                            model.add(Dense(fourth_layer, activation=activation_function))
                        model.add(Dense(10, activation='softmax'))

                        model.compile(loss='categorical_crossentropy', optimizer=kopt.SGD(lr=learning_rate),
                                      metrics=['accuracy'])

                        # Can modify epochs
                        model.fit(xtrain, oh_ytrain, epochs=100, batch_size=40)

                        result = model.evaluate(xtest, oh_ytest, batch_size=40)

                        if result[1] > 0.80:
                            # print(result)
                            save_name = 'model_{}_{}_{}_{}_{}_{}'.format(activation_function, learning_rate, first_layer,
                                                                         second_layer, third_layer, fourth_layer)
                            model.save(save_name)
