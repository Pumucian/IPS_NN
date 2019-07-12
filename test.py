from keras.models import load_model
import numpy as np

# (16/32/64/10)
# model = load_model('my_model.h5')
# (16/64/64/10)
# model = load_model('my_model_2.h5')
# (16/64/64/128/10)
# model = load_model('my_model_3.h5')


# a1_47__6_58 = np.array([-72, -63, -76, -91])
# a1_47__6_58_2 = np.array([-72, -63, -76, -93])
# a1_47__6_58_3 = np.array([-72, -63, -76, -92])

# a1_66_8_45 = np.array([-72, -69, -71, -80])
# a1_66_8_45_2 = np.array([-72, -71, -71, -80])
# a1_66_8_45_3 = np.array([-72, -69, -70, -79])

# Max for each beacon (8 first signals)
a1_47__6_58_max = np.array([-72, -64, -76, -93])
a1_66_8_45_max = np.array([-72, -71, -71, -80])

al_points = np.array([a1_47__6_58_max, a1_66_8_45_max])

print(model.predict(al_points, batch_size=6))

