import os

# In case model extension is missing

not_fixed = "./regression_models/not_fixed/"
fixed = "./regression_models/fixed/"

files = [f for f in os.listdir(not_fixed)]


def missing_extension():
    for f in files:
        os.rename(not_fixed + f, fixed + f + ".h5")


def double_point():
    for f in files:
        os.rename(not_fixed + f, fixed + f.replace(".", "_", 1))


double_point()
