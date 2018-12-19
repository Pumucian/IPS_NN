import os, csv
from xlrd import open_workbook
from statistics import mean

# This script generates the data set for the NN to learn

# Gets excel file names
files = [f for f in os.listdir('.') if '.xlsx' in f]

data = []

# Points represented in "human" alphabetic order (from 1_6 to 2_10)
point_order = [4, 0, 1, 2, 3, 9, 5, 6, 7, 8]


# Data has 10 rows representing 10 ref points, each point should
# have arrays of 4 representing frequencies from 4 beacons
for file in files:

    book = open_workbook(file)
    sheet = book.sheet_by_index(0)
    point = []

    # 100 values ignoring column names and time
    for i in range(1, 101):
        frequencies = sheet.row_values(i, 1)[0:7:2]
        # Empty data (represented as "") becomes the mean of previous entries
        if "" in frequencies:
            empty_index = frequencies.index("")
            frequencies[empty_index] = mean([row[empty_index] for row in point])

        point.append(frequencies)

    # Label provided in the point_order list since files are processed in a real alphabetic order
    label = point_order.pop(0)
    for p in point:
        p.append(label)
        data.append(p)


with open("dataset_iBeacon_ch37_0degrees_test1_ED_IB500.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

