from keras.models import Sequential, model_from_json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

dt_delta = timedelta(seconds=1)
target_file_geyser = "D:\\RNN_Data\\Data\\G2G\\DATALOG.CSV"
input_main_file = "D:\\preprocessed_data\\test_main_file.csv"

MAX_NUM_ROWS = 100
WINDOW_SIZE = 100

geyser_df = pd.read_csv(target_file_geyser)
geyser_df.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
# removing nan values
geyser_df.dropna()
geyser_df.drop(geyser_df[geyser_df["Current"] == "  NAN "].index, inplace=True)
geyser_df.drop(geyser_df[geyser_df["Frequency"] == ""].index, inplace=True)
geyser_df = geyser_df[["Time", "Current", "Voltage", "Power", "Power Factor"]]
main_df = pd.read_csv(input_main_file)


# load json and create model
json_file = open('model_data/geyser_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_data/geyser_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


main_data = []
gey_data = []
data_a = np.empty([0, MAX_NUM_ROWS], dtype=float)
data_b = np.empty([0, MAX_NUM_ROWS], dtype=float)
labels = np.empty(0, dtype=int)
batch_ind = 0

given_time = '17/02/2023 11:07:19'
# print(main_df.loc[10, 'Start_time'])
main_index = main_df[main_df['Start_time'] == given_time].index.values
if len(main_index) != 0:
    print(main_index)
    sub_df = main_df[main_index[0]:(main_index[0] + WINDOW_SIZE)]

main_data = sub_df["Current"].to_list()
X2 = np.array(main_data, dtype=float)
X2 = X2.reshape(1, len(X2))
len_to_pad = MAX_NUM_ROWS - X2[0].size
X2 = np.pad(X2, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))
data_b = np.append(data_b, X2, axis=0)

gey_index = geyser_df[geyser_df['Time'] == given_time].index.values
if len(gey_index) != 0:
    print(gey_index)
    sub1_df = geyser_df[gey_index[0]:(gey_index[0] + WINDOW_SIZE)]

gey_data = sub1_df["Current"].to_list()
X1 = np.array(gey_data, dtype=float)
X1 = X1.reshape(1, len(X1))
len_to_pad = MAX_NUM_ROWS - X1[0].size
X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))
data_a = np.append(data_a, X1, axis=0)
ans = loaded_model.predict([data_a, data_b])
data_a = np.empty([0, MAX_NUM_ROWS], dtype=float)
print("Actual result: " + str(ans))

gey_data = []
for j in range(1000, len(geyser_df)):
    temp_j = j
    while True:
        gey_data.append(float(geyser_df.loc[temp_j, "Current"]))
        temp_j += 1
        if len(gey_data) == 100:
            X1 = np.array(gey_data)
            X1 = X1.reshape(1, len(X1))
            len_to_pad = MAX_NUM_ROWS - X1[0].size
            X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))
            data_a = np.append(data_a, X1, axis=0)
            ans = loaded_model.predict([data_a, data_b])
            data_a = np.empty([0, MAX_NUM_ROWS], dtype=float)
            print(geyser_df.loc[j, "Time"])
            print("Predicted result: " + str(ans))
            gey_data = []
            break