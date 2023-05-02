from keras.models import model_from_json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

target_file_fridge = "D:\\RNN_Data\\Data\\FG Split\\FG_till24.CSV"
input_main_file = "D:\\preprocessed_data\\train_main_file.csv"
target_file_geyser1 = "D:\\RNN_Data\\Data\\GG\\DATALOG.CSV"

MAX_NUM_ROWS = 100
WINDOW_SIZE = 100

geyser_df = pd.read_csv(target_file_geyser1)
geyser_df.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
# removing nan values
geyser_df.dropna()
geyser_df.drop(geyser_df[geyser_df["Current"] == "  NAN "].index, inplace=True)
geyser_df.drop(geyser_df[geyser_df["Frequency"] == ""].index, inplace=True)
geyser_df = geyser_df[["Time", "Current", "Voltage", "Power", "Power Factor"]]

fridge_df = pd.read_csv(target_file_fridge)
fridge_df.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
# removing nan values
fridge_df.dropna()
fridge_df.drop(fridge_df[fridge_df["Current"] == "  NAN "].index, inplace=True)
fridge_df.drop(fridge_df[fridge_df["Frequency"] == ""].index, inplace=True)
fridge_df = fridge_df[["Time", "Current", "Voltage", "Power", "Power Factor"]]
main_df = pd.read_csv(input_main_file)

# load json and create model
json_file = open('model_data/fridge_binary_model.json', 'r')
loaded_fridge_json = json_file.read()
json_file.close()
loaded_fridge = model_from_json(loaded_fridge_json)
# load weights into new model
loaded_fridge.load_weights("model_data/fridge_binary_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_fridge.compile(loss='mean_squared_error', optimizer='adam')

# load json and create model
json_file = open('model_data/geyser_binary_model.json', 'r')
loaded_geyser_json = json_file.read()
json_file.close()
loaded_geyser = model_from_json(loaded_geyser_json)
# load weights into new model
loaded_geyser.load_weights("model_data/geyser_binary_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_geyser.compile(loss='mean_squared_error', optimizer='adam')

main_data = []
gey_data = []
data_a = np.empty([0, MAX_NUM_ROWS], dtype=float)
data_b = np.empty([0, MAX_NUM_ROWS], dtype=float)
data_B = np.empty([0, MAX_NUM_ROWS], dtype=float)
labels = np.empty(0, dtype=int)
batch_ind = 0

# given_time = "09/02/2023 15:11:17"
# given_time = "09/02/2023 18:54:40"
# given_time = "14/02/2023 19:13:35"
# given_time = "14/02/2023 19:25:56"
given_time = "11/02/2023 10:09:13"
# print(main_df.loc[10, 'Start_time'])
main_index = main_df[main_df['Start_time'] == given_time].index.values
sub_df = None
if len(main_index) != 0:
    print("main_index: " + str(main_index[0]))
    sub_df = main_df[main_index[0]:(main_index[0] + WINDOW_SIZE)]

main_data = sub_df["Current"].to_list()
X2 = np.array(main_data, dtype=float)
X2 = X2.reshape(1, len(X2))
len_to_pad = MAX_NUM_ROWS - X2[0].size
X2 = np.pad(X2, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))
data_b = np.append(data_b, X2, axis=0)

gey_index = geyser_df[geyser_df['Time'] == given_time].index.values
if len(gey_index) != 0:
    print("gey_index: " + str(gey_index[0]))
    sub1_df = geyser_df[gey_index[0] -17:(gey_index[0] - 17 + WINDOW_SIZE)]

    gey_data = sub1_df["Current"].to_list()
    X1 = np.array(gey_data, dtype=float)
    X1 = X1.reshape(1, len(X1))
    len_to_pad = MAX_NUM_ROWS - X1[0].size
    X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))
    data_c = np.append(data_a, X1, axis=0)
    data_B = data_b - data_c

fri_index = fridge_df[fridge_df['Time'] == given_time].index.values
if len(fri_index) != 0:
    print("fridge_index: " + str(fri_index[0]))
    sub1_df = fridge_df[fri_index[0]:(fri_index[0] + WINDOW_SIZE)]

    fri_data = sub1_df["Current"].to_list()
    X1 = np.array(fri_data, dtype=float)
    X1 = X1.reshape(1, len(X1))
    len_to_pad = MAX_NUM_ROWS - X1[0].size
    X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))
    data_a = np.append(data_a, X1, axis=0)

print("Given dataset: " + str(data_b))
print("Expected result: " + str(data_a))

data_c = loaded_geyser.predict(data_b)
# data_b -= data_c

print("Predicted geyser: " + str(data_c))
# print("Augmented dataset: " + str(data_b))

ans = loaded_fridge.predict([data_b, data_c])
print("predicted ans: " + str(ans))

print(loaded_fridge.evaluate([data_b, data_c], data_a))