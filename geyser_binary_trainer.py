from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout
import pandas as pd
import numpy as np
from datetime import datetime
import sys

WINDOW_SIZE = 100
BATCH_SIZE = 100
MAX_NUM_ROWS = 100
FORMAT = "%d/%m/%Y %H:%M:%S"
GEY_THRESHOLD = 5.0
LEG_BATCH_THRESHOLD = 100

target_file_geyser = "D:\\RNN_Data\\Data\\GG\\DATALOG.CSV"
input_main_file = "D:\\preprocessed_data\\train_main_file.csv"


geyser_df = pd.read_csv(target_file_geyser)
geyser_df.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
# removing nan values
geyser_df.dropna()
geyser_df.drop(geyser_df[geyser_df["Current"] == "  NAN "].index, inplace=True)
geyser_df.drop(geyser_df[geyser_df["Frequency"] == ""].index, inplace=True)
geyser_df = geyser_df[["Time", "Current", "Voltage", "Power", "Power Factor"]]
main_df = pd.read_csv(input_main_file)

# Define the model architecture
model = Sequential()
model.add(LSTM(units=128, input_shape=(100, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dense(units=100, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

print(model.summary())

i = 100
j = 0
main_data = []
gey_data = []
labels = []
data_a = np.empty([0, MAX_NUM_ROWS], dtype=float)
data_b = np.empty([0, MAX_NUM_ROWS], dtype=float)
batch_ind = 0

print("For loop started..")
while j < len(main_df) and i < len(geyser_df):
    try:
        main_dt = datetime.strptime(main_df.loc[j, "Start_time"], FORMAT)
        gey_dt = datetime.strptime(geyser_df.loc[i, "Time"], FORMAT)
        # adjuster
        while gey_dt < main_dt and i < len(geyser_df):
            i = i + 1
            gey_dt = datetime.strptime(geyser_df.loc[i, "Time"], FORMAT)

        temp_j = j
        temp_i = i
        while main_dt == gey_dt:
            main_data.append(float(main_df.loc[temp_j, "Current"]))
            # gey_data.append(float(geyser_df.loc[temp_i, "Current"]))
            labels.append(1 if float(geyser_df.loc[temp_i, "Current"]) > GEY_THRESHOLD else 0)
            temp_i += 1
            temp_j += 1
            main_dt = datetime.strptime(main_df.loc[temp_j, "Start_time"], FORMAT)
            gey_dt = datetime.strptime(geyser_df.loc[temp_i, "Time"], FORMAT)
            if len(main_data) == WINDOW_SIZE:
                X1 = np.array(labels)
                X1 = X1.reshape(1, len(X1))
                len_to_pad = MAX_NUM_ROWS - X1[0].size
                X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                X2 = np.array(main_data)
                X2 = X2.reshape(1, len(X2))
                len_to_pad = MAX_NUM_ROWS - X2[0].size
                X2 = np.pad(X2, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                # print(X2)
                # print(X1)
                data_b = np.append(data_b, X2, axis=0)
                data_a = np.append(data_a, X1, axis=0)
                labels = []
                main_data = []
                # gey_data = []
                i = i + 1
                break

            elif main_dt != gey_dt and len(main_data) > LEG_BATCH_THRESHOLD:
                X1 = np.array(labels)
                X1 = X1.reshape(1, len(X1))
                len_to_pad = MAX_NUM_ROWS - X1[0].size
                X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                X2 = np.array(main_data)
                X2 = X2.reshape(1, len(X2))
                len_to_pad = MAX_NUM_ROWS - X2[0].size
                X2 = np.pad(X2, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                # print("2: " + str(X1))
                data_b = np.append(data_b, X2, axis=0)
                data_a = np.append(data_a, X1, axis=0)
                labels = []
                main_data = []
                # gey_data = []
                i = i + 1
                break
            elif main_dt != gey_dt:
                main_data = []
                # gey_data = []
                labels = []
                i = i + 1

    except Exception as e:
        print(e)
        labels = []
        main_data = []
        # gey_data = []
        print("Error in i:" + str(i) + " j:" + str(j))
        i += 1
    j += 1

print("Model training in progress..")
model.fit(data_b, data_a, epochs=20, batch_size=128, shuffle=True)
print("Model training completed :)")

# serialize model to JSON
model_json = model.to_json()
with open("model_data/geyser_binary_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_data/geyser_binary_model.h5")
print("Saved model to disk")
