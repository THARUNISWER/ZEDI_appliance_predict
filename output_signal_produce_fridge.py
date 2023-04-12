from keras.models import Sequential, model_from_json
from keras.layers import Dense, SimpleRNN, LSTM, Dropout
import pandas as pd
import numpy as np
from datetime import datetime

WINDOW_SIZE = 100
BATCH_SIZE = 100
MAX_NUM_ROWS = 100
FORMAT = "%d/%m/%Y %H:%M:%S"
GEY_THRESHOLD = 5.0
LEG_BATCH_THRESHOLD = 100

target_file_fridge1 = "D:\\RNN_Data\\Data\\FG Split\\FG_till24.CSV"
target_file_fridge2 = "D:\\RNN_Data\\Data\\FG Split\\FG_from25.CSV"

input_main_file = "D:\\preprocessed_data\\train_main_file.csv"


fridge_df1 = pd.read_csv(target_file_fridge1)
fridge_df1.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
fridge_df2 = pd.read_csv(target_file_fridge2)
fridge_df2.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
fridge_df = pd.concat([fridge_df1, fridge_df2], ignore_index=True)
# removing nan values
fridge_df.dropna()
fridge_df.drop(fridge_df[fridge_df["Current"] == "  NAN "].index, inplace=True)
fridge_df.drop(fridge_df[fridge_df["Frequency"] == ""].index, inplace=True)
fridge_df = fridge_df[["Time", "Current", "Voltage", "Power", "Power Factor"]]
main_df = pd.read_csv(input_main_file)

# load json and create model
json_file = open('model_data/geyser_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_gey = model_from_json(loaded_model_json)
# load weights into new model
loaded_model_gey.load_weights("model_data/geyser_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model_gey.compile(loss='mean_squared_error', optimizer='adam')

# Define the model architecture
model = Sequential()
model.add(LSTM(units=128, input_shape=(100, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
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
fri_data = []
data_a = np.empty([0, MAX_NUM_ROWS], dtype=float)
data_b = np.empty([0, MAX_NUM_ROWS], dtype=float)
labels = np.empty(0, dtype=int)
batch_ind = 0

print("For loop started..")
while j < len(main_df) and i < len(fridge_df):
    print(i)
    try:
        # print(fridge_df.loc[i, "Time"])
        # print(str(i) + " " + str(j))
        main_dt = datetime.strptime(main_df.loc[j, "Start_time"], FORMAT)
        fri_dt = datetime.strptime(str(fridge_df.loc[i, "Time"]), FORMAT)
        # adjuster
        while fri_dt < main_dt and i < len(fridge_df):
            i = i + 1
            fri_dt = datetime.strptime(str(fridge_df.loc[i, "Time"]), FORMAT)

        temp_j = j
        temp_i = i
        while main_dt == fri_dt:
            main_data.append(float(main_df.loc[temp_j, "Current"]))
            fri_data.append(float(fridge_df.loc[temp_i, "Current"]))
            temp_i += 1
            temp_j += 1
            main_dt = datetime.strptime(main_df.loc[temp_j, "Start_time"], FORMAT)
            fri_dt = datetime.strptime(fridge_df.loc[temp_i, "Time"], FORMAT)
            if len(fri_data) == WINDOW_SIZE:
                X1 = np.array(fri_data)
                X1 = X1.reshape(1, len(X1))
                len_to_pad = MAX_NUM_ROWS - X1[0].size
                X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                X2 = np.array(main_data)
                X2 = X2.reshape(1, len(X2))
                len_to_pad = MAX_NUM_ROWS - X2[0].size
                X2 = np.pad(X2, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                data_a = np.append(data_a, X1, axis=0)
                data_b = np.append(data_b, X2, axis=0)
                # if X1[0][0] > GEY_THRESHOLD:
                #     y = np.array(1).reshape(1)
                # else:
                #     y = np.array(0).reshape(1)
                #
                # labels = np.append(labels, y)
                main_data = []
                fri_data = []
                i = i + 1
                break

            elif main_dt != fri_dt and len(main_data) > LEG_BATCH_THRESHOLD:
                X1 = np.array(fri_data)
                X1 = X1.reshape(1, len(X1))
                len_to_pad = MAX_NUM_ROWS - X1[0].size
                X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                X2 = np.array(main_data)
                X2 = X2.reshape(1, len(X2))
                len_to_pad = MAX_NUM_ROWS - X2[0].size
                X2 = np.pad(X2, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                # print("2: " + str(X1))
                data_a = np.append(data_a, X1, axis=0)
                data_b = np.append(data_b, X2, axis=0)

                # if X1[0][0] > GEY_THRESHOLD:
                #     y = np.array(1).reshape(1)
                # else:
                #     y = np.array(0).reshape(1)
                #
                # labels = np.append(labels, y)
                main_data = []
                fri_data = []
                i = i + 1
                break
            elif main_dt != fri_dt:
                main_data = []
                fri_data = []
                i = i + 1

        # if len(data_a) % BATCH_SIZE == 0:
        #     print("Batch " + str(batch_ind) + " ready")
        #     batch_ind += 1
        #     if batch_ind == 50:
        #         break

    except Exception as e:
        print(e)
        main_data = []
        fri_data = []
        print("Error in i:" + str(i) + " j:" + str(j))
        i += 1
    j += 1

print("Model training in progress..")
X3 = loaded_model_gey.predict(data_a)
data_c = data_a - X3
for i in range(0, len(data_a)):
    print("data_c: " + str(data_c[i]))
model.fit(data_c, data_b, epochs=10, batch_size=128, shuffle=True)
print("Model training completed :)")

# serialize model to JSON
model_json = model.to_json()
with open("model_data/fridge_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_data/fridge_model.h5")
print("Saved model to disk")