from keras.models import Sequential, model_from_json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

WINDOW_SIZE = 100
BATCH_SIZE = 100
MAX_NUM_ROWS = 100
FORMAT = "%d/%m/%Y %H:%M:%S"
GEY_THRESHOLD = 5.0
LEG_BATCH_THRESHOLD = 60

dt_delta = timedelta(seconds=1)
target_file_geyser = "D:\\RNN_Data\\Data\\G2G\\DATALOG.CSV"
input_main_file = "D:\\preprocessed_data\\test_main_file.csv"


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

i = 100
j = 0
main_data = []
gey_data = []
data_a = np.empty([0, MAX_NUM_ROWS], dtype=float)
data_b = np.empty([0, MAX_NUM_ROWS], dtype=float)
labels = np.empty(0, dtype=int)
batch_ind = 0

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
            gey_data.append(float(geyser_df.loc[temp_i, "Current"]))
            temp_i += 1
            temp_j += 1
            main_dt = datetime.strptime(main_df.loc[temp_j, "Start_time"], FORMAT)
            gey_dt = datetime.strptime(geyser_df.loc[temp_i, "Time"], FORMAT)
            if len(gey_data) == WINDOW_SIZE:
                X1 = np.array(gey_data)
                X1 = X1.reshape(1, len(X1))
                len_to_pad = MAX_NUM_ROWS - X1[0].size
                X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                X2 = np.array(main_data)
                X2 = X2.reshape(1, len(X2))
                len_to_pad = MAX_NUM_ROWS - X2[0].size
                X2 = np.pad(X2, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                data_a = np.append(data_a, X1, axis=0)
                data_b = np.append(data_b, X2, axis=0)
                if X1[0][0] > GEY_THRESHOLD:
                    y = np.array(1).reshape(1)
                else:
                    y = np.array(0).reshape(1)

                labels = np.append(labels, y)
                main_data = []
                gey_data = []
                i = i + 1
                break

            elif main_dt != gey_dt and len(main_data) > LEG_BATCH_THRESHOLD:
                X1 = np.array(gey_data)
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

                if X1[0][0] > GEY_THRESHOLD:
                    y = np.array(1).reshape(1)
                else:
                    y = np.array(0).reshape(1)

                labels = np.append(labels, y)
                main_data = []
                gey_data = []
                i = i + 1
                break
            elif main_dt != gey_dt:
                main_data = []
                gey_data = []
                i = i + 1

        if len(data_a) == BATCH_SIZE:
            print("Batch " + str(batch_ind) + " ready")
            batch_ind += 1
            score = loaded_model.evaluate([data_a, data_b], labels, verbose=0)
            print(score)
            print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
            print(labels)
            data_a = np.empty([0, MAX_NUM_ROWS], dtype=float)
            data_b = np.empty([0, MAX_NUM_ROWS], dtype=float)
            labels = np.empty(0, dtype=int)

    except Exception as e:
        print(e)
        main_data = []
        gey_data = []
        print("Error in i:" + str(i) + " j:" + str(j))
        i += 1
    j += 1