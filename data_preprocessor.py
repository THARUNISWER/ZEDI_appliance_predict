import pandas as pd
import datetime

DELIM = ","
dt_format = "%d/%m/%Y %H:%M:%S"
gey_curr_threshold = 5.0
dt_delta = datetime.timedelta(seconds=1)
target_file_geyser = "D:\\RNN_Data\\Data\\GG\\DATALOG.CSV"
output_file_geyser = "D:\\preprocessed_data\\geyser_input.csv"
main_file1 = "D:\\RNN_Data\\Data\\Mains\\ES.052 (SN 40844202)_230208_1750_trend.csv"
main_file2 = "D:\\RNN_Data\\Data\\Mains\\ES.053 (SN 40844202)_230216_1601_trend.csv"
main_file3 = "D:\\RNN_Data\\Data\\Mains\\ES.054 (SN 40844202)_230227_1042_trend.csv"
output_main_file = "D:\\preprocessed_data\\main_file.csv"

'''
# main data processing
print("Mains file processing started...")
main1_df = pd.read_csv(main_file1)
main2_df = pd.read_csv(main_file2)
main3_df = pd.read_csv(main_file3)
main_df = pd.concat([main1_df, main2_df, main3_df])
main_df = main_df[["Start(India Standard Time)", "Vrms_AN_avg", "Irms_A_avg", "PowerP_Total_avg",  "Frequency_avg"]]
main_df.columns = ['Start_time', 'Voltage', 'Current', 'Power', 'Frequency']
main_df.to_csv(output_main_file)
print("Mains file processing completed :)")
'''

'''
# geyser data processing
print("Geyser data processing started...")
geyser_df = pd.read_csv(target_file_geyser)
geyser_main_df = pd.DataFrame(columns = ['Start_time', 'End_time', 'size', 'dataset', 'label'])

# removing nan values
geyser_df.dropna()
geyser_df.drop(geyser_df[geyser_df["Current"] == "  NAN "].index, inplace=True)
geyser_df.drop(geyser_df[geyser_df["Frequency"] == ""].index, inplace=True)


start_dt = datetime.datetime(2018, 1, 1)
curr_dt = datetime.datetime(2018, 1, 1)
prev_label = 0
geyser_temp_df = pd.DataFrame(columns=["Time", "Current", "Voltage", "Frequency", "Power Factor"])

for ind in geyser_df.index:
    data_dt = datetime.datetime.strptime(geyser_df["Time"][ind], dt_format)
    try:
        label = 1 if float(geyser_df["Current"][ind]) > gey_curr_threshold else 0
    except:
        break
    if curr_dt != data_dt or label != prev_label:
        if len(geyser_temp_df.index) > 5:
            if curr_dt != data_dt:
                print("time_change: " + str(curr_dt) + " " + str(data_dt))
            else:
                print("label change: " + str(geyser_df["Current"][ind]))
            geyser_main_df.loc[len(geyser_main_df.index)] = [start_dt.strftime(dt_format), (curr_dt - dt_delta).strftime(dt_format), len(geyser_temp_df.index), geyser_temp_df, prev_label]
        geyser_temp_df = pd.DataFrame(columns = ["Time", "Current", "Voltage", "Frequency", "Power Factor"])
        start_dt = data_dt
    curr_dt = data_dt + dt_delta
    geyser_temp_df.loc[len(geyser_temp_df.index)] = [geyser_df["Time"][ind], geyser_df["Current"][ind], geyser_df["Voltage"][ind],
                          geyser_df["Frequency"][ind], geyser_df["Power Factor"][ind]]
    prev_label = label

geyser_main_df.to_csv(output_file_geyser)
print("Geyser data processing over :)")

'''