import pandas as pd
import datetime

input_file_geyser = "D:\\preprocessed_data\\geyser_input.csv"
input_main_file = "D:\\preprocessed_data\\main_file.csv"
train_file_geyser = "D:\\preprocessed_data\\geyser_train.csv"

main_df = pd.read_csv(input_main_file)
geyser_df = pd.read_csv(input_file_geyser)
fin_df = pd.DataFrame(columns = ["Data_a", "Data_b", "Label"])

print("Loading training data for geyser...")
for ind in geyser_df.index:
    main_index = main_df[main_df['Start_time'] == geyser_df['Start_time'][ind]].index.values
    if len(main_index) != 0:
        print(main_index)
        sub_df = main_df[main_index[0]:(main_index[0] + geyser_df['size'][ind])]
        fin_df.loc[len(fin_df.index)] = [geyser_df['dataset'][ind], sub_df, geyser_df['label'][ind]]
fin_df.to_csv(train_file_geyser)
print("Data loaded for geyser:)")