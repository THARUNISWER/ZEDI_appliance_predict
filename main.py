import datetime as datetime
import matplotlib.pyplot as plt
import pandas as pd
import ast
from datetime import datetime

dt_format = "%d/%m/%Y %H:%M:%S"

train_file_geyser = "D:\\preprocessed_data\\geyser_train.csv"
train_df_geyser = pd.read_csv(train_file_geyser)
train_df_geyser = train_df_geyser[["Data_a", "Data_b", "Label"]]
df_a = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_a"][0]))
arr = []
for x in df_a.index:
    arr.append(datetime.strptime(df_a["Time"][x], dt_format))

plt.scatter(arr, df_a["Current"])
df_b = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_b"][0]))
arr = []
print(df_b)
for x in df_b.index:
    arr.append(datetime.strptime(df_b["Time"][x], dt_format))
plt.scatter(arr, df_b["Current"])

