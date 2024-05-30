import pandas as pd
import numpy as np

df_dict = dict()

model_name = sys.argv[1]
data_file = sys.argv[2]
save_dir = sys.argv[3]

model_name_str = model_name.split('/')[-1]
data_file_str = data_file.split('/')[-1]
INPUT_PATH = save_dir + f'/gpteval_{model_name_str}_{data_file_str}'
OUTPUT_PATH = save_dir + f'/conclusion_gpteval_{model_name_str}_{data_file_str}'

df_dict = pd.read_csv(INPUT_PATH)

out_df = df_dict.groupby("type", sort=False).gpt4_label.value_counts().unstack(fill_value=0)

total_number = np.sum(out_df.values)
total_type_number = np.sum(out_df.values, axis=0)
total_type_acc = total_type_number / total_number

# divide values in each row by the sum of the row
out_df = out_df.div(out_df.sum(axis=1), axis=0)

# restore original order of rows
out_df = out_df.reindex(df_dict.type.unique().tolist())

out_df.loc['总和'] = total_type_acc.tolist()

# display, with numbers as integer percentages
out_df.style.format("{:.0%}")

out_df.to_csv(OUTPUT_PATH)