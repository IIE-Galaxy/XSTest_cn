import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pandas as pd
from tqdm import tqdm
import sys

def load_model_and_tokenzier(model_name, half_model_size = ['13b'], device = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    torch_dtype = None
    for i in half_model_size:
        if i in model_name:
            torch_dtype = torch.float16
            break
    device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(device)
    return model, tokenizer

model_name = sys.argv[1]
data_file = sys.argv[2]
save_dir = sys.argv[3]


model_name_str = model_name.split('/')[-1]
os.makedirs(save_dir, exist_ok = True)

data_file_str = data_file.split('/')[-1]
save_path = save_dir + f'/{model_name_str}_{data_file_str}'

model, tokenizer = load_model_and_tokenzier(model_name, device='cuda:0')

df = pd.read_csv(data_file)
col_name = df.columns.tolist()
col_name.append('answer')
prompt_index = col_name.index('prompt')

save_data = []
start = 0
if os.path.exists(save_path):
    df_data = pd.read_csv(save_path, encoding='utf-8').values.tolist()
    for df_ in df_data:
        save_data.append(df_)
    start = len(df_data)
    
for d in tqdm(df.values.tolist()[start:]):
    prompt = d[prompt_index]
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(input_ids, max_new_tokens = 100)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
    save_data.append(d + [answer])

    df = pd.DataFrame(columns = col_name, data = save_data)
    df.to_csv(save_path, index = False)
