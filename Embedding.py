import pandas as pd
import torch
import numpy as np
import pickle
 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

from utils.NormalizeCode import normalize_code


#Load model and tokenizer
MODEL_NAME = "microsoft/codebert-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
codebert = AutoModel.from_pretrained(MODEL_NAME).to(device)


#Load data
dataset = pd.read_json(r"data\function.json")
print("Dataset loaded successfully.")
print("Dataset shape:", dataset.shape)


#Split data into train and test sets
train_df, test_df = train_test_split(
    dataset,
    test_size=0.2,
    random_state=42,
    stratify=dataset["target"]
)
print("Data split into train and test sets.")
print("Train set shape:", train_df.shape)
print("Test set shape:", test_df.shape)


#Take code and labels from the train and test sets
func_train = train_df["func"]
label_train = train_df["target"]

func_test = test_df["func"]
label_test = test_df["target"]

code_train = func_train.tolist()
code_test = func_test.tolist()


#Normalize code snippets in the dataset
code_train_clean = [
    normalize_code(code)
    for code in code_train
]

code_test_clean = [
    normalize_code(code)
    for code in code_test
]


#Tokenize code snippets using CodeBERT tokenizer
codebert.eval()

batch_size = 16

file_count_train = 0
file_count_test = 0

for i in tqdm(range(0, len(code_train_clean), batch_size)):

    batch = code_train_clean[i:i+batch_size]

    encoded = tokenizer(
        batch,
        max_length=256,
        truncation=True,
        stride=128,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    with torch.inference_mode():
        outputs = codebert(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device)
        )
    mapping = encoded["overflow_to_sample_mapping"].tolist()
    mapping_cout = [mapping.count(i) for i in range(max(mapping) + 1)]

    # list tensor
    torch.save(input_ids, f"tok_data/train_tok/input_ids_train_{file_count_train}.pt")
    torch.save(attention_mask, f"tok_data/train_tok/attention_mask_train_{file_count_train}.pt")
    torch.save(outputs.last_hidden_state, f"tok_data/train_tok/outputs_train_{file_count_train}.pt")

    # list numpy array
    np.save(f"tok_data/train_tok/mapping_{file_count_train}.npy", np.array(mapping_cout, dtype=object), allow_pickle=True)
    file_count_train += 1

    del input_ids
    del attention_mask
    del mapping
    del mapping_cout
    del outputs

    torch.cuda.empty_cache()

with open("tok_data/train_tok/file_count_train.pkl", "wb") as f:
    pickle.dump(file_count_train, f)
with open("tok_data/train_tok/label_train.pkl", "wb") as f:
    pickle.dump(label_train, f)

for i in tqdm(range(0, len(code_test_clean), batch_size)):

    batch = code_test_clean[i:i+batch_size]

    encoded = tokenizer(
        batch,
        max_length=256,
        truncation=True,
        stride=128,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    with torch.inference_mode():
        outputs = codebert(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device)
        )
    mapping = encoded["overflow_to_sample_mapping"].tolist()
    mapping_cout = [mapping.count(i) for i in range(max(mapping) + 1)]

    # list tensor
    torch.save(input_ids, f"tok_data/test_tok/input_ids_test_{file_count_test}.pt")
    torch.save(attention_mask, f"tok_data/test_tok/attention_mask_test_{file_count_test}.pt")
    torch.save(outputs.last_hidden_state, f"tok_data/test_tok/outputs_test_{file_count_test}.pt")

    # list numpy array
    np.save(f"tok_data/test_tok/mapping_{file_count_test}.npy", np.array(mapping_cout, dtype=object), allow_pickle=True)
    file_count_test += 1

    del input_ids
    del attention_mask
    del mapping
    del mapping_cout
    del outputs

    torch.cuda.empty_cache()

with open("tok_data/test_tok/file_count_test.pkl", "wb") as f:
    pickle.dump(file_count_test, f)
with open("tok_data/test_tok/label_test.pkl", "wb") as f:
    pickle.dump(label_test, f)

print("Tokenization completed successfully.")

