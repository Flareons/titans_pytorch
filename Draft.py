import pickle
import torch
import numpy as np

from model.Attention_Pooling import AttentionPooling,  WindowAttentionPooling
from tqdm import tqdm
from model.TITANS_Core_MAC import Core

model = AttentionPooling(hidden_size=768)
window_model = WindowAttentionPooling(hidden_size=768)
core_model = Core(dim=768)
classifier = torch.nn.Linear(768, 2)

with open("tok_data/train_tok/file_count_train.pkl", "rb") as f:
    file_count_train = pickle.load(f)

with open("tok_data/train_tok/label_train.pkl", "rb") as f:
    label_train = pickle.load(f)

for i in range(file_count_train):
    outputs = torch.load(f"tok_data/train_tok/outputs_train_{i}.pt")
    mapping_count = np.load(f"tok_data/train_tok/mapping_{i}.npy", allow_pickle=True)

    print("Outputs shape:", outputs.shape)
    print("Mapping count shape:", mapping_count.shape)
    print("Mapping count sum:", mapping_count.sum())

    start = 0
    for j in range(len(mapping_count)):
        end = start + mapping_count[j]

        print(f"Mapping count for sample {j}: {mapping_count[j]}")

        output = outputs[start:end]
        print(f"Output for sample {j}: {output.shape}")

        # output = model(output.to(next(model.parameters()).device))
        # print(f"Pooled output for sample {j}: {output.shape}")

        output, loss_mem = core_model(output.to(device=next(core_model.parameters()).device))
        print(f"Core model output for sample {j}: {output.shape}, Memory loss: {loss_mem.item()}")

        output = model(output.to(next(model.parameters()).device))
        print(f"Pooled output for sample {j}: {output.shape}")

        output = window_model(output.to(next(window_model.parameters()).device))
        print(f"Window pooled output for sample {j}: {output.shape}")

        print(f"5 sample values from the final output for sample {j}: {output[:5]}")

        output = classifier(output.to(next(classifier.parameters()).device))
        print(f"Classifier output for sample {j}: {output}")

        prob = torch.softmax(output, dim=-1)
        print(f"Probabilities for sample {j}: {prob}")

        start = end

        print("================================")

        break

    break