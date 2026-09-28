import gc

import pandas as pd
import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from dataloader.dataclass import FFMP_QUEMU
from model.TITANS_Core_MAC import Core
from model.Attention_Pooling import AttentionPooling, WindowAttentionPooling
from titans_pytorch import NeuralMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading training and testing data
with open("tok_data/train_tok/file_count_train.pkl", "rb") as f:
    file_count_train = pickle.load(f)

with open("tok_data/train_tok/label_train.pkl", "rb") as f:
    label_train = pickle.load(f)

with open("tok_data/test_tok/file_count_test.pkl", "rb") as f:
    file_count_test = pickle.load(f)

with open("tok_data/test_tok/label_test.pkl", "rb") as f:
    label_test = pickle.load(f)

#Model structure
class TITANS_model(nn.Module):
    def __init__(self, num_classes, dim_in=768, dim=768, n_blocks=3):
        super().__init__()
        self.in_prj = nn.Linear(dim_in, dim) 
        # self.titans_cores = nn.ModuleList([Core(dim=dim) for _ in range(n_blocks)])
        # self.titans_cores = nn.ModuleList([TitansMAC(input_dim=dim, num_heads=4) for _ in range(n_blocks)])
        self.titans_cores = nn.ModuleList([NeuralMemory(dim=dim, chunk_size=64) for _ in range(n_blocks)])
        
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.SiLU(),
            nn.Linear(dim//2, dim//4),
            nn.SiLU(),
            nn.Linear(dim//4, num_classes)
        )
        self.attn_pooling = AttentionPooling(hidden_size=dim)
        self.window_pooling = WindowAttentionPooling(hidden_size=dim)

    def forward(self, x):
        x = self.in_prj(x)
        # print("Shape before Pooling:", x.shape)
        # x = self.attn_pooling(x).unsqueeze(0)  # Add batch dimension

        # mem_states = []
        # loss_mem = []
        print("Shape before TITANS:", x.shape)
        for block in self.titans_cores:

            x, _ = block(x)
            # mem_states.append(memory_states)

            # loss_mem.append(loss)
            # memory_states.append(memory_state)

        # loss_mem = torch.stack(loss_mem).mean()
        # print("Shape after TITANS:", x.shape)
        # Pool sequence dimension rồi đưa qua classifier đa lớp
        # x = x.squeeze(0)  # Remove batch dimension for window pooling
        x = self.attn_pooling(x)
        # print("Shape after Pooling:", x.shape)
        # x = x.unsqueeze(0)  # Add batch dimension back for classifier
        x = self.window_pooling(x)
        # print("Shape after Pooling:", x.shape)
        x = x.unsqueeze(0)
        logits = self.classifier(x)
        # print("Shape logits:", x.shape)

        return logits


#Model configuration
model = TITANS_model(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay= 0.00001)

batch_size = 16

for epoch in range(10):
    model.train()
    total_loss = 0.0
    total_mem_loss = 0.0
    total_correct = 0
    total_samples = 0

    # for i in tqdm(range(file_count_train)):
    #     outputs = torch.load(f"tok_data/train_tok/outputs_train_{i}.pt")

    #     mapping_cout = np.load(f"tok_data/train_tok/mapping_{i}.npy", allow_pickle=True)

    #     start = 0

    #     batch_data = []

    #     for j in range(len(mapping_cout)):
    #         end = start + mapping_cout[j]
    #         output = outputs[start:end]
    #         batch_data.append(output.to(device))
            
    #         start = end

    #     labels = torch.tensor(
    #         label_train.iloc[i * batch_size:(i + 1) * batch_size].to_numpy(),
    #         dtype=torch.long
    #     ).to(device)

    #     optimizer.zero_grad()

    #     logits, loss_mem = model(batch_data)

    #     loss_cls = criterion(logits, labels)
    #     loss = loss_cls + gamma * loss_mem

    #     loss.backward()
    #     optimizer.step()

    #     total_loss += loss_cls.item() * len(labels)
    #     total_mem_loss += loss_mem.item() * len(labels)
    #     total_correct += (logits.argmax(dim=1) == labels).sum().item()
    #     total_samples += len(labels)
    for i in tqdm(range(file_count_train)):

        outputs = torch.load(
            f"tok_data/train_tok/outputs_train_{i}.pt"
        )

        mapping_count = np.load(
            f"tok_data/train_tok/mapping_{i}.npy",
            allow_pickle=True
        )

        start = 0

        for j in range(len(mapping_count)):

            # ==========================================
            # Lấy windows của 1 sample
            # ==========================================
            end = start + mapping_count[j]

            x = outputs[start:end].to(device)

            # x:
            # [num_windows, 256, 768]

            start = end

            # ==========================================
            # Label của sample
            # ==========================================
            label = torch.tensor(
                [label_train.iloc[i * batch_size + j]],
                dtype=torch.long,
                device=device
            )

            # ==========================================
            # Forward
            # ==========================================
            optimizer.zero_grad()

            logits = model(x)

            # logits: [1, num_classes]
            # label : [1]
            
            loss = criterion(logits, label)

            # ==========================================
            # Backward
            # ==========================================
            loss.backward()
            optimizer.step()

            # ==========================================
            # Statistics
            # ==========================================
            total_loss += loss.item()

            total_correct += (
                logits.argmax(dim=1) == label
            ).sum().item()

            total_samples += 1

            del x
            del label
            del logits
            del loss
            # Python RAM cleanup
            gc.collect()
    
            # GPU cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        del outputs
        del mapping_count
        
    avg_loss = total_loss / total_samples
    avg_mem_loss = total_mem_loss / total_samples
    accuracy = total_correct / total_samples

    print(
        f"Epoch {epoch+1}: "
        f"Loss: {avg_loss:.4f}, "
        f"Memory Loss: {avg_mem_loss:.4f}, "
        f"Accuracy: {accuracy:.4f}"
    )

    # avg_loss = total_loss / total_samples
    # avg_mem_loss = total_mem_loss / total_samples
    # accuracy = total_correct / total_samples

    # print(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}, Memory Loss: {avg_mem_loss:.4f}, Accuracy: {accuracy:.4f}")

print("Training completed!")

torch.save(
    model.state_dict(),
    "model_weight_save/TITANS_model.pth"
)

print("Model weights saved!")
