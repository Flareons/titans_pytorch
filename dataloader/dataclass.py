from torch.utils.data import Dataset

class FFMP_QUEMU(Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        label,
        window_length
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label
        self.window_length = window_length

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.label[idx],
            self.window_length[idx]
        )