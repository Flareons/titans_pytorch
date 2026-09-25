# clear_memory.py

import gc
import torch


def clear_memory():
    """
    Clear Python RAM objects that are no longer referenced
    and release unused CUDA cached memory.
    """

    # Garbage collector - RAM
    collected = gc.collect()

    # CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Đồng bộ CUDA trước khi kết thúc
        torch.cuda.synchronize()

    print(f"Garbage collected: {collected} objects")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2

        print(f"CUDA allocated: {allocated:.2f} MB")
        print(f"CUDA reserved : {reserved:.2f} MB")
    else:
        print("CUDA is not available.")


if __name__ == "__main__":
    clear_memory()

clear_memory()
