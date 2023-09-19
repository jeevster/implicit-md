import torch
from threading import Event

if __name__ == "__main__":
    a = torch.randn([1], device="cuda")
    print("Reserving CUDA_VISIBLE_DEVICES until KeyboardInterrupt.")
    Event().wait()