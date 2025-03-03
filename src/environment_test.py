import torch

def check_device():
    if torch.cuda.is_available():
        print("CUDA is available. Device:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        print("MPS is available.")
    else:
        print("Neither CUDA nor MPS is available.")

if __name__ == "__main__":
    check_device()