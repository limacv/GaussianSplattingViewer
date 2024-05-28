import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda
torch_version = torch.__version__

print(f"CUDA Available: {cuda_available}")
print(f"CUDA Version: {cuda_version}")
print(f"PyTorch Version: {torch_version}")

if cuda_available:
    # Print details about the GPU
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f"Number of GPUs: {device_count}")
    print(f"GPU Name: {device_name}")
else:
    print("CUDA is not available. Ensure that CUDA is installed and properly configured.")
