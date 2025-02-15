from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
ds = load_dataset("clane9/imagenet-100")

print(ds)
