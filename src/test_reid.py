import torch
import torchreid

print("Loading OSNet...")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

model = model.to(device)
model.eval()

print("Device:", device)
print("OSNet ready!")

# Create a dummy person image
dummy = torch.randn(1, 3, 256, 128).to(device)

with torch.no_grad():
    embedding = model(dummy)

print("Embedding shape:", embedding.shape)
print("Embedding generated successfully!")
