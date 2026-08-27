import torch

from tokenizer import BPETokenizer
from model import TinyGPT


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VOCAB_SIZE = 300
BLOCK_SIZE = 32
BATCH_SIZE = 16

EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 4

LEARNING_RATE = 3e-4
TRAINING_STEPS = 2000


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Training device:", device)


# --------------------------------------------------
# Load text
# --------------------------------------------------

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Characters in dataset:", len(text))


# --------------------------------------------------
# Train tokenizer
# --------------------------------------------------

tokenizer = BPETokenizer()

tokenizer.train(
    text,
    vocab_size=VOCAB_SIZE
)

tokens = tokenizer.encode(text)

print("Tokens in dataset:", len(tokens))


# --------------------------------------------------
# Convert tokens into PyTorch tensor
# --------------------------------------------------

data = torch.tensor(
    tokens,
    dtype=torch.long
)


# --------------------------------------------------
# Train / validation split
# --------------------------------------------------

split_index = int(0.9 * len(data))

train_data = data[:split_index]
validation_data = data[split_index:]

print("Training tokens:", len(train_data))
print("Validation tokens:", len(validation_data))


# --------------------------------------------------
# Batch generator
# --------------------------------------------------

def get_batch(source_data):
    max_start = len(source_data) - BLOCK_SIZE - 1

    if max_start <= 0:
        raise ValueError(
            "Dataset is too small for the selected BLOCK_SIZE."
        )

    starts = torch.randint(
        0,
        max_start,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        source_data[i:i + BLOCK_SIZE]
        for i in starts
    ])

    y = torch.stack([
        source_data[i + 1:i + BLOCK_SIZE + 1]
        for i in starts
    ])

    return x.to(device), y.to(device)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = TinyGPT(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    block_size=BLOCK_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
)

model = model.to(device)

print(
    "Model parameters:",
    sum(p.numel() for p in model.parameters())
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training loop
# --------------------------------------------------

model.train()

for step in range(TRAINING_STEPS):

    x, y = get_batch(train_data)

    logits, loss = model(x, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.4f}"
        )


print("Training complete.")

torch.save(
    model.state_dict(),
    "tinygpt.pt"
)

print("Model saved to tinygpt.pt")