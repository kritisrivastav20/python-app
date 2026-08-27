import torch

from tokenizer import BPETokenizer
from model import TinyGPT


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VOCAB_SIZE = 300
BLOCK_SIZE = 32

EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 4


# --------------------------------------------------
# Device
# --------------------------------------------------

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Load training text
# --------------------------------------------------

with open(
    "data/input.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()


# --------------------------------------------------
# Rebuild tokenizer
# --------------------------------------------------

tokenizer = BPETokenizer()

tokenizer.train(
    text,
    vocab_size=VOCAB_SIZE
)


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


# --------------------------------------------------
# Load trained parameters
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        "tinygpt.pt",
        map_location=device
    )
)

model.eval()


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = "The Earth"

prompt_tokens = tokenizer.encode(prompt)

input_tensor = torch.tensor(
    [prompt_tokens],
    dtype=torch.long,
    device=device
)


# --------------------------------------------------
# Generate
# --------------------------------------------------

generated_tokens = model.generate(
    input_tensor,
    max_new_tokens=100,
    temperature=0.2
)


# --------------------------------------------------
# Decode
# --------------------------------------------------

generated_text = tokenizer.decode(
    generated_tokens[0].tolist()
)

print("\nGenerated text:\n")
print(generated_text)