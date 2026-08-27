from tokenizer import BPETokenizer


# -----------------------------------
# 1. Load training text
# -----------------------------------

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()


print("Characters in dataset:", len(text))


# -----------------------------------
# 2. Train tokenizer
# -----------------------------------

tokenizer = BPETokenizer()

tokenizer.train(
    text,
    vocab_size=300
)


# -----------------------------------
# 3. Encode entire dataset
# -----------------------------------

tokens = tokenizer.encode(text)

print("Number of tokens:", len(tokens))

print("\nFirst 30 tokens:")
print(tokens[:30])


# -----------------------------------
# 4. Decode them again as a test
# -----------------------------------

decoded = tokenizer.decode(tokens)

print("\nDecode successful:", decoded == text)

import torch


data = torch.tensor(tokens, dtype=torch.long)

block_size = 8


x = data[:block_size]

y = data[1:block_size + 1]


print("\nInput tokens:")
print(x)

print("\nTarget tokens:")
print(y)

batch_size = 4


def get_batch(data, block_size, batch_size):
    max_start = len(data) - block_size - 1

    starts = torch.randint(
        0,
        max_start,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return x, y


x, y = get_batch(
    data,
    block_size=8,
    batch_size=4
)


print("\nBatch X shape:")
print(x.shape)

print("\nBatch Y shape:")
print(y.shape)

print("\nX:")
print(x)

print("\nY:")
print(y)