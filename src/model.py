import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim: int, head_dim: int):
        super().__init__()

        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

    def forward(self, x):
        # x shape:
        # [batch_size, sequence_length, embedding_dim]
        _, T, _ = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Compare queries against keys
        attention_scores = q @ k.transpose(-2, -1)

        # Scale scores for numerical stability
        attention_scores = attention_scores / math.sqrt(k.shape[-1])

        # Causal mask:
        # token at position t can only see positions <= t
        mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        attention_scores = attention_scores.masked_fill(
            ~mask,
            float("-inf")
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        output = attention_weights @ v

        return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList([
            SelfAttentionHead(
                embedding_dim=embedding_dim,
                head_dim=head_dim
            )
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):
        # Run all attention heads independently
        outputs = [
            head(x)
            for head in self.heads
        ]

        # Join them along the embedding dimension
        x = torch.cat(outputs, dim=-1)

        # Mix information from all heads
        x = self.projection(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int
    ):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        # Attention + residual connection
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # Feed-forward + residual connection
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x


class TinyGPT(nn.Module):
    @torch.no_grad()
    def generate(
        self,
        token_ids,
        max_new_tokens=50,
        temperature=1.0
    ):
        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the most recent context
            token_ids_context = token_ids[:, -self.block_size:]

            logits, _ = self(token_ids_context)

            # Take predictions for the last position
            logits = logits[:, -1, :]

            # Control randomness
            logits = logits / temperature

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Sample one next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append it to the sequence
            token_ids = torch.cat(
                [token_ids, next_token],
                dim=1
            )

        return token_ids


    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
        num_heads: int,
        num_layers: int
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids, targets=None):
        _, T = token_ids.shape

        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block size {self.block_size}"
            )

        token_embeddings = self.token_embedding(token_ids)

        positions = torch.arange(
            T,
            device=token_ids.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.final_layer_norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits_flat = logits.reshape(
                B * T,
                C
            )

            targets_flat = targets.reshape(
                B * T
            )

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss


if __name__ == "__main__":
    model = TinyGPT(
        vocab_size=300,
        embedding_dim=32,
        block_size=8,
        num_heads=4,
        num_layers=3
    )

    # --------------------------------------------------
    # Test 1: Forward pass without targets
    # --------------------------------------------------

    x = torch.tensor([
        [10, 25, 50, 100],
        [20, 30, 40, 50]
    ])

    logits, loss = model(x)

    print("Input shape:")
    print(x.shape)

    print("\nOutput logits shape:")
    print(logits.shape)

    print("\nLoss without targets:")
    print(loss)

    print("\nNumber of parameters:")
    print(sum(p.numel() for p in model.parameters()))

    # --------------------------------------------------
    # Test 2: Convert logits to probabilities
    # --------------------------------------------------

    last_token_logits = logits[0, -1]

    probabilities = torch.softmax(
        last_token_logits,
        dim=-1
    )

    print("\nProbability sum:")
    print(probabilities.sum().item())

    top_probabilities, top_tokens = torch.topk(
        probabilities,
        k=5
    )

    print("\nTop 5 predicted tokens:")

    for token, probability in zip(
        top_tokens,
        top_probabilities
    ):
        print(
            f"Token {token.item()} -> "
            f"{probability.item():.4f}"
        )

    # --------------------------------------------------
    # Test 3: Forward pass with targets and loss
    # --------------------------------------------------

    y = torch.tensor([
        [25, 50, 100, 72],
        [30, 40, 50, 60]
    ])

    logits, loss = model(x, y)

    print("\nLogits shape:")
    print(logits.shape)

    print("\nCross-entropy loss:")
    print(loss.item())
