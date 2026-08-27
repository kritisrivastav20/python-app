class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def get_pair_counts(self, tokens):
        counts = {}

        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def merge_pair(self, tokens, pair, new_token_id):
        new_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                new_tokens.append(new_token_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def train(self, text, vocab_size):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        tokens = list(text.encode("utf-8"))

        num_merges = vocab_size - 256

        for i in range(num_merges):
            pair_counts = self.get_pair_counts(tokens)

            if not pair_counts:
                break

            pair = max(pair_counts, key=pair_counts.get)

            new_token_id = 256 + i

            tokens = self.merge_pair(
                tokens,
                pair,
                new_token_id
            )

            self.merges[pair] = new_token_id

            self.vocab[new_token_id] = (
                self.vocab[pair[0]] + self.vocab[pair[1]]
            )

        print("Tokenizer training complete")
        print("Vocabulary size:", len(self.vocab))

    def encode(self, text):
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            pair_counts = self.get_pair_counts(tokens)

            possible_merges = {
                pair: self.merges[pair]
                for pair in pair_counts
                if pair in self.merges
            }

            if not possible_merges:
                break

            pair = min(
                possible_merges,
                key=possible_merges.get
            )

            tokens = self.merge_pair(
                tokens,
                pair,
                self.merges[pair]
            )

        return tokens

    def decode(self, tokens):
        raw_bytes = b"".join(
            self.vocab[token]
            for token in tokens
        )

        return raw_bytes.decode(
            "utf-8",
            errors="replace"
        )


if __name__ == "__main__":
    tokenizer = BPETokenizer()

    training_text = """
    hello world
    hello GPT
    hello machine learning
    hello artificial intelligence
    """

    tokenizer.train(
        training_text,
        vocab_size=300
    )

    text = "hello GPT"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("\nOriginal:")
    print(text)

    print("\nEncoded:")
    print(encoded)

    print("\nDecoded:")
    print(decoded)

    texts = [
    "hello",
    "hello hello",
    "GPT",
    "machine learning",
    "🚀",
]