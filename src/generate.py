import torch

from model import TinyGPT
from tokenizer import BPETokenizer


MODEL_PATH = "tinygpt-best.pt"
TOKENIZER_PATH = "tokenizer.json"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_K = 20


# select device
#
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# load tokenizer
#
def load_tokenizer():
    tokenizer = BPETokenizer()

    tokenizer.load(
        TOKENIZER_PATH
    )

    return tokenizer


# load trained model
#
def load_model(
    model_path,
    device
):
    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model = TinyGPT(
        vocab_size=checkpoint[
            "vocab_size"
        ],
        embedding_dim=checkpoint[
            "embedding_dim"
        ],
        block_size=checkpoint[
            "block_size"
        ],
        num_heads=checkpoint[
            "num_heads"
        ],
        num_layers=checkpoint[
            "num_layers"
        ]
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# generate response
#
def generate_text(
    prompt,
    model,
    tokenizer,
    device
):
    prompt_tokens = tokenizer.encode(
        prompt
    )

    prompt_tokens = [
        tokenizer.bos_token_id
    ] + prompt_tokens

    input_tensor = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device
    )

    generated_tokens = model.generate(
        input_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        eos_token_id=tokenizer.eos_token_id,
        blocked_token_ids=[
            tokenizer.pad_token_id,
            tokenizer.bos_token_id
        ]
    )

    generated_text = tokenizer.decode(
        generated_tokens[0].tolist()
    )

    return generated_text


# run model
#
def main():
    device = get_device()

    print(
        "Using device:",
        device
    )

    tokenizer = load_tokenizer()

    model = load_model(
        MODEL_PATH,
        device
    )

    while True:
        print()

        prompt = input(
            "You: "
        ).strip()

        if prompt.lower() in {
            "exit",
            "quit"
        }:
            break

        if len(prompt) == 0:
            continue

        response = generate_text(
            prompt,
            model,
            tokenizer,
            device
        )

        print()
        print(
            "TinyGPT:",
            response
        )


if __name__ == "__main__":
    main()