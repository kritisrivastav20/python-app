import torch

from model import TinyGPT
from tokenizer import BPETokenizer


MODEL_PATH = "tinygpt-best.pt"
TOKENIZER_PATH = "tokenizer.json"

MAX_NEW_TOKENS = 30
TEMPERATURE = 0.3
TOP_K = 10


# select device
#
def get_device():

    if torch.backends.mps.is_available():
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


# load tokenizer
#
def load_tokenizer():

    tokenizer = BPETokenizer()

    tokenizer.load(
        TOKENIZER_PATH
    )

    return tokenizer


# load model
#
def load_model(
    device
):
    checkpoint = torch.load(
        MODEL_PATH,
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


# build QA prompt
#
def build_prompt_tokens(
    context,
    question,
    tokenizer
):
    tokens = [
        tokenizer.bos_token_id,
        tokenizer.context_token_id
    ]

    tokens.extend(
        tokenizer.encode(
            context
        )
    )

    tokens.append(
        tokenizer.question_token_id
    )

    tokens.extend(
        tokenizer.encode(
            question
        )
    )

    tokens.append(
        tokenizer.answer_token_id
    )

    return tokens


# generate answer
#
def answer_question(
    context,
    question,
    model,
    tokenizer,
    device
):
    prompt_tokens = build_prompt_tokens(
        context,
        question,
        tokenizer
    )

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
            tokenizer.bos_token_id,
            tokenizer.context_token_id,
            tokenizer.question_token_id,
            tokenizer.answer_token_id
        ],
        do_sample=False
    )

    generated_tokens = (
        generated_tokens[0]
        .tolist()
    )

    answer_tokens = generated_tokens[
        len(prompt_tokens):
    ]

    print()
    print(
        "Generated answer token IDs:",
        answer_tokens
    )

    print(
        "Generated answer tokens:"
    )

    for token_id in answer_tokens:

        print(
            token_id,
            tokenizer.tokenizer.id_to_token(
                token_id
            )
        )

    answer = tokenizer.decode(
        answer_tokens
    )

    return answer.strip()


# test model
#
def main():

    device = get_device()

    print(
        "Device:",
        device
    )

    tokenizer = load_tokenizer()

    model = load_model(
        device
    )

    tests = [
        (
            "The bag is purple.",
            "What color is the bag?"
        ),
        (
            "The book is under the table.",
            "Where is the book?"
        ),
        (
            "Alice has 2 balls and Ben has 8 balls.",
            "Who has more balls?"
        ),
        (
            "Maya drank water because was thirsty.",
            "Why did Maya drank water?"
        )
    ]

    print()

    for context, question in tests:

        answer = answer_question(
            context,
            question,
            model,
            tokenizer,
            device
        )

        print(
            "Context:",
            context
        )

        print(
            "Question:",
            question
        )

        print(
            "TinyGPT:",
            answer
        )

        print(
            "-" * 50
        )


if __name__ == "__main__":
    main()