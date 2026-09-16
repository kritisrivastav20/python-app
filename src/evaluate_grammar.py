import torch

from tokenizer import BPETokenizer
from model import TinyGPT


TOKENIZER_PATH = "tokenizer.json"
MODEL_PATH = "tinygpt-best.pt"

BLOCK_SIZE = 64
EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 4


# use MPS when running on Apple Silicon
#
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# grammar questions that we want our model to answer
#
TESTS = [
    {
        "category": "subject_verb",
        "prompt": "The boy",
        "correct": "runs",
        "wrong": "run"
    },
    {
        "category": "subject_verb",
        "prompt": "The boys",
        "correct": "run",
        "wrong": "runs"
    },
    {
        "category": "subject_verb",
        "prompt": "He",
        "correct": "runs",
        "wrong": "run"
    },
    {
        "category": "subject_verb",
        "prompt": "They",
        "correct": "run",
        "wrong": "runs"
    },

    {
        "category": "is_are",
        "prompt": "The boy",
        "correct": "is",
        "wrong": "are"
    },
    {
        "category": "is_are",
        "prompt": "The boys",
        "correct": "are",
        "wrong": "is"
    },
    {
        "category": "is_are",
        "prompt": "He",
        "correct": "is",
        "wrong": "are"
    },
    {
        "category": "is_are",
        "prompt": "They",
        "correct": "are",
        "wrong": "is"
    },

    {
        "category": "has_have",
        "prompt": "He",
        "correct": "has",
        "wrong": "have"
    },
    {
        "category": "has_have",
        "prompt": "She",
        "correct": "has",
        "wrong": "have"
    },
    {
        "category": "has_have",
        "prompt": "They",
        "correct": "have",
        "wrong": "has"
    },
    {
        "category": "has_have",
        "prompt": "We",
        "correct": "have",
        "wrong": "has"
    },

    {
        "category": "do_does",
        "prompt": "The boy",
        "correct": "does",
        "wrong": "do"
    },
    {
        "category": "do_does",
        "prompt": "The boys",
        "correct": "do",
        "wrong": "does"
    },
    {
        "category": "do_does",
        "prompt": "He",
        "correct": "does",
        "wrong": "do"
    },
    {
        "category": "do_does",
        "prompt": "They",
        "correct": "do",
        "wrong": "does"
    },

    {
        "category": "verb_object",
        "prompt": "The student reads the",
        "correct": "book",
        "wrong": "water"
    },
    {
        "category": "verb_object",
        "prompt": "The girl drinks the",
        "correct": "water",
        "wrong": "book"
    },
    {
        "category": "verb_object",
        "prompt": "The teacher opens the",
        "correct": "door",
        "wrong": "milk"
    },
    {
        "category": "verb_object",
        "prompt": "The mother carries the",
        "correct": "bag",
        "wrong": "water"
    },

    {
        "category": "semantic",
        "prompt": "The boy eats the food because the boy is",
        "correct": "hungry",
        "wrong": "tired"
    },
    {
        "category": "semantic",
        "prompt": "The girl drinks the water because the girl is",
        "correct": "thirsty",
        "wrong": "happy"
    },
    {
        "category": "semantic",
        "prompt": "The child sleeps because the child is",
        "correct": "tired",
        "wrong": "hungry"
    },
    {
        "category": "semantic",
        "prompt": "The mother smiles because the mother is",
        "correct": "happy",
        "wrong": "thirsty"
    }
]


# load tokenizer
#
def load_tokenizer():
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)

    return tokenizer


# load trained TinyGPT model
#
def load_model(tokenizer):
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        block_size=BLOCK_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )

    model = model.to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    return model


# get probability of a candidate word after a prompt
#
@torch.no_grad()
def get_word_probability(
    model,
    tokenizer,
    prompt,
    word
):
    prompt_tokens = tokenizer.encode(prompt)

    word_tokens = tokenizer.encode(word)

    if len(word_tokens) != 1:
        raise ValueError(
            f"Expected '{word}' to be one token."
        )

    input_tensor = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device
    )

    # only keep the latest block of tokens
    #
    input_tensor = input_tensor[
        :,
        -BLOCK_SIZE:
    ]

    logits, _ = model(input_tensor)

    # probability distribution for the next word
    #
    next_word_logits = logits[:, -1, :]

    probabilities = torch.softmax(
        next_word_logits,
        dim=-1
    )

    word_token_id = word_tokens[0]

    probability = probabilities[
        0,
        word_token_id
    ].item()

    return probability


# run one grammar test
#
def evaluate_test(
    model,
    tokenizer,
    test
):
    prompt = test["prompt"]
    correct_word = test["correct"]
    wrong_word = test["wrong"]

    correct_probability = get_word_probability(
        model,
        tokenizer,
        prompt,
        correct_word
    )

    wrong_probability = get_word_probability(
        model,
        tokenizer,
        prompt,
        wrong_word
    )

    passed = (
        correct_probability
        > wrong_probability
    )

    print()
    print("Prompt:", prompt)
    print(
        correct_word,
        ":",
        round(correct_probability, 4)
    )
    print(
        wrong_word,
        ":",
        round(wrong_probability, 4)
    )

    if passed:
        print("Result: PASS")
    else:
        print("Result: FAIL")

    return passed


# evaluate all grammar tests
#
def evaluate_grammar():
    print("Device:", device)

    tokenizer = load_tokenizer()
    model = load_model(tokenizer)

    category_results = {}

    total_tests = 0
    total_passed = 0

    for test in TESTS:
        category = test["category"]

        if category not in category_results:
            category_results[category] = {
                "passed": 0,
                "total": 0
            }

        passed = evaluate_test(
            model,
            tokenizer,
            test
        )

        category_results[category]["total"] += 1
        total_tests += 1

        if passed:
            category_results[
                category
            ]["passed"] += 1

            total_passed += 1

    print()
    print("-----------------------------")
    print("GRAMMAR EVALUATION RESULTS")
    print("-----------------------------")

    for category in category_results:
        result = category_results[
            category
        ]

        percentage = (
            result["passed"]
            / result["total"]
        ) * 100

        print(
            category,
            ":",
            f"{percentage:.2f}%"
        )

    overall_percentage = (
        total_passed
        / total_tests
    ) * 100

    print()
    print(
        "Overall grammar accuracy:",
        f"{overall_percentage:.2f}%"
    )

    print(
        "Passed:",
        total_passed,
        "/",
        total_tests
    )


evaluate_grammar()