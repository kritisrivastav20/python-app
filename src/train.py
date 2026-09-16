import torch

from pathlib import Path

from tokenizer import BPETokenizer
from model import TinyGPT


# --------------------------------------------------
# Files
# --------------------------------------------------

TRAINING_FILE = Path(
    "data/raw/qa_generated.txt"
)

TOKENIZER_FILE = "tokenizer.json"
MODEL_FILE = "tinygpt-best.pt"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

LEARNING_RATE = 3e-4

BLOCK_SIZE = 256
BATCH_SIZE = 16

EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 4

VOCAB_SIZE = 5000

TRAINING_STEPS = 500

EVAL_INTERVAL = 50
EVAL_STEPS = 20

TRAIN_SPLIT = 0.9

RANDOM_SEED = 42

IGNORE_INDEX = -100


# --------------------------------------------------
# Device
# --------------------------------------------------

# select training device
#
def get_device():

    if torch.backends.mps.is_available():
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


# --------------------------------------------------
# Parse QA record
# --------------------------------------------------

# extract context, question and answer
#
def parse_qa_record(
    record
):
    context = ""
    question = ""
    answer = ""

    for line in record.splitlines():

        line = line.strip()

        if line.startswith(
            "Context:"
        ):
            context = line[
                len("Context:"):
            ].strip()

        elif line.startswith(
            "Question:"
        ):
            question = line[
                len("Question:"):
            ].strip()

        elif line.startswith(
            "Answer:"
        ):
            answer = line[
                len("Answer:"):
            ].strip()

    if (
        len(context) == 0
        or len(question) == 0
        or len(answer) == 0
    ):
        return None

    return (
        context,
        question,
        answer
    )


# --------------------------------------------------
# Encode QA record
# --------------------------------------------------

# encode one QA record
#
def encode_qa_record(
    record,
    tokenizer
):
    parsed_record = parse_qa_record(
        record
    )

    if parsed_record is None:
        return None

    context, question, answer = (
        parsed_record
    )

    # ----------------------------------------------
    # Encode individual sections
    # ----------------------------------------------

    context_tokens = tokenizer.encode(
        context
    )

    question_tokens = tokenizer.encode(
        question
    )

    answer_tokens = tokenizer.encode(
        answer
    )


    # ----------------------------------------------
    # Build complete sequence
    # ----------------------------------------------

    sequence = [
        tokenizer.bos_token_id,
        tokenizer.context_token_id
    ]

    sequence.extend(
        context_tokens
    )

    sequence.append(
        tokenizer.question_token_id
    )

    sequence.extend(
        question_tokens
    )

    sequence.append(
        tokenizer.answer_token_id
    )

    answer_start_index = len(
        sequence
    )

    sequence.extend(
        answer_tokens
    )

    sequence.append(
        tokenizer.eos_token_id
    )


    # ----------------------------------------------
    # Reject records larger than context window
    # ----------------------------------------------

    if len(sequence) > BLOCK_SIZE:
        return None


    # ----------------------------------------------
    # Create shifted input and target
    # ----------------------------------------------

    input_ids = sequence[
        :-1
    ]

    target_ids = sequence[
        1:
    ]


    # ----------------------------------------------
    # Ignore context and question loss
    # ----------------------------------------------

    #
    # Example:
    #
    # sequence:
    # BOS CONTEXT ... QUESTION ... ANSWER Purple . EOS
    #
    # input:
    # BOS CONTEXT ... QUESTION ... ANSWER Purple .
    #
    # target:
    # CONTEXT ... QUESTION ... ANSWER Purple . EOS
    #
    # We only want loss for:
    #
    # Purple . EOS
    #

    answer_target_start = (
        answer_start_index - 1
    )

    for index in range(
        answer_target_start
    ):
        target_ids[
            index
        ] = IGNORE_INDEX

    return {
        "input_ids":
            input_ids,

        "target_ids":
            target_ids,

        "context":
            context,

        "question":
            question,

        "answer":
            answer
    }


# --------------------------------------------------
# Encode dataset
# --------------------------------------------------

# encode all QA records
#
def encode_training_records(
    text,
    tokenizer
):
    encoded_records = []

    raw_records = text.split(
        "<END>"
    )

    invalid_records = 0
    oversized_records = 0

    for raw_record in raw_records:

        raw_record = (
            raw_record.strip()
        )

        if len(raw_record) == 0:
            continue

        parsed_record = parse_qa_record(
            raw_record
        )

        if parsed_record is None:

            invalid_records += 1

            continue

        encoded_record = encode_qa_record(
            raw_record,
            tokenizer
        )

        if encoded_record is None:

            oversized_records += 1

            continue

        encoded_records.append(
            encoded_record
        )

    print(
        "QA training records:",
        len(encoded_records)
    )

    if invalid_records > 0:

        print(
            "Invalid records:",
            invalid_records
        )

    if oversized_records > 0:

        print(
            "Records exceeding BLOCK_SIZE:",
            oversized_records
        )

    return encoded_records


# --------------------------------------------------
# Split dataset
# --------------------------------------------------

# shuffle and split complete QA records
#
def split_records(
    records
):
    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )

    indices = torch.randperm(
        len(records),
        generator=generator
    ).tolist()

    shuffled_records = [
        records[index]
        for index in indices
    ]

    split_index = int(
        len(shuffled_records)
        * TRAIN_SPLIT
    )

    train_records = shuffled_records[
        :split_index
    ]

    validation_records = shuffled_records[
        split_index:
    ]

    return (
        train_records,
        validation_records
    )


# --------------------------------------------------
# Create batch
# --------------------------------------------------

# create batch from complete QA records
#
def get_batch(
    records,
    tokenizer,
    device
):
    if len(records) == 0:
        raise ValueError(
            "Cannot create batch "
            "from empty dataset."
        )

    indices = torch.randint(
        0,
        len(records),
        (BATCH_SIZE,)
    )

    selected_records = [
        records[index]
        for index in indices.tolist()
    ]


    # ----------------------------------------------
    # Find longest record in this batch
    # ----------------------------------------------

    max_length = max(
        len(record["input_ids"])
        for record in selected_records
    )

    max_length = min(
        max_length,
        BLOCK_SIZE
    )


    # ----------------------------------------------
    # Create padded batch
    # ----------------------------------------------

    input_batch = []
    target_batch = []

    for record in selected_records:

        input_ids = list(
            record["input_ids"]
        )

        target_ids = list(
            record["target_ids"]
        )


        # ------------------------------------------
        # Truncate if necessary
        # ------------------------------------------

        input_ids = input_ids[
            :max_length
        ]

        target_ids = target_ids[
            :max_length
        ]


        # ------------------------------------------
        # Calculate padding
        # ------------------------------------------

        padding_length = (
            max_length
            - len(input_ids)
        )


        # ------------------------------------------
        # Pad input
        # ------------------------------------------

        input_ids.extend(
            [
                tokenizer.pad_token_id
            ]
            * padding_length
        )


        # ------------------------------------------
        # Ignore padded targets
        # ------------------------------------------

        target_ids.extend(
            [
                IGNORE_INDEX
            ]
            * padding_length
        )


        input_batch.append(
            input_ids
        )

        target_batch.append(
            target_ids
        )


    # ----------------------------------------------
    # Convert to tensors
    # ----------------------------------------------

    x = torch.tensor(
        input_batch,
        dtype=torch.long,
        device=device
    )

    y = torch.tensor(
        target_batch,
        dtype=torch.long,
        device=device
    )

    return (
        x,
        y
    )


# --------------------------------------------------
# Save model
# --------------------------------------------------

# save model checkpoint
#
def save_model(
    model,
    tokenizer
):
    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "vocab_size":
                tokenizer.vocab_size,

            "block_size":
                BLOCK_SIZE,

            "embedding_dim":
                EMBEDDING_DIM,

            "num_heads":
                NUM_HEADS,

            "num_layers":
                NUM_LAYERS
        },
        MODEL_FILE
    )


# --------------------------------------------------
# Estimate loss
# --------------------------------------------------

# calculate train and validation loss
#
@torch.no_grad()
def estimate_loss(
    model,
    train_records,
    validation_records,
    tokenizer,
    device
):
    model.eval()

    results = {}

    datasets = {
        "train":
            train_records,

        "validation":
            validation_records
    }

    for (
        split_name,
        records
    ) in datasets.items():

        losses = []

        for _ in range(
            EVAL_STEPS
        ):

            x, y = get_batch(
                records,
                tokenizer,
                device
            )

            _, loss = model(
                x,
                y
            )

            losses.append(
                loss.item()
            )

        average_loss = (
            sum(losses)
            / len(losses)
        )

        results[
            split_name
        ] = average_loss

    model.train()

    return results


# --------------------------------------------------
# Main training
# --------------------------------------------------

def main():

    # ----------------------------------------------
    # Device
    # ----------------------------------------------

    device = get_device()

    print(
        "Training device:",
        device
    )


    # ----------------------------------------------
    # Load dataset
    # ----------------------------------------------

    if not TRAINING_FILE.exists():

        raise FileNotFoundError(
            f"Training file not found: "
            f"{TRAINING_FILE}"
        )

    text = TRAINING_FILE.read_text(
        encoding="utf-8"
    )

    print(
        "Characters in dataset:",
        len(text)
    )


    # ----------------------------------------------
    # Train tokenizer
    # ----------------------------------------------

    tokenizer = BPETokenizer()

    tokenizer.train(
        text,
        vocab_size=VOCAB_SIZE
    )

    tokenizer.save(
        TOKENIZER_FILE
    )


    # ----------------------------------------------
    # Encode records
    # ----------------------------------------------

    records = encode_training_records(
        text,
        tokenizer
    )

    if len(records) < 2:

        raise ValueError(
            "Not enough valid QA records."
        )


    # ----------------------------------------------
    # Special token information
    # ----------------------------------------------

    print()

    print(
        "BOS:",
        tokenizer.bos_token_id
    )

    print(
        "CONTEXT:",
        tokenizer.context_token_id
    )

    print(
        "QUESTION:",
        tokenizer.question_token_id
    )

    print(
        "ANSWER:",
        tokenizer.answer_token_id
    )

    print(
        "EOS:",
        tokenizer.eos_token_id
    )

    print(
        "PAD:",
        tokenizer.pad_token_id
    )


    # ----------------------------------------------
    # Show first record
    # ----------------------------------------------

    first_record = records[
        0
    ]

    print()
    print(
        "First QA record:"
    )

    print(
        "Context:",
        first_record[
            "context"
        ]
    )

    print(
        "Question:",
        first_record[
            "question"
        ]
    )

    print(
        "Answer:",
        first_record[
            "answer"
        ]
    )

    print()

    print(
        "Input token count:",
        len(
            first_record[
                "input_ids"
            ]
        )
    )

    print(
        "Target token count:",
        len(
            first_record[
                "target_ids"
            ]
        )
    )


    # ----------------------------------------------
    # Show supervised targets
    # ----------------------------------------------

    print()
    print(
        "Tokens contributing to loss:"
    )

    for target_id in first_record[
        "target_ids"
    ]:

        if target_id == IGNORE_INDEX:
            continue

        print(
            target_id,
            tokenizer.tokenizer.id_to_token(
                target_id
            )
        )


    # ----------------------------------------------
    # Split dataset
    # ----------------------------------------------

    (
        train_records,
        validation_records
    ) = split_records(
        records
    )

    print()

    print(
        "Total QA records:",
        len(records)
    )

    print(
        "Training records:",
        len(train_records)
    )

    print(
        "Validation records:",
        len(validation_records)
    )


    # ----------------------------------------------
    # Create model
    # ----------------------------------------------

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        block_size=BLOCK_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )

    model = model.to(
        device
    )


    # ----------------------------------------------
    # Model information
    # ----------------------------------------------

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Model parameters:",
        parameter_count
    )


    # ----------------------------------------------
    # Optimizer
    # ----------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ----------------------------------------------
    # Start training
    # ----------------------------------------------

    print()
    print(
        "Starting answer-focused training..."
    )
    print()

    model.train()

    best_validation_loss = float(
        "inf"
    )


    # ----------------------------------------------
    # Training loop
    # ----------------------------------------------

    for step in range(
        TRAINING_STEPS
    ):

        # ------------------------------------------
        # Evaluate
        # ------------------------------------------

        if (
            step
            % EVAL_INTERVAL
            == 0
        ):

            losses = estimate_loss(
                model,
                train_records,
                validation_records,
                tokenizer,
                device
            )

            train_loss = losses[
                "train"
            ]

            validation_loss = losses[
                "validation"
            ]

            print(
                f"Step {step:4d} | "
                f"Train Loss: "
                f"{train_loss:.4f} | "
                f"Validation Loss: "
                f"{validation_loss:.4f}"
            )


            # --------------------------------------
            # Save best model
            # --------------------------------------

            if (
                validation_loss
                < best_validation_loss
            ):

                best_validation_loss = (
                    validation_loss
                )

                save_model(
                    model,
                    tokenizer
                )

                print(
                    "✓ New best model saved"
                )


        # ------------------------------------------
        # Get training batch
        # ------------------------------------------

        x, y = get_batch(
            train_records,
            tokenizer,
            device
        )


        # ------------------------------------------
        # Forward pass
        # ------------------------------------------

        _, loss = model(
            x,
            y
        )


        # ------------------------------------------
        # Clear gradients
        # ------------------------------------------

        optimizer.zero_grad()


        # ------------------------------------------
        # Backpropagation
        # ------------------------------------------

        loss.backward()


        # ------------------------------------------
        # Update model
        # ------------------------------------------

        optimizer.step()


    # ----------------------------------------------
    # Final evaluation
    # ----------------------------------------------

    final_losses = estimate_loss(
        model,
        train_records,
        validation_records,
        tokenizer,
        device
    )

    final_train_loss = (
        final_losses[
            "train"
        ]
    )

    final_validation_loss = (
        final_losses[
            "validation"
        ]
    )

    print()

    print(
        "Final Train Loss:",
        f"{final_train_loss:.4f}"
    )

    print(
        "Final Validation Loss:",
        f"{final_validation_loss:.4f}"
    )


    # ----------------------------------------------
    # Save final model if better
    # ----------------------------------------------

    if (
        final_validation_loss
        < best_validation_loss
    ):

        best_validation_loss = (
            final_validation_loss
        )

        save_model(
            model,
            tokenizer
        )

        print(
            "✓ Final model is "
            "the new best model"
        )


    # ----------------------------------------------
    # Complete
    # ----------------------------------------------

    print()
    print(
        "Training complete."
    )

    print(
        "Best validation loss:",
        best_validation_loss
    )

    print(
        "Best model saved as:",
        MODEL_FILE
    )

    print(
        "Tokenizer saved as:",
        TOKENIZER_FILE
    )


if __name__ == "__main__":
    main()