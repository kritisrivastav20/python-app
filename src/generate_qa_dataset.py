import random
from pathlib import Path


RESULT_PATH = Path(
    "data/raw/qa_generated.txt"
)

DATASET_SIZE = 5000

random.seed(
    42
)


NAMES = [
    "Alice",
    "Ben",
    "Maya",
    "Ravi",
    "Emma",
    "Noah",
    "Liam",
    "Sophia",
    "Arjun",
    "Sara"
]


OBJECTS = [
    "book",
    "ball",
    "pencil",
    "bag",
    "bottle",
    "box",
    "toy",
    "cup",
    "phone",
    "key"
]


PLACES = [
    "table",
    "chair",
    "bed",
    "door",
    "window",
    "tree",
    "shelf",
    "desk",
    "sofa",
    "wall"
]


COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "orange",
    "purple"
]


REASONS = [
    (
        "went to the market",
        "needed vegetables"
    ),
    (
        "carried an umbrella",
        "it was raining"
    ),
    (
        "drank water",
        "was thirsty"
    ),
    (
        "went to bed",
        "was tired"
    ),
    (
        "opened the window",
        "the room was hot"
    ),
    (
        "wore a jacket",
        "the weather was cold"
    ),
    (
        "visited the library",
        "needed a book"
    ),
    (
        "turned on the light",
        "the room was dark"
    )
]


# create location question
#
def generate_location():
    object_name = random.choice(
        OBJECTS
    )

    place = random.choice(
        PLACES
    )

    relation = random.choice([
        "on",
        "under",
        "beside",
        "near"
    ])

    context = (
        f"The {object_name} is "
        f"{relation} the {place}."
    )

    question = (
        f"Where is the {object_name}?"
    )

    answer = (
        f"{relation.capitalize()} "
        f"the {place}."
    )

    return (
        context,
        question,
        answer
    )


# create attribute question
#
def generate_attribute():
    object_name = random.choice(
        OBJECTS
    )

    color = random.choice(
        COLORS
    )

    context = (
        f"The {object_name} is {color}."
    )

    question = (
        f"What color is the {object_name}?"
    )

    answer = (
        f"{color.capitalize()}."
    )

    return (
        context,
        question,
        answer
    )


# create comparison question
#
def generate_comparison():
    first_name, second_name = (
        random.sample(
            NAMES,
            2
        )
    )

    first_number, second_number = (
        random.sample(
            range(1, 11),
            2
        )
    )

    object_name = random.choice(
        OBJECTS
    )

    context = (
        f"{first_name} has {first_number} "
        f"{object_name}s and "
        f"{second_name} has {second_number} "
        f"{object_name}s."
    )

    question = (
        f"Who has more {object_name}s?"
    )

    if first_number > second_number:
        answer = first_name
    else:
        answer = second_name

    return (
        context,
        question,
        f"{answer}."
    )


# create cause and effect question
#
def generate_reason():
    name = random.choice(
        NAMES
    )

    action, reason = random.choice(
        REASONS
    )

    context = (
        f"{name} {action} because "
        f"{reason}."
    )

    question = (
        f"Why did {name} {action}?"
    )

    answer = (
        f"Because {reason}."
    )

    return (
        context,
        question,
        answer
    )


# format training record
#
def format_record(
    context,
    question,
    answer
):
    return (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"<END>"
    )


# generate dataset
#
def generate_dataset():
    records = set()

    generators = [
        generate_location,
        generate_attribute,
        generate_comparison,
        generate_reason
    ]

    attempts = 0

    while (
        len(records) < DATASET_SIZE
        and attempts < DATASET_SIZE * 20
    ):
        generator = random.choice(
            generators
        )

        context, question, answer = (
            generator()
        )

        record = format_record(
            context,
            question,
            answer
        )

        records.add(
            record
        )

        attempts += 1

    records = list(
        records
    )

    random.shuffle(
        records
    )

    return records


# save dataset
#
def save_dataset(
    records
):
    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_PATH.write_text(
        "\n\n".join(records),
        encoding="utf-8"
    )

    print(
        "QA records:",
        len(records)
    )

    print(
        "Saved to:",
        RESULT_PATH
    )


def main():
    records = generate_dataset()

    save_dataset(
        records
    )


if __name__ == "__main__":
    main()