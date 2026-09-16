import random
from pathlib import Path


OUTPUT_PATH = Path("data/raw/grammar_generated.txt")
SENTENCES_PER_PATTERN = 2500

random.seed(42)

sentences = set()


# people and animals
#
people = [
    "boy",
    "girl",
    "child",
    "student",
    "teacher",
    "doctor",
    "farmer",
    "worker",
    "engineer",
    "driver",
    "artist",
    "mother",
    "father",
    "brother",
    "sister",
    "friend"
]

animals = [
    "dog",
    "cat",
    "horse",
    "cow",
    "lion",
    "tiger",
    "rabbit",
    "monkey"
]

birds = [
    "bird"
]


# adjectives
#
person_adjectives = [
    "happy",
    "sad",
    "tired",
    "hungry",
    "thirsty",
    "busy",
    "ready",
    "kind",
    "strong",
    "young",
    "old",
    "careful"
]

animal_adjectives = [
    "hungry",
    "thirsty",
    "tired",
    "strong",
    "fast",
    "slow",
    "quiet",
    "small",
    "big"
]


# verbs that do not require an object
#
intransitive_actions = {
    "run": {
        "third": "runs",
        "past": "ran",
        "continuous": "running",
        "subjects": people + animals
    },

    "walk": {
        "third": "walks",
        "past": "walked",
        "continuous": "walking",
        "subjects": people + animals
    },

    "sleep": {
        "third": "sleeps",
        "past": "slept",
        "continuous": "sleeping",
        "subjects": people + animals + birds
    },

    "jump": {
        "third": "jumps",
        "past": "jumped",
        "continuous": "jumping",
        "subjects": people + animals
    },

    "swim": {
        "third": "swims",
        "past": "swam",
        "continuous": "swimming",
        "subjects": [
            "boy",
            "girl",
            "child",
            "student",
            "dog",
            "horse",
            "tiger",
            "monkey"
        ]
    },

    "work": {
        "third": "works",
        "past": "worked",
        "continuous": "working",
        "subjects": [
            "teacher",
            "doctor",
            "farmer",
            "worker",
            "engineer",
            "driver",
            "artist",
            "mother",
            "father"
        ]
    },

    "smile": {
        "third": "smiles",
        "past": "smiled",
        "continuous": "smiling",
        "subjects": people
    },

    "laugh": {
        "third": "laughs",
        "past": "laughed",
        "continuous": "laughing",
        "subjects": people
    },

    "fly": {
        "third": "flies",
        "past": "flew",
        "continuous": "flying",
        "subjects": birds
    }
}


# verbs that require an object
#
transitive_actions = {
    "eat": {
        "third": "eats",
        "past": "ate",
        "continuous": "eating",
        "subjects": people + animals + birds,
        "objects": [
            "apple",
            "banana",
            "bread",
            "rice",
            "food",
            "fruit"
        ]
    },

    "drink": {
        "third": "drinks",
        "past": "drank",
        "continuous": "drinking",
        "subjects": people + animals + birds,
        "objects": [
            "water",
            "milk"
        ]
    },

    "read": {
        "third": "reads",
        "past": "read",
        "continuous": "reading",
        "subjects": [
            "boy",
            "girl",
            "child",
            "student",
            "teacher",
            "doctor",
            "engineer",
            "artist",
            "mother",
            "father",
            "brother",
            "sister"
        ],
        "objects": [
            "book",
            "story",
            "letter"
        ]
    },

    "write": {
        "third": "writes",
        "past": "wrote",
        "continuous": "writing",
        "subjects": people,
        "objects": [
            "letter",
            "story",
            "note"
        ]
    },

    "open": {
        "third": "opens",
        "past": "opened",
        "continuous": "opening",
        "subjects": people,
        "objects": [
            "door",
            "window",
            "book",
            "box"
        ]
    },

    "close": {
        "third": "closes",
        "past": "closed",
        "continuous": "closing",
        "subjects": people,
        "objects": [
            "door",
            "window",
            "book",
            "box"
        ]
    },

    "carry": {
        "third": "carries",
        "past": "carried",
        "continuous": "carrying",
        "subjects": people,
        "objects": [
            "bag",
            "book",
            "box",
            "ball"
        ]
    },

    "clean": {
        "third": "cleans",
        "past": "cleaned",
        "continuous": "cleaning",
        "subjects": people,
        "objects": [
            "room",
            "table",
            "chair",
            "kitchen"
        ]
    },

    "watch": {
        "third": "watches",
        "past": "watched",
        "continuous": "watching",
        "subjects": people,
        "objects": [
            "bird",
            "dog",
            "cat",
            "horse"
        ]
    },

    "see": {
        "third": "sees",
        "past": "saw",
        "continuous": "seeing",
        "subjects": people,
        "objects": [
            "bird",
            "dog",
            "cat",
            "tree",
            "house",
            "car"
        ]
    },

    "find": {
        "third": "finds",
        "past": "found",
        "continuous": "finding",
        "subjects": people,
        "objects": [
            "book",
            "bag",
            "phone",
            "ball",
            "key"
        ]
    },

    "help": {
        "third": "helps",
        "past": "helped",
        "continuous": "helping",
        "subjects": people,
        "objects": [
            "boy",
            "girl",
            "child",
            "student",
            "teacher",
            "doctor",
            "farmer",
            "worker",
            "friend"
        ]
    }
}


# meaningful subject and location combinations
#
location_actions = {
    "study": {
        "third": "studies",
        "subjects": [
            "boy",
            "girl",
            "child",
            "student"
        ],
        "places": [
            "school",
            "library",
            "room"
        ]
    },

    "work": {
        "third": "works",
        "subjects": [
            "teacher",
            "doctor",
            "farmer",
            "worker",
            "engineer",
            "artist"
        ],
        "places": [
            "school",
            "hospital",
            "farm",
            "office"
        ]
    },

    "play": {
        "third": "plays",
        "subjects": [
            "boy",
            "girl",
            "child",
            "student",
            "dog",
            "cat"
        ],
        "places": [
            "park",
            "garden",
            "school"
        ]
    },

    "walk": {
        "third": "walks",
        "subjects": people,
        "places": [
            "park",
            "garden",
            "road"
        ]
    },

    "sleep": {
        "third": "sleeps",
        "subjects": people,
        "places": [
            "house",
            "room"
        ]
    }
}


# add a sentence without duplicates
#
def add_sentence(sentence):
    sentences.add(sentence.strip())


# generate descriptions
#
def generate_descriptions(count):
    for _ in range(count):
        person = random.choice(people)
        adjective = random.choice(person_adjectives)

        add_sentence(
            f"The {person} is {adjective}."
        )

        animal = random.choice(
            animals + birds
        )

        adjective = random.choice(
            animal_adjectives
        )

        add_sentence(
            f"The {animal} is {adjective}."
        )


# generate simple intransitive sentences
#
def generate_intransitive_sentences(count):
    actions = list(
        intransitive_actions.items()
    )

    for _ in range(count):
        base, action = random.choice(actions)

        subject = random.choice(
            action["subjects"]
        )

        add_sentence(
            f"The {subject} {action['third']}."
        )

        add_sentence(
            f"The {subject} {action['past']}."
        )

        add_sentence(
            f"The {subject} is "
            f"{action['continuous']}."
        )

        add_sentence(
            f"The {subject} will {base}."
        )

        add_sentence(
            f"The {subject} can {base}."
        )

        add_sentence(
            f"The {subject} does not {base}."
        )

        add_sentence(
            f"Does the {subject} {base}?"
        )


# generate transitive sentences
#
def generate_transitive_sentences(count):
    actions = list(
        transitive_actions.items()
    )

    for _ in range(count):
        base, action = random.choice(actions)

        subject = random.choice(
            action["subjects"]
        )

        obj = random.choice(
            action["objects"]
        )

        add_sentence(
            f"The {subject} "
            f"{action['third']} the {obj}."
        )

        add_sentence(
            f"The {subject} "
            f"{action['past']} the {obj}."
        )

        add_sentence(
            f"The {subject} is "
            f"{action['continuous']} the {obj}."
        )

        add_sentence(
            f"The {subject} will "
            f"{base} the {obj}."
        )

        add_sentence(
            f"The {subject} can "
            f"{base} the {obj}."
        )

        add_sentence(
            f"The {subject} does not "
            f"{base} the {obj}."
        )

        add_sentence(
            f"Does the {subject} "
            f"{base} the {obj}?"
        )


# generate location sentences
#
def generate_location_sentences(count):
    actions = list(
        location_actions.items()
    )

    for _ in range(count):
        _, action = random.choice(actions)

        subject = random.choice(
            action["subjects"]
        )

        place = random.choice(
            action["places"]
        )

        add_sentence(
            f"The {subject} "
            f"{action['third']} "
            f"in the {place}."
        )


# generate a large pronoun curriculum
#
def generate_pronoun_sentences():
    base_verbs = [
        "run",
        "walk",
        "sleep",
        "jump",
        "work",
        "smile",
        "laugh"
    ]

    adjectives = [
        "happy",
        "sad",
        "tired",
        "hungry",
        "thirsty",
        "ready",
        "strong",
        "kind"
    ]

    objects = [
        "book",
        "bag",
        "ball",
        "letter"
    ]

    singular_pronouns = [
        "He",
        "She"
    ]

    plural_pronouns = [
        "We",
        "They"
    ]

    # I and You
    #
    for pronoun in [
        "I",
        "You"
    ]:

        for verb in base_verbs:
            add_sentence(
                f"{pronoun} {verb}."
            )

            add_sentence(
                f"{pronoun} do not {verb}."
            )

            add_sentence(
                f"{pronoun} will {verb}."
            )

            add_sentence(
                f"{pronoun} can {verb}."
            )

        for adjective in adjectives:
            if pronoun == "I":
                add_sentence(
                    f"I am {adjective}."
                )
            else:
                add_sentence(
                    f"You are {adjective}."
                )

        for obj in objects:
            add_sentence(
                f"{pronoun} have a {obj}."
            )

    # He and She
    #
    for pronoun in singular_pronouns:

        for base in base_verbs:

            third = (
                intransitive_actions[
                    base
                ]["third"]
            )

            add_sentence(
                f"{pronoun} {third}."
            )

            add_sentence(
                f"{pronoun} does not {base}."
            )

            add_sentence(
                f"{pronoun} will {base}."
            )

            add_sentence(
                f"{pronoun} can {base}."
            )

        for adjective in adjectives:
            add_sentence(
                f"{pronoun} is {adjective}."
            )

        for obj in objects:
            add_sentence(
                f"{pronoun} has a {obj}."
            )

    # We and They
    #
    # intentionally generate many examples because
    # these were weak during evaluation
    #
    for _ in range(20):

        for pronoun in plural_pronouns:

            for verb in base_verbs:
                add_sentence(
                    f"{pronoun} {verb}."
                )

                add_sentence(
                    f"{pronoun} do not {verb}."
                )

                add_sentence(
                    f"{pronoun} will {verb}."
                )

                add_sentence(
                    f"{pronoun} can {verb}."
                )

            for adjective in adjectives:
                add_sentence(
                    f"{pronoun} are {adjective}."
                )

            for obj in objects:
                add_sentence(
                    f"{pronoun} have a {obj}."
                )


# directly teach singular versus plural contrasts
#
def generate_pronoun_contrasts():
    verbs = [
        "run",
        "walk",
        "sleep",
        "jump",
        "work",
        "smile"
    ]

    adjectives = [
        "happy",
        "tired",
        "hungry",
        "ready",
        "strong"
    ]

    for verb in verbs:

        third = (
            intransitive_actions[
                verb
            ]["third"]
        )

        add_sentence(
            f"He {third}."
        )

        add_sentence(
            f"They {verb}."
        )

        add_sentence(
            f"She {third}."
        )

        add_sentence(
            f"We {verb}."
        )

        add_sentence(
            f"He does not {verb}."
        )

        add_sentence(
            f"They do not {verb}."
        )

        add_sentence(
            f"She does not {verb}."
        )

        add_sentence(
            f"We do not {verb}."
        )

    for adjective in adjectives:

        add_sentence(
            f"He is {adjective}."
        )

        add_sentence(
            f"They are {adjective}."
        )

        add_sentence(
            f"She is {adjective}."
        )

        add_sentence(
            f"We are {adjective}."
        )

    objects = [
        "book",
        "bag",
        "ball"
    ]

    for obj in objects:

        add_sentence(
            f"He has a {obj}."
        )

        add_sentence(
            f"They have a {obj}."
        )

        add_sentence(
            f"She has a {obj}."
        )

        add_sentence(
            f"We have a {obj}."
        )

def generate_targeted_curriculum():
    for _ in range(25):

        # is / are
        add_sentence("He is happy.")
        add_sentence("She is happy.")
        add_sentence("They are happy.")
        add_sentence("We are happy.")

        add_sentence("He is tired.")
        add_sentence("They are tired.")

        # has / have
        add_sentence("He has a book.")
        add_sentence("She has a book.")
        add_sentence("They have a book.")
        add_sentence("We have a book.")

        # do / does
        add_sentence("He does not run.")
        add_sentence("She does not run.")
        add_sentence("They do not run.")
        add_sentence("We do not run.")

        # tired -> sleep
        add_sentence(
            "The boy sleeps because the boy is tired."
        )

        add_sentence(
            "The girl sleeps because the girl is tired."
        )

        add_sentence(
            "The dog sleeps because the dog is tired."
        )

        # happy -> smile
        add_sentence(
            "The boy smiles because the boy is happy."
        )

        add_sentence(
            "The girl smiles because the girl is happy."
        )

        add_sentence(
            "The mother smiles because the mother is happy."
        )

        # thirsty -> drink
        add_sentence(
            "The boy drinks the water because the boy is thirsty."
        )

        add_sentence(
            "The girl drinks the water because the girl is thirsty."
        )

        # hungry -> eat
        add_sentence(
            "The boy eats the food because the boy is hungry."
        )

        add_sentence(
            "The girl eats the food because the girl is hungry."
        )

# strengthen meaningful verb-object relationships
#
def generate_semantic_object_curriculum():
    repetitions = 25

    for _ in range(repetitions):

        for base, action in (
            transitive_actions.items()
        ):

            for subject in action[
                "subjects"
            ]:

                for obj in action[
                    "objects"
                ]:

                    add_sentence(
                        f"The {subject} "
                        f"{action['third']} "
                        f"the {obj}."
                    )

                    add_sentence(
                        f"The {subject} "
                        f"{action['past']} "
                        f"the {obj}."
                    )

                    add_sentence(
                        f"The {subject} will "
                        f"{base} the {obj}."
                    )

                    add_sentence(
                        f"The {subject} can "
                        f"{base} the {obj}."
                    )

                    add_sentence(
                        f"The {subject} does not "
                        f"{base} the {obj}."
                    )


# strengthen cause and effect relationships
#
def generate_cause_effect_curriculum():
    hungry_subjects = (
        people + animals
    )

    thirsty_subjects = (
        people + animals
    )

    tired_subjects = (
        people + animals
    )

    happy_subjects = people

    food_objects = [
        "food",
        "bread",
        "rice",
        "apple"
    ]

    # hungry -> eat
    #
    for subject in hungry_subjects:

        for obj in food_objects:

            add_sentence(
                f"The {subject} eats "
                f"the {obj} because "
                f"the {subject} is hungry."
            )

            add_sentence(
                f"The {subject} is hungry."
            )

            add_sentence(
                f"The {subject} eats "
                f"the {obj}."
            )

    # thirsty -> drink
    #
    for subject in thirsty_subjects:

        add_sentence(
            f"The {subject} drinks "
            f"the water because "
            f"the {subject} is thirsty."
        )

        add_sentence(
            f"The {subject} is thirsty."
        )

        add_sentence(
            f"The {subject} drinks "
            f"the water."
        )

    # tired -> sleep
    #
    for subject in tired_subjects:

        add_sentence(
            f"The {subject} sleeps "
            f"because the {subject} "
            f"is tired."
        )

        add_sentence(
            f"The {subject} is tired."
        )

        add_sentence(
            f"The {subject} sleeps."
        )

    # happy -> smile
    #
    for subject in happy_subjects:

        add_sentence(
            f"The {subject} smiles "
            f"because the {subject} "
            f"is happy."
        )

        add_sentence(
            f"The {subject} is happy."
        )

        add_sentence(
            f"The {subject} smiles."
        )


# generate compound sentences only using complete verbs
#
def generate_compound_sentences(count):
    actions = list(
        intransitive_actions.items()
    )

    for _ in range(count):

        _, first_action = random.choice(
            actions
        )

        _, second_action = random.choice(
            actions
        )

        first_subject = random.choice(
            first_action["subjects"]
        )

        second_subject = random.choice(
            second_action["subjects"]
        )

        add_sentence(
            f"The {first_subject} "
            f"{first_action['third']} "
            f"and the {second_subject} "
            f"{second_action['third']}."
        )


# generate all training sentences
#
def generate_dataset():
    generate_descriptions(
        SENTENCES_PER_PATTERN
    )

    generate_intransitive_sentences(
        SENTENCES_PER_PATTERN
    )

    generate_transitive_sentences(
        SENTENCES_PER_PATTERN
    )

    generate_location_sentences(
        SENTENCES_PER_PATTERN
    )

    generate_compound_sentences(
        SENTENCES_PER_PATTERN
    )

    generate_pronoun_sentences()

    generate_pronoun_contrasts()

    generate_semantic_object_curriculum()

    generate_cause_effect_curriculum()

    generate_targeted_curriculum()

    return list(sentences)


# save dataset
#
def save_dataset():
    result = generate_dataset()

    random.shuffle(result)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        "\n".join(result),
        encoding="UTF-8"
    )

    print(
        "Grammar dataset generated."
    )

    print(
        "Unique sentences:",
        len(result)
    )

    print(
        "Output:",
        OUTPUT_PATH
    )


save_dataset()