from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class BPETokenizer:
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    CONTEXT_TOKEN = "<CONTEXT>"
    QUESTION_TOKEN = "<QUESTION>"
    ANSWER_TOKEN = "<ANSWER>"

    def __init__(self):
        self.tokenizer = None


    # train tokenizer
    #
    def train(
        self,
        text,
        vocab_size=5000
    ):
        tokenizer = Tokenizer(
            BPE(
                byte_fallback=True
            )
        )

        tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=False
        )

        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[
                self.PAD_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
                self.CONTEXT_TOKEN,
                self.QUESTION_TOKEN,
                self.ANSWER_TOKEN
            ],
            initial_alphabet=ByteLevel.alphabet()
        )

        tokenizer.train_from_iterator(
            [text],
            trainer=trainer
        )

        self.tokenizer = tokenizer

        print(
            "Tokenizer training complete"
        )

        print(
            "Vocabulary size:",
            self.vocab_size
        )


    # encode text into token ids
    #
    def encode(
        self,
        text,
        add_special_tokens=False
    ):
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer has not been trained or loaded."
            )

        encoding = self.tokenizer.encode(
            text
        )

        token_ids = encoding.ids

        if add_special_tokens:
            token_ids = [
                self.bos_token_id
            ] + token_ids + [
                self.eos_token_id
            ]

        return token_ids


    # decode token ids into text
    #
    def decode(
        self,
        token_ids
    ):
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer has not been trained or loaded."
            )

        token_ids = [
            int(token_id)
            for token_id in token_ids
        ]

        token_ids = [
            token_id
            for token_id in token_ids
            if token_id not in {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id
            }
        ]

        return self.tokenizer.decode(
            token_ids
        )


    # save tokenizer
    #
    def save(
        self,
        path
    ):
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer has not been trained or loaded."
            )

        self.tokenizer.save(
            path
        )

        print(
            "Tokenizer saved to",
            path
        )


    # load tokenizer
    #
    def load(
        self,
        path
    ):
        self.tokenizer = (
            Tokenizer.from_file(
                path
            )
        )

        print(
            "Tokenizer loaded from",
            path
        )

        print(
            "Vocabulary size:",
            self.vocab_size
        )


    @property
    def vocab_size(self):
        if self.tokenizer is None:
            return 0

        return self.tokenizer.get_vocab_size()


    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id(
            self.PAD_TOKEN
        )


    @property
    def bos_token_id(self):
        return self.tokenizer.token_to_id(
            self.BOS_TOKEN
        )


    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id(
            self.EOS_TOKEN
        )

    @property
    def context_token_id(self):
        return self.tokenizer.token_to_id(
            self.CONTEXT_TOKEN
        )


    @property
    def question_token_id(self):
        return self.tokenizer.token_to_id(
            self.QUESTION_TOKEN
        )


    @property
    def answer_token_id(self):
        return self.tokenizer.token_to_id(
            self.ANSWER_TOKEN
        )

if __name__ == "__main__":
    text = """
    The boy is reading a book.
    The programmer is writing Python code.
    Quantum computing uses quantum mechanics.
    """

    tokenizer = BPETokenizer()

    tokenizer.train(
        text,
        vocab_size=500
    )

    tokens = tokenizer.encode(
        "Photosynthesis converts sunlight into energy."
    )

    print()
    print(
        "Token IDs:",
        tokens
    )

    print(
        "Decoded:",
        tokenizer.decode(tokens)
    )