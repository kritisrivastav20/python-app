import re


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_RESULTS = 3


# words that do not add much meaning
#
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "what",
    "when",
    "where",
    "who",
    "why",
    "how"
}


# split text into words
#
def get_words(text):
    words = re.findall(
        r"\b\w+\b",
        text.lower()
    )

    result = []

    for word in words:
        if word not in STOP_WORDS:
            result.append(word)

    return result


# split webpage text into smaller chunks
#
def chunk_text(text):
    chunks = []

    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        chunk = text[
            start:end
        ].strip()

        if len(chunk) > 0:
            chunks.append(chunk)

        start += (
            CHUNK_SIZE
            - CHUNK_OVERLAP
        )

    return chunks


# create chunks from all documents
#
def create_chunks(documents):
    chunks = []

    for document in documents:
        document_chunks = chunk_text(
            document["text"]
        )

        for chunk in document_chunks:
            chunks.append({
                "title": document["title"],
                "url": document["url"],
                "text": chunk
            })

    return chunks


# calculate how relevant a chunk is
# to the user's question
#
def calculate_score(
    question,
    chunk
):
    question_words = get_words(
        question
    )

    chunk_words = get_words(
        chunk
    )

    score = 0

    for word in question_words:
        if word in chunk_words:
            score += 1

    return score


# score all chunks
#
def score_chunks(
    question,
    chunks
):
    scored_chunks = []

    for chunk in chunks:
        score = calculate_score(
            question,
            chunk["text"]
        )

        if score == 0:
            continue

        scored_chunks.append({
            "title": chunk["title"],
            "url": chunk["url"],
            "text": chunk["text"],
            "score": score
        })

    return scored_chunks


# return the most relevant chunks
#
def get_top_chunks(
    scored_chunks,
    max_results=MAX_RESULTS
):
    scored_chunks.sort(
        key=lambda chunk: chunk["score"],
        reverse=True
    )

    return scored_chunks[
        :max_results
    ]


# retrieve relevant context
#
def retrieve(
    question,
    documents
):
    chunks = create_chunks(
        documents
    )

    scored_chunks = score_chunks(
        question,
        chunks
    )

    top_chunks = get_top_chunks(
        scored_chunks
    )

    return top_chunks
