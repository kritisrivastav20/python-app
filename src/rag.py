from web_search import (
    search_web,
    extract_results,
    read_results
)

from retriever import retrieve


# build context from retrieved chunks
#
def build_context(
    relevant_chunks
):
    context_parts = []

    for chunk in relevant_chunks:
        context_parts.append(
            chunk["text"]
        )

    return "\n\n".join(
        context_parts
    )


# retrieve internet context
#
def retrieve_context(
    question
):
    print()
    print(
        "Searching internet..."
    )

    search_results = search_web(
        question
    )

    results = extract_results(
        search_results
    )

    documents = read_results(
        results
    )

    relevant_chunks = retrieve(
        question,
        documents
    )

    context = build_context(
        relevant_chunks
    )

    return context


# run RAG
#
def main():
    question = input(
        "Ask a question: "
    ).strip()

    context = retrieve_context(
        question
    )

    print()
    print(
        "-----------------------------"
    )

    print(
        "RETRIEVED CONTEXT"
    )

    print(
        "-----------------------------"
    )

    print()
    print(
        context
    )


if __name__ == "__main__":
    main()