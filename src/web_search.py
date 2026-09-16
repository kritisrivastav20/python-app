import requests
from web_reader import read_page
from retriever import retrieve
from concurrent.futures import ThreadPoolExecutor
import time


SEARCH_URL = "http://localhost:8080/search"
MAX_RESULTS = 3


# search question on the internet
#
def search_web(question):
    parameters = {
        "q": question,
        "format": "json"
    }

    response = requests.get(
        SEARCH_URL,
        params=parameters,
        timeout=30
    )

    response.raise_for_status()

    search_results = response.json()

    return search_results


# get useful results from search response
#
def extract_results(search_results):
    results = []

    web_results = search_results.get(
        "results",
        []
    )

    for result in web_results[
        :MAX_RESULTS
    ]:
        results.append({
            "title": result.get(
                "title",
                ""
            ),

            "url": result.get(
                "url",
                ""
            ),

            "content": result.get(
                "content",
                ""
            )
        })

    return results


# print search results
#
def print_results(results):
    for result in results:
        print()
        print(
            "Title:",
            result["title"]
        )

        print(
            "URL:",
            result["url"]
        )

        print(
            "Content:",
            result["content"]
        )

# read one search result
#
def read_result(result):
    start_time = time.time()

    print(
        "Reading:",
        result["url"]
    )

    text = read_page(
        result["url"]
    )

    total_time = (
        time.time()
        - start_time
    )

    print(
        "Finished:",
        round(
            total_time,
            2
        ),
        "seconds"
    )

    if len(text) == 0:
        return None

    return {
        "title": result["title"],
        "url": result["url"],
        "text": text
    }


# read search result pages concurrently
#
def read_results(results):
    documents = []

    with ThreadPoolExecutor(
        max_workers=MAX_RESULTS
    ) as executor:

        downloaded_results = executor.map(
            read_result,
            results
        )

        for result in downloaded_results:

            if result is None:
                continue

            documents.append(
                result
            )

    return documents


# ask question and search internet
#
def main():
    start_time = time.time()

    question = input(
        "Ask a question: "
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

    print()
    print("-----------------------------")
    print("RELEVANT CONTEXT")
    print("-----------------------------")

    for chunk in relevant_chunks:
        print()
        print(
            "Score:",
            chunk["score"]
        )

        print(
            "Title:",
            chunk["title"]
        )

        print(
            "URL:",
            chunk["url"]
        )

        print()
        print(
            chunk["text"]
        )

    end_time = time.time()

    print()
    print(
        "Total time:",
        round(
            end_time - start_time,
            2
        ),
        "seconds"
    )


if __name__ == "__main__":
    main()