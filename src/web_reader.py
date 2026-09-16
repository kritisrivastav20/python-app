import trafilatura

MAX_CONTENT_LENGTH = 10000
import requests


REQUEST_TIMEOUT = 8


# download webpage
#
def download_page(url):
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/152.0 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

        return response.text

    except requests.RequestException as error:
        print(
            "Unable to download:",
            url
        )

        print(
            "Error:",
            error
        )

        return None


# extract useful text from webpage
#
def extract_text(page):
    if page is None:
        return ""

    text = trafilatura.extract(
        page,
        include_comments=False,
        include_tables=False,
        include_links=False
    )

    if text is None:
        return ""

    return text.strip()


# limit webpage content
#
def limit_text(text):
    if len(text) <= MAX_CONTENT_LENGTH:
        return text

    return text[
        :MAX_CONTENT_LENGTH
    ]


# read webpage
#
def read_page(url):
    page = download_page(
        url
    )

    text = extract_text(
        page
    )

    return limit_text(
        text
    )


# extract useful text from webpage
#
def extract_text(page):
    if page is None:
        return ""

    text = trafilatura.extract(
        page,
        include_comments=False,
        include_tables=False,
        include_links=False
    )

    if text is None:
        return ""

    return text.strip()


# limit amount of text returned
#
def limit_text(text):
    if len(text) <= MAX_CONTENT_LENGTH:
        return text

    return text[
        :MAX_CONTENT_LENGTH
    ]


# read webpage
#
def read_page(url):
    page = download_page(
        url
    )

    text = extract_text(
        page
    )

    text = limit_text(
        text
    )

    return text


# test webpage reader
#
def main():
    url = input(
        "Enter webpage URL: "
    )

    text = read_page(
        url
    )

    print()
    print(
        "Extracted characters:",
        len(text)
    )

    print()
    print(text)


if __name__ == "__main__":
    main()