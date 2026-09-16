from pathlib import Path

BASE_PATH = Path("data/raw")
RESULTS_PATH = Path("data/input.txt")

# load documents in raw folder
# 
def load_documents():
    documents = []
    for doc in BASE_PATH.glob("*.txt"): 
        text = doc.read_text("UTF-8").strip()
        if len(text) == 0: return
        documents.append(text)
    return documents

# write all text from documents to input.txt file
# 
def combine_documents():
    documents = load_documents()
    final_result = ""

    for doc in documents:
        final_result += "\n\n" + doc
    RESULTS_PATH.write_text(final_result, "UTF-8")
    return RESULTS_PATH.read_text("UTF-8")

print(combine_documents())