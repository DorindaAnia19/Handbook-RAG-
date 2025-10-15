from config import PDF_PATH
from extractText import extract_text_from_pdf, split_text
from embedUpload import embed_and_store
from retrieveAnswer import retrieve_chunks, generate_answer

print("Extracting text...")
text = extract_text_from_pdf(PDF_PATH)

print("Splitting text...")
chunks = split_text(text)

print("Generating embeddings and uploading to Qdrant...")
embed_and_store(chunks)

# Ask the user for their query
query = input("\nEnter your question: ")
print(f"\nQuery: {query}")

retrieved = retrieve_chunks(query)
answer = generate_answer(query, retrieved)

print("\nAnswer:")
print(answer)


