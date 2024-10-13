import os

import pytesseract
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pdf2image import convert_from_path

pdf_path = os.environ.get("FILE_PATH")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

INDEX_NAME = os.environ.get("PINECONE_INDEX")


def pdf_to_images(pdf_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert PDF to images
    images = convert_from_path(pdf_path)

    # Save each page as an image
    for i, image in enumerate(images):
        image_filename = os.path.join(output_dir, f"page_{i + 1}.png")
        image.save(image_filename, "PNG")

    print(f"All pages have been extracted to {output_dir}")


def extract_text_from_pdf_ocr():

    # Step 1: Convert PDF to images
    pages = convert_from_path(pdf_path, 300)  # DPI 300
    print(f"Number of pages: {len(pages)}")

    # Step 2: Create a directory to store the images
    image_dir = "temp_images"
    os.makedirs(image_dir, exist_ok=True)

    # Step 3 & 4: Save each page as an image and perform OCR
    documents = []
    for i, page in enumerate(pages):
        image_path = os.path.join(image_dir, f"page_{i+1}.png")
        page.save(image_path, "PNG")

        text = pytesseract.image_to_string(image_path)
        doc = Document(page_content=text, metadata={"page": i + 1, "source": pdf_path})
        documents.append(doc)

        # Clean up - remove temporary image file
        os.remove(image_path)

    # Clean up - remove temporary directory
    os.rmdir(image_dir)

    return documents


def ingest_docs(documents):
    # Ensure each page is within the token limit for text-embedding-3-small
    # The exact limit isn't public, but we'll use a conservative estimate
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,  # Conservative estimate, adjust if needed
        chunk_overlap=200,
        length_function=len,
    )

    # Split documents if they're too long and update metadata
    split_documents = []
    for doc in documents:
        splits = text_splitter.split_documents([doc])
        for j, split in enumerate(splits):
            # Update metadata
            split.metadata.update({"chunk": j + 1})
            split_documents.append(split)

    # Create vector store
    vector_store = PineconeVectorStore.from_documents(split_documents, embeddings, index_name=INDEX_NAME)

    print(f"Ingested {len(split_documents)} document chunks.")


if __name__ == "__main__":
    # pdf_to_images(pdf_path, "data/travel_guide_pdf2images")
    documents = extract_text_from_pdf_ocr()
    ingest_docs(documents)
