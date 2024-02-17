import pathlib
import shutil

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings

DATA_PATH = pathlib.Path(__file__).parent / "data"
CHROMA_PATH = pathlib.Path(__file__).parent / "chroma"


def load_documents():
    """Load PDF documents from the data folder"""
    loader = DirectoryLoader("data", glob="*.pdf")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    """Slit our text into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    # TODO: Use logging
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks: list[Document]):
    """Save our chunked documents to local Chroma database"""
    # Clear out the database first.
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH.absolute().as_posix(),
    )
    db.persist()
    # TODO: Switch to logging
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def main():
    """Create a new version of the database at CHROMA_PATH from the PDF documents in DATA_PATH"""
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


if __name__ == "__main__":
    main()
