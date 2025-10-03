import re
import textwrap
import os

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from transformers import AutoTokenizer
from unstructured.documents.elements import ListItem, NarrativeText, Title, Element
from unstructured.partition.pdf import partition_pdf

FILE_PATH = "documents"
FILE_NAME = "OWASP-Top-10-for-LLMs-2025.pdf"
START_MARKER = "LLM01:2025"
END_MARKER = "Appendix 1"
SKIP_FILLER = "OWASP Top 10 for LLM Applications v2.0"
SKIP_SECTION_TITLES = {
    "Reference Links",
    "Related Frameworks and Taxonomies",
}
LLM = "llama3.2:1b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def get_vectorstore() -> Chroma:
    """
    Load vectorstore or build a new one.

    Returns
    -------
    Chroma
        vectorstore.

    """
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db_path = "./owasp_chroma_db"

    if os.path.exists(db_path):
        print("Loading existing vectorstore...")
        vectorstore = Chroma(
            embedding_function=embedding_model,
            persist_directory=db_path,
        )
        print("✓ Vector store ready!")
        return vectorstore
    else:
        print("Building vectorstore...")

        content_elements = load_pdf_elements()
        chunks = create_chunks(content_elements)
        rag_ready_chunks = tokenize_chunks(chunks)

        documents = [
            Document(page_content=chunk["page_content"], metadata=chunk["metadata"])
            for chunk in rag_ready_chunks
        ]

        vectorstore = Chroma.from_documents(
            embedding=embedding_model,
            collection_name="owasp_db",
            persist_directory=db_path,
            documents=documents,
        )
        print("✓ Vector store ready!")
        return vectorstore


def load_pdf_elements() -> list[Element]:
    """
    Load and extract relevant elements from the PDF.

    Returns
    -------
    list[Element]
        List of content elements between the start and end markers.
    """
    print("Parsing PDF...")
    elements = partition_pdf(filename=f"{FILE_PATH}/{FILE_NAME}")

    start_index = None
    end_index = None
    for i, e in enumerate(elements):
        if START_MARKER in e.text.strip() and isinstance(e, Title):
            start_index = i

        if END_MARKER in e.text.strip() and isinstance(e, Title):
            end_index = i
            break

    content_elements = [
        e for e in elements[start_index:end_index]
        if isinstance(e, (Title, NarrativeText, ListItem))
    ]

    print(f"✓ Found {len(content_elements)} content elements.")
    return content_elements

def create_chunks(content_elements):
    """
    Create chunks from content elements.


    Parameters
    ----------
    content_elements : list[Elements]
        List of elements from the PDF.

    Returns
    -------
    list[dict]
        List of chunks with "page_content" and "metadata".
    """
    chunks = []
    current_chunk_text = []
    section_context = ""
    current_heading = ""
    is_skipping_section = False

    for e in content_elements:
        text = e.text.strip()
        if not text or text == SKIP_FILLER:
            continue

        if isinstance(e, Title):
            if section_context and text != section_context:
                if len(current_chunk_text) > 1:
                    metadata = {"section": section_context, "heading": current_heading, "source": f"{FILE_NAME}"}
                    chunks.append({"page_content": "\n".join(current_chunk_text), "metadata": metadata})
                    current_chunk_text = []

            if text in SKIP_SECTION_TITLES:
                is_skipping_section = True
                continue
            
            is_skipping_section = False
            
            if re.match(r"^LLM\d{2}:2025", text):
                section_context = text
            
            if not current_chunk_text:
                current_heading = text
                current_chunk_text.append(text)

        elif not is_skipping_section:
            current_chunk_text.append(text)

    if current_chunk_text and not is_skipping_section:
        metadata = {"section": section_context, "heading": current_heading, "source": f"{FILE_NAME}"}
        chunks.append({"page_content": "\n".join(current_chunk_text), "metadata": metadata})

    print(f"✓ Created {len(chunks)} chunks!")
    return chunks

def split_by_tokens(text: str, tokenizer: AutoTokenizer, max_tokens: int =512, overlap: int =50) -> list[str]:
    """
    Splits a single string of text into smaller token-sized pieces.

    Parameters
    ----------
    text : str
        The text to split into smaller chunks.
    tokenizer : AutoTokenizer
        The tokenizer to use for encoding/decoding text.
    max_tokens : int
        Max number of tokens per chunk.
    overlap : int
        Number of overlapping tokens between chunks.

    Returns
    -------
    list[str]
        List of text chunks, each under token limit.

    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return [text]
        
    result = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        result.append(chunk_text)
        start += max_tokens - overlap

    return result

def tokenize_chunks(chunks: list[dict]) -> list[dict]:
    """
    Split chunks into smaller token-sized chunks.

    Parameters
    ----------
    chunks : list[dict]
        List of chunks with "page_content and "metadata".

    Returns
    -------
    list[dict]
        List of smaller chunks, each under token limit.

    """
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    rag_ready_chunks = []

    for ch in chunks:
        original_text = ch['page_content']
        metadata = ch['metadata']
        
        sub_texts = split_by_tokens(original_text, tokenizer, max_tokens=512, overlap=50)
        
        for sub_text in sub_texts:
            rag_ready_chunks.append({
                "page_content": sub_text,
                "metadata": metadata
            })

    print(f"✓ Split into {len(rag_ready_chunks)} RAG-ready chunks.")
    return rag_ready_chunks


def setup_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """
    QA chain with LLM retriever.

    Parameters
    ----------
    vectorstore : Chroma
        Vectorstore used for retrieval.

    Returns
    -------
    RetrievalQA
        Configured QA chain.

    """
    print("Loading LLM model...")

    llm = OllamaLLM(
        model=LLM,
        temperature=0.1,
    )

    custom_prompt = PromptTemplate(
        template="""You are an expert assistant for answering questions about the OWASP Top 10 for LLM Applications document.
        Use only the following retrieved context to answer the question.
        If you don't know the answer from the context provided, just say that you do not have enough information to answer.
        Be concise and directly answer the question.
        
        CONTEXT: {context}
        QUESTION: {question}
        
        Provide a concise security-focused answer:""",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    test_response = llm.invoke("Say 'ready' if you're working")

    print(f"✓ QA system response: {test_response}")
    return qa_chain

def ask_question(question: str, qa_chain: RetrievalQA) -> None:
    """
    Ask a question and display answer with sources.

    Parameters
    ----------
    question : str
        The question.
    qa_chain : RetrievalQA
        QA chain used for answering.

    """
    print(f"\nQUESTION: {question}\n")
    
    response = qa_chain.invoke({"query": question})
    
    print("ANSWER:")
    print("=" * 80)
    # Use textwrap to make the answer readable.
    print(textwrap.fill(response['result'], width=80))
    print("=" * 80)


def main() -> None:
    """Main function running the QA system."""
    print("Starting OWASP QA System...")

    vectorstore = get_vectorstore()

    qa_chain = setup_qa_chain(vectorstore)

    print("\nReady for questions. type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Enter your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("Bye!")
                break
            if not question:
                print("Please enter a question.\n")
                continue

            ask_question(question, qa_chain)
            print("\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")



if __name__ == "__main__":
    main()