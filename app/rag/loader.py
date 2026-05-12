from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document

# bổ sung ocr 
# đọc pdf, docx, chưa hỗ trợ pdf bằng scan

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

def validate_file_extension(file_path: str | Path) -> None:
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}, Only PDF and TXT are supported."
        )

def load_txt(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    return [
        Document(
            page_content=text,
            metadata={
                "file_name": path.name,
                "source_path": str(path),
                "page_number": 1, # tại sao??
            },
        )
    ]

def load_pdf(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    reader = PdfReader(str(path))

    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "file_name": path.name,
                    "source_path": str(path),
                    "page_number": page_index,
                },
            )
        )
    
    return documents

def load_document(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    validate_file_extension(file_path)

    extension = path.suffix.lower()

    if extension == ".txt":
        return load_txt(file_path)
    elif extension == ".pdf":
        return load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")
    
