from pydantic import BaseModel

class Source(BaseModel):
    file_name: str
    page_number: int | None = None 
    chunk_id: str
    content_preview: str | None = None 