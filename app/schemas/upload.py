from pydantic import BaseModel

class UploadResponse(BaseModel):
    file_name: str
    status: int
    total_chunks: int
    message: str 