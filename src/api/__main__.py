"""python -m src.api  ->  uvicorn on API_HOST:API_PORT (one worker: FAISS/Chroma are single-process)."""
import uvicorn

from src.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("src.api.app:app", host=s.api_host, port=s.api_port, workers=1)
