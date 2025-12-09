import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _load_docx_text(path: Path) -> str:
    try:
        import docx
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError("python-docx is required for DOCX support") from exc

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".docx", ".doc"}:
        return _load_docx_text(path)
    # Try UTF-8 first, then fallback to common Windows encodings, finally ignore errors.
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        for enc in ("cp1251", "latin-1"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _split_text(text: str, chunk_size: int = 600, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    idx = 0
    while idx < len(words):
        chunk = words[idx : idx + chunk_size]
        chunks.append(" ".join(chunk))
        idx += chunk_size - overlap
    return chunks


@dataclass
class RagResult:
    text: str
    score: float


class RAGEngine:
    def __init__(self, base_path: str = "./vector_stores") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._model_cache: dict[str, SentenceTransformer] = {}

    def _store_dir(self, user_id: int, store_name: str) -> Path:
        path = self.base_path / str(user_id) / store_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _metadata_path(self, user_id: int, store_name: str) -> Path:
        return self._store_dir(user_id, store_name) / "chunks.json"

    def _index_path(self, user_id: int, store_name: str) -> Path:
        return self._store_dir(user_id, store_name) / "index.faiss"

    def _get_model(self, embedding_model: str) -> SentenceTransformer:
        if embedding_model not in self._model_cache:
            self._model_cache[embedding_model] = SentenceTransformer(embedding_model)
        return self._model_cache[embedding_model]

    async def ingest_file(
        self, user_id: int, store_name: str, file_path: str, embedding_model: str
    ) -> Tuple[int, str]:
        path = Path(file_path)
        text = await asyncio.to_thread(_read_text, path)
        chunks = _split_text(text)
        if not chunks:
            raise ValueError("No text extracted from document.")

        model = self._get_model(embedding_model)
        embeddings = await asyncio.to_thread(model.encode, chunks, convert_to_numpy=True)

        index_path = self._index_path(user_id, store_name)
        meta_path = self._metadata_path(user_id, store_name)

        if index_path.exists():
            index = faiss.read_index(str(index_path))
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            index = faiss.IndexFlatL2(embeddings.shape[1])
            meta = []

        index.add(embeddings)
        start_idx = len(meta)
        for i, chunk in enumerate(chunks):
            meta.append({"text": chunk, "source": path.name, "idx": start_idx + i})

        faiss.write_index(index, str(index_path))
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("Ingested %s chunks for user %s", len(chunks), user_id)
        return len(chunks), str(index_path)

    async def search(
        self, user_id: int, store_name: str, query: str, embedding_model: str, top_k: int = 3
    ) -> List[RagResult]:
        index_path = self._index_path(user_id, store_name)
        meta_path = self._metadata_path(user_id, store_name)
        if not index_path.exists() or not meta_path.exists():
            return []

        model = self._get_model(embedding_model)
        query_vec = await asyncio.to_thread(
            model.encode, [query], convert_to_numpy=True
        )
        index = faiss.read_index(str(index_path))
        scores, ids = index.search(query_vec, top_k)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        results: List[RagResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1 or idx >= len(meta):
                continue
            results.append(RagResult(text=meta[idx]["text"], score=float(score)))
        return results

    async def reset_store(self, user_id: int, store_name: str) -> None:
        dir_path = self._store_dir(user_id, store_name)
        for file in ["index.faiss", "chunks.json"]:
            fp = dir_path / file
            if fp.exists():
                fp.unlink()

    async def ensure_base(self) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)

