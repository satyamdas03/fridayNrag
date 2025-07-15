# rag_utils.py
import os
import json
import numpy as np
import logging
import openai
from faster_whisper import WhisperModel
from moviepy import AudioFileClip
import PyPDF2
import docx
import pandas as pd
import pytesseract
from pytesseract import TesseractNotFoundError
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make sure your OPENAI_API_KEY is set in .env
openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR     = "uploads"
EMBED_MODEL    = "text-embedding-ada-002"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50

# Whisper for audio transcription
_whisper = WhisperModel("base", device="cpu", compute_type="int8")


def list_uploaded_files() -> list[str]:
    return [
        os.path.join(UPLOAD_DIR, fn)
        for fn in os.listdir(UPLOAD_DIR)
        if os.path.isfile(os.path.join(UPLOAD_DIR, fn))
    ]


def transcribe_audio(path: str) -> str:
    segments, _ = _whisper.transcribe(path)
    return " ".join(seg.text for seg in segments)


def extract_text(path: str) -> str:
    """Extract readable text from various file types."""
    suffix = os.path.splitext(path)[1].lower()
    try:
        if suffix in (".mp3", ".wav", ".m4a"):
            return transcribe_audio(path)

        if suffix in (".mp4", ".mov", ".mkv"):
            wav = path + ".wav"
            AudioFileClip(path).write_audiofile(wav, verbose=False, logger=None)
            return transcribe_audio(wav)

        if suffix in (".jpg", ".jpeg", ".png"):
            try:
                return pytesseract.image_to_string(path)
            except TesseractNotFoundError:
                logger.warning("Tesseract not installed; skipping OCR for %s", path)
                return ""

        if suffix == ".pdf":
            try:
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    return "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                logger.warning("PyPDF2 failed on %s; no text extracted", path)
                return ""

        if suffix == ".docx":
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)

        if suffix in (".txt", ".py"):
            with open(path, encoding="utf8") as f:
                return f.read()

        if suffix == ".csv":
            return pd.read_csv(path).to_string(index=False)

        if suffix == ".xlsx":
            return pd.read_excel(path, engine="openpyxl").to_string(index=False)

        raise ValueError(f"Unsupported file type: {suffix}")

    except Exception as e:
        logger.error("extract_text error for %s: %s", path, e)
        return ""


def chunk_text(text: str):
    """Yield overlapping chunks of CHUNK_SIZE words."""
    words = text.split()
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        yield " ".join(words[i : i + CHUNK_SIZE])


def embed_text(text: str) -> list[float]:
    """
    Embed a single chunk of text.
    Uses the new v1 embeddings interface: openai.embeddings.create(...)
    """
    try:
        resp = openai.embeddings.create(
            model=EMBED_MODEL,
            input=[text],     # must be a list
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise


def retrieve_similar_chunks(query: str, top_k: int = 3):
    """
    1) Embed the query
    2) Load & embed all chunks from all uploaded files
    3) Compute cosine similarities
    4) Return top_k matches
    """
    # 1) embed the query
    qv = np.array(embed_text(query))
    qn = np.linalg.norm(qv)
    if qn == 0:
        return []

    # 2) build corpus on the fly
    corpus = []
    for path in list_uploaded_files():
        txt = extract_text(path)
        for idx, chunk in enumerate(chunk_text(txt), start=1):
            try:
                emb = embed_text(chunk)
                corpus.append((emb, path, idx, chunk))
            except Exception as e:
                logger.warning("Skipping failed chunk embedding [%s:%d]: %s", path, idx, e)

    # 3) score them
    sims = []
    for emb, path, idx, chunk in corpus:
        ev = np.array(emb)
        denom = np.linalg.norm(ev) * qn
        if denom > 0:
            sims.append((float(np.dot(ev, qv) / denom), path, idx, chunk))

    # 4) pick top_k
    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[:top_k]

