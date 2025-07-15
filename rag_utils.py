# rag_utils.py
import os
import json
import logging
import numpy as np
import openai

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from moviepy import AudioFileClip
import pandas as pd
import pdfplumber
import PyPDF2
import docx

import pytesseract
from pytesseract import TesseractNotFoundError

try:
    import easyocr
except ImportError:
    easyocr = None

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────
UPLOAD_DIR    = "uploads"
EMBED_MODEL   = "text-embedding-ada-002"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── Whisper for audio ───────────────────────────────────────────────────────────
_whisper = WhisperModel("base", device="cpu", compute_type="int8")

# ─── In‐memory cache of (embedding, path, idx, chunk_text) ────────────────────────
_CORPUS = None

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
    """Extract human‐readable text from any supported file."""
    suffix = os.path.splitext(path)[1].lower()
    try:
        if suffix in (".mp3", ".wav", ".m4a"):
            return transcribe_audio(path)

        if suffix in (".mp4", ".mov", ".mkv"):
            wav = path + ".wav"
            AudioFileClip(path).write_audiofile(wav, verbose=False, logger=None)
            return transcribe_audio(wav)

        if suffix in (".jpg", ".jpeg", ".png"):
            # Try Tesseract first
            try:
                return pytesseract.image_to_string(path)
            except TesseractNotFoundError:
                logger.warning("Tesseract not installed; falling back to EasyOCR for %s", path)
                if easyocr:
                    reader = easyocr.Reader(["en"], gpu=False)
                    return "\n".join(reader.readtext(path, detail=0))
                else:
                    logger.warning("EasyOCR not installed; skipping OCR for %s", path)
                    return ""

        if suffix == ".pdf":
            # pdfplumber is much more reliable
            try:
                with pdfplumber.open(path) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)
            except Exception:
                logger.warning("pdfplumber failed on %s, falling back to PyPDF2", path)
                reader = PyPDF2.PdfReader(path)
                return "\n".join(page.extract_text() or "" for page in reader.pages)

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
        logger.error("extract_text failed for %s: %s", path, e)
        return ""

def chunk_text(text: str) -> list[str]:
    """Yield overlapping word‐based chunks."""
    words = text.split()
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        yield " ".join(words[i : i + CHUNK_SIZE])

def embed_text(text: str) -> list[float]:
    """Call OpenAI once per chunk."""
    resp = openai.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def _build_corpus():
    """Scan uploads/, extract & embed all chunks once."""
    global _CORPUS
    _CORPUS = []
    for path in list_uploaded_files():
        txt = extract_text(path)
        for idx, chunk in enumerate(chunk_text(txt), start=1):
            try:
                emb = embed_text(chunk)
                _CORPUS.append((emb, path, idx, chunk))
            except Exception as e:
                logger.warning("Embedding chunk %d of %s failed: %s", idx, path, e)
    logger.info("Built corpus of %d chunks", len(_CORPUS))

def retrieve_similar_chunks(query: str, top_k: int = 3):
    """
    1) Lazy‐build corpus on first call
    2) Embed query
    3) Cosine‐rank chunks
    """
    global _CORPUS
    if _CORPUS is None:
        _build_corpus()
    # embed query
    qv = np.array(embed_text(query))
    qn = np.linalg.norm(qv)
    sims = []
    for emb, path, idx, chunk in _CORPUS:
        ev = np.array(emb)
        denom = np.linalg.norm(ev) * qn
        if denom > 0:
            sims.append((float(np.dot(ev, qv) / denom), path, idx, chunk))
    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[:top_k]
