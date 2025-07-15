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

openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = "uploads"
EMBED_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

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
    suffix = os.path.splitext(path)[1].lower()
    try:
        if suffix in (".mp3", ".wav", ".m4a"):
            return transcribe_audio(path)
        if suffix in (".mp4", ".mov", ".mkv"):
            wav = path + ".wav"
            AudioFileClip(path).write_audiofile(wav)
            return transcribe_audio(wav)
        if suffix in (".jpg", ".jpeg", ".png"):
            try:
                return pytesseract.image_to_string(path)
            except TesseractNotFoundError:
                logger.warning("Tesseract not installed; skipping OCR for image %s", path)
                return ""
        if suffix == ".pdf":
            reader = PyPDF2.PdfReader(path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix == ".docx":
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        if suffix in (".txt", ".py"):
            return open(path, encoding="utf8").read()
        if suffix == ".csv":
            return pd.read_csv(path).to_string(index=False)
        if suffix == ".xlsx":
            return pd.read_excel(path, engine="openpyxl").to_string(index=False)
        raise ValueError(f"Unsupported file type: {suffix}")
    except Exception as e:
        logger.error("extract_text error for %s: %s", path, e)
        return ""


def chunk_text(text: str):
    words = text.split()
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        yield " ".join(words[i : i + CHUNK_SIZE])


def embed_text(text: str) -> list[float]:
    resp = openai.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def retrieve_similar_chunks(query: str, top_k: int = 3):
    corpus = []
    for path in list_uploaded_files():
        txt = extract_text(path)
        for idx, chunk in enumerate(chunk_text(txt), start=1):
            emb = embed_text(chunk)
            corpus.append((emb, path, idx, chunk))

    qv = np.array(embed_text(query))
    qn = np.linalg.norm(qv)
    sims = []
    for emb, path, idx, chunk in corpus:
        ev = np.array(emb)
        denom = np.linalg.norm(ev) * qn
        if denom > 0:
            sims.append((float(np.dot(ev, qv) / denom), path, idx, chunk))

    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[:top_k]
