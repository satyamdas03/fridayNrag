# rag_utils.py
import os
import logging
import numpy as np
import openai
from faster_whisper import WhisperModel
from moviepy import AudioFileClip
import PyPDF2
import docx
import pandas as pd
import pytesseract
from pytesseract import TesseractNotFoundError
from dotenv import load_dotenv
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image, ImageOps
import fitz  # PyMuPDF
from PIL import Image, ImageOps

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
UPLOAD_DIR    = "uploads"
EMBED_MODEL   = "text-embedding-ada-002"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# Whisper model for audio
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


def preprocess_for_ocr(path: str) -> str:
    """
    Grayscale + upscale to improve Tesseract accuracy.
    Returns a new temp filepath.
    """
    img = Image.open(path)
    img = ImageOps.grayscale(img)
    w, h = img.size
    img = img.resize((w * 2, h * 2), Image.ANTIALIAS)
    temp_path = path + "_prep.png"
    img.save(temp_path)
    return temp_path


def extract_text(path: str) -> str:
    suffix = os.path.splitext(path)[1].lower()
    try:
        # ─── Audio ───────────────────────────────────────────
        if suffix in (".mp3", ".wav", ".m4a"):
            return transcribe_audio(path)

        # ─── Video ───────────────────────────────────────────
        if suffix in (".mp4", ".mov", ".mkv"):
            wav = path + ".wav"
            AudioFileClip(path).write_audiofile(wav, verbose=False, logger=None)
            return transcribe_audio(wav)

        # ─── Image ───────────────────────────────────────────
        if suffix in (".jpg", ".jpeg", ".png"):
            # 1) Preprocess + Tesseract
            prep = preprocess_for_ocr(path)
            try:
                text = pytesseract.image_to_string(prep, config="--psm 6")
                if text.strip():
                    return text
            except TesseractNotFoundError:
                logger.warning("Tesseract not found on PATH")

            # 2) Last‑resort: return empty so you know OCR failed
            return ""

        # ─── PDF ──────────────────────────────────────────────
        if suffix == ".pdf":
            pages = []

            # 1) pdfplumber
            try:
                with pdfplumber.open(path) as pdf:
                    for p in pdf.pages:
                        txt = p.extract_text()
                        if txt:
                            pages.append(txt)
            except Exception as e:
                logger.warning("pdfplumber failed on %s: %s", path, e)
            if pages:
                return "\n".join(pages)

            # 2) PyPDF2
            try:
                reader = PyPDF2.PdfReader(path)
                for p in reader.pages:
                    txt = p.extract_text()
                    if txt:
                        pages.append(txt)
            except Exception as e:
                logger.warning("PyPDF2 failed on %s: %s", path, e)
            if pages:
                return "\n".join(pages)

            # 3) **Pure‑local OCR fallback via PyMuPDF + Tesseract**
            try:
                ocr_pages = []
                # open PDF
                doc = fitz.open(path)
                for i, page in enumerate(doc, start=1):
                    # render page at 2× scale
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat)
                    # convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    tmp_img = f"{path}_page{i}.png"
                    img.save(tmp_img, format="PNG")

                    # preprocess + OCR
                    prep = preprocess_for_ocr(tmp_img)
                    txt = pytesseract.image_to_string(prep, config="--psm 6")
                    if txt.strip():
                        ocr_pages.append(txt)

                return "\n\n".join(ocr_pages)
            except Exception as e:
                logger.error("PyMuPDF‑based OCR fallback failed on %s: %s", path, e)
                return ""

        # ─── DOCX ─────────────────────────────────────────────
        if suffix == ".docx":
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)

        # ─── CSV / XLSX / TXT / PY ───────────────────────────
        if suffix == ".csv":
            return pd.read_csv(path).to_string(index=False)
        if suffix == ".xlsx":
            return pd.read_excel(path, engine="openpyxl").to_string(index=False)
        if suffix in (".txt", ".py"):
            with open(path, encoding="utf8") as f:
                return f.read()

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
    Embed a single chunk of text using OpenAI v1 embeddings.
    """
    try:
        resp = openai.embeddings.create(
            model=EMBED_MODEL,
            input=[text],
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise


def retrieve_similar_chunks(query: str, top_k: int = 3):
    """
    1) Embed the query
    2) Build & embed all chunks
    3) Cosine‑score & return top_k matches
    """
    qv = np.array(embed_text(query))
    qn = np.linalg.norm(qv)
    if qn == 0:
        return []

    corpus = []
    for path in list_uploaded_files():
        txt = extract_text(path)
        for idx, chunk in enumerate(chunk_text(txt), start=1):
            try:
                emb = embed_text(chunk)
                corpus.append((emb, path, idx, chunk))
            except Exception as e:
                logger.warning("Skipping embedding [%s:%d]: %s", path, idx, e)

    sims = []
    for emb, path, idx, chunk in corpus:
        ev = np.array(emb)
        denom = np.linalg.norm(ev) * qn
        if denom > 0:
            sims.append((float(np.dot(ev, qv) / denom), path, idx, chunk))

    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[:top_k]
