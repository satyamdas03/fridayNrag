# main.py - Complete Updated Version

import os
import json
import logging
import time
import signal
import subprocess
import threading

import numpy as np
import boto3
import psycopg2
import ssl
import certifi

from faster_whisper import WhisperModel
from moviepy import AudioFileClip
from PIL import Image
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# ─── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── AWS & DB Config ───────────────────────────────────────────────────────────
BEDROCK_REGION        = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY        = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MODEL_ID_EMBED        = os.getenv("MODEL_ID_EMBED")
MODEL_ID_CHAT         = os.getenv("MODEL_ID_CHAT")
DATABASE_URL          = os.getenv("DATABASE_URL")

# ─── Whisper model ─────────────────────────────────────────────────────────────
logger.info("Loading Whisper model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
logger.info("Whisper model loaded successfully.")

# ─── Bedrock client ────────────────────────────────────────────────────────────
_ssl_ctx = ssl._create_default_https_context
ssl._create_default_https_context = ssl._create_unverified_context

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
logger.info("Bedrock client initialized.")

# ─── Flask app setup ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Database initialization ───────────────────────────────────────────────────
def init_database():
    """Initialize database with embeddings table, add created_at if missing, and add indexes."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    # 1) Create table if it doesn't exist (no created_at here to avoid errors on re-run)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            embedding FLOAT8[],
            embedding_size INTEGER,
            chunk_text TEXT NOT NULL,
            UNIQUE(file_name, chunk_index)
        );
    """)
    # 2) Add the created_at column if it's not already there
    cursor.execute("""
        ALTER TABLE embeddings
        ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
    """)
    # 3) Create indexes (they’ll only be made if missing)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_file_name 
        ON embeddings(file_name);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_created_at 
        ON embeddings(created_at DESC);
    """)
    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Database initialized (table, created_at column, indexes all set).")


# ─── Utilities ─────────────────────────────────────────────────────────────────
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn, conn.cursor()

def split_text(text, max_words=500, overlap=50):
    words = text.split()
    step = max_words - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i:i+max_words])

def transcribe_audio(path):
    segments, _ = whisper_model.transcribe(path)
    return " ".join(seg.text for seg in segments)

def embed_text_or_image(content, content_type="text", model_id=None):
    model_id = model_id or MODEL_ID_EMBED
    if content_type == "text" and len(content) > 16000:
        content = content[:16000]
    body = json.dumps({
        "inputText" if content_type == "text" else "inputImage": content,
        "dimensions": 256,
        "normalize": True
    })
    resp = bedrock.invoke_model(
        modelId=model_id,
        body=body,
        accept="application/json",
        contentType="application/json",
    )
    data = json.loads(resp["body"].read())
    return data.get("embedding")

def retrieve_similar_chunks(query_embedding, top_k=3):
    conn, cursor = get_db_connection()
    try:
        cursor.execute("""
            SELECT chunk_index, embedding, file_name, file_path, chunk_text
              FROM embeddings
             WHERE embedding IS NOT NULL
             ORDER BY created_at DESC
             LIMIT 1000
        """)
        rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    sims = []
    qv = np.array(query_embedding)
    qn = np.linalg.norm(qv)
    if qn == 0:
        return []
    for idx, emb, fn, fp, txt in rows:
        if not emb:
            continue
        ev = np.array(emb)
        en = np.linalg.norm(ev)
        if en == 0:
            continue
        sim = float(np.dot(ev, qv)/(en*qn))
        sims.append((sim, idx, fn, fp, txt))
    sims.sort(key=lambda x: x[0], reverse=True)
    # filter low sims
    sims = [s for s in sims if s[0] > 0.1]
    return sims[:top_k]

def extract_text(file):
    filename = file.filename
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)
    suffix = os.path.splitext(filename)[1].lower()
    try:
        if suffix in ('.mp3','.wav','.m4a'):
            text = transcribe_audio(path)
        elif suffix in ('.mp4','.mov','.mkv'):
            wav = path + ".wav"
            AudioFileClip(path).write_audiofile(wav)
            text = transcribe_audio(wav)
        elif suffix in ('.jpg','.jpeg','.png'):
            with open(path,'rb') as f:
                img_bytes = f.read()
            tex = boto3.client('textract',
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=BEDROCK_REGION
            )
            res = tex.detect_document_text(Document={'Bytes':img_bytes})
            text = "\n".join([b["Text"] for b in res["Blocks"] if b["BlockType"]=="LINE"])
        elif suffix == '.pdf':
            import PyPDF2
            reader = PyPDF2.PdfReader(path)
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        elif suffix == '.docx':
            import docx
            doc = docx.Document(path)
            text = "\n".join(p.text for p in doc.paragraphs)
        elif suffix in ('.txt','.py','.csv','.xlsx'):
            import pandas as pd
            if suffix=='.csv':
                text = pd.read_csv(path).to_string(index=False)
            elif suffix=='.xlsx':
                text = pd.read_excel(path, engine="openpyxl").to_string(index=False)
            else:
                text = open(path,encoding='utf8').read()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        return text, filename, path
    except Exception as e:
        logger.error(f"extract_text error for {filename}: {e}")
        raise

# ─── Voice Agent process control ──────────────────────────────────────────────
agent_process = None

def start_agent():
    global agent_process
    if agent_process and agent_process.poll() is None:
        return
    agent_process = subprocess.Popen(
        ["python", "livekit_agent.py", "console"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    logger.info(f"Voice agent started (PID {agent_process.pid})")

def stop_agent():
    global agent_process
    if agent_process and agent_process.poll() is None:
        agent_process.terminate()
        agent_process.wait(5)
    agent_process = None

# ─── Flask Routes ────────────────────────────────────────────────────────────

@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')
    if not files:
        return jsonify(error="No files"), 400

    conn, cursor = get_db_connection()
    inserted = []
    for f in files:
        try:
            text, fn, fp = extract_text(f)
            # remove old
            cursor.execute("DELETE FROM embeddings WHERE file_name=%s", (fn,))
            # embed chunks
            count = 0
            for idx, chunk in enumerate(split_text(text), start=1):
                emb = embed_text_or_image(chunk, "text")
                cursor.execute("""
                    INSERT INTO embeddings 
                     (file_name,file_path,chunk_index,embedding,embedding_size,chunk_text)
                     VALUES (%s,%s,%s,%s,%s,%s)
                """, (fn, fp, idx, emb, len(emb), chunk))
                count += 1
            conn.commit()
            inserted.append({'file': fn, 'chunks': count})
            logger.info(f"Uploaded {fn}, {count} chunks")
        except Exception as e:
            logger.error(f"Upload error {f.filename}: {e}")
            conn.rollback()
    cursor.close()
    conn.close()
    return jsonify(results=inserted)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    q = data.get('question') or data.get('message')
    if not q:
        return jsonify(error="No question"), 400

    emb = embed_text_or_image(q)
    chunks = retrieve_similar_chunks(emb, top_k=3)
    context = "\n\n".join(f"From {fn}:\n{txt}" for _,_,fn,_,txt in chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
    body = {
        "schemaVersion": "messages-v1",
        "system": [{"text": "Answer using ONLY context."}],
        "messages": [{"role":"user","content":[{"text":prompt}]}],
        "inferenceConfig":{"maxTokens":512,"temperature":0.7}
    }
    resp = bedrock.invoke_model_with_response_stream(modelId=MODEL_ID_CHAT, body=json.dumps(body))
    def stream():
        for event in resp["body"]:
            chunk = event.get("chunk")
            if chunk:
                data = json.loads(chunk["bytes"].decode())
                yield data["contentBlockDelta"]["delta"]["text"]
        yield ''
    return Response(stream(), content_type='text/plain')

@app.route('/voice', methods=['POST'])
def voice_start():
    start_agent()
    return jsonify(status="started")

@app.route('/voice/stop', methods=['POST'])
def voice_stop():
    stop_agent()
    return jsonify(status="stopped")

@app.route('/voice/status', methods=['GET'])
def voice_status():
    running = agent_process and agent_process.poll() is None
    return jsonify(status="running" if running else "stopped")

# Graceful shutdown
import atexit
atexit.register(stop_agent)
signal.signal(signal.SIGINT, lambda s,f: stop_agent())
signal.signal(signal.SIGTERM, lambda s,f: stop_agent())

# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
