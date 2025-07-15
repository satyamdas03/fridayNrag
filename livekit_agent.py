# livekit_agent.py
import os
import logging
import asyncio
import openai as _openai_sdk

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentSession, Agent, WorkerOptions, JobContext,
    RoomInputOptions, ChatContext, ChatMessage
)
from livekit.plugins import openai, sarvam, noise_cancellation, silero

from rag_utils import retrieve_similar_chunks, list_uploaded_files

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-agent")

BASE_SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant. Answer using ONLY the information "
    "provided in the context."
)

SESSION_GREETING = (
    "Hello! I'm your voice assistant. I can answer questions about the files you've uploaded. What would you like to know?"
)


class RAGAssistant(Agent):
    def __init__(self, chat_ctx: ChatContext = None):
        super().__init__(
            chat_ctx=chat_ctx or ChatContext(),
            instructions=BASE_SYSTEM_INSTRUCTION
        )

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        q = new_message.text_content.strip()
        if not list_uploaded_files():
            turn_ctx.add_message(
                role="assistant",
                content="I don’t see any uploaded files. Please upload some files first."
            )
            return

        loop = asyncio.get_event_loop()
        top = await loop.run_in_executor(None, retrieve_similar_chunks, q, 3)

        if not top:
            reply = "I don't have enough information in the uploaded files to answer that question."
        else:
            ctx = "\n\n".join(
                f"From {os.path.basename(path)} (chunk {idx}):\n{chunk}"
                for _, path, idx, chunk in top
            )
            reply = f"{ctx}\n\nAnswer:"
        turn_ctx.add_message(role="assistant", content=reply)


async def entrypoint(ctx: JobContext):
    sdk_client = _openai_sdk.AsyncClient(api_key=os.getenv("OPENAI_API_KEY"))

    session = AgentSession(
        stt=sarvam.STT(
            language=os.getenv("SARVAM_STT_LANG", "en-IN"),
            model=os.getenv("SARVAM_STT_MODEL", "saarika:v2.5"),
        ),
        llm=openai.LLM(
            client=sdk_client,
            model="gpt-4o",
            temperature=0.3
        ),
        tts=sarvam.TTS(
            target_language_code=os.getenv("SARVAM_TTS_LANG", "en-IN"),
            model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v2"),
            speaker=os.getenv("SARVAM_SPEAKER", "anushka"),
        ),
        vad=silero.VAD.load(),
    )

    await ctx.connect()

    assistant = RAGAssistant(chat_ctx=ChatContext())
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            video_enabled=False,
        ),
    )

    # always the same greeting
    await session.generate_reply(instructions=SESSION_GREETING)


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("AGENT_NAME", "rag-voice-agent"),
        )
    )



