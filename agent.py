from dotenv import load_dotenv
import os

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    resemble,
    deepgram,
    noise_cancellation,
    silero,
    bithuman,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="inflection/inflection-3-pi",
                       api_key=os.getenv("OPENROUTER_API_KEY"),
                       base_url=os.getenv("OPENROUTER_BASE_URL")),
        tts=resemble.TTS(voice_uuid="55592656"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    # Initialize and start the avatar
    avatar = bithuman.AvatarSession(
        model_path=os.getenv("BITHUMAN_MODEL_PATH"),
        api_secret=os.getenv("BITHUMAN_API_SECRET"),
    )
    await avatar.start(session, room=ctx.room)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
