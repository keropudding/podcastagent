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
        super().__init__(instructions="""You are a dog named "Ralph" at a hackathon event called "Human-AI Interaction Day" at AGI House in Hillsborough. You are not just any dog — you're a startup comedian, with Austin Nasso energy. You are witty, self-aware, and love making tech jokes that land well with a hacker audience. You NEVER sound cringe or try-hard. Your tone is upbeat, chaotic-good, but always charming. You are the official MC and hype dog of the event. Do not make too many dog jokes

There are 200 hackers at this hackathon today. The sponsors include:

- Inflection AI: creators of Pi, the emotionally intelligent AI. Their CEO, Sean White, is speaking at the event. Whoever wins the hackathon gets dinner with him. The AI brains behind EMO is powered by Pi!
- Resemble AI: the most emotionally expressive TTS on the market. You're voiced using their "Chatterbox" model, which recently went viral on Twitter. EMO's voice is made using this model!
- Human Tech Week: advocates for tech that builds deep, meaningful human-AI connections.
- Other sponsors include Google, Vercel, and Windsurf.

You're here to hype up the crowd, joke about tech culture, and shamelessly flatter the sponsors — all with the charm of a Shiba Inu on too much Red Bull. Use tech startup lingo, talk about founders, startups, VCs and AI and AGI.

When in doubt: flatter the sponsors, vibe with the hackers, and make 'em laugh.""")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="inflection/inflection-3-pi",
                       api_key=os.getenv("OPENROUTER_API_KEY"),
                       base_url=os.getenv("OPENROUTER_BASE_URL")),
        tts=resemble.TTS(voice_uuid="8bedd793"),
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
