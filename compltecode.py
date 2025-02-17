import asyncio
import os
from typing import Annotated

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import ChatContext, ChatImage, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero
from livekit.jwt import AccessToken, VideoGrant
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LiveKit Credentials
LIVEKIT_WS_URL = os.getenv("LIVEKIT_WS_URL", "wss://sinusproject-7m7ej8j7.livekit.cloud")  # Use LiveKit Cloud Playground
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Check if credentials are set
if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
    raise ValueError("Missing LIVEKIT_API_KEY or LIVEKIT_API_SECRET in environment variables.")

# Generate Access Token for LiveKit Agent
def generate_livekit_token():
    grant = VideoGrant(room_join=True, room="test-room")  # Replace with actual room name
    token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, grant=grant, identity="agent")
    return token.to_jwt()


class AssistantFunction(agents.llm.FunctionContext):
    """Defines functions that will be called by the assistant."""

    @agents.llm.ai_callable(
        description="Called when asked to analyze an image, video, or webcam feed."
    )
    async def image(self, user_msg: Annotated[str, agents.llm.TypeInfo(description="User's message triggering vision")]):
        print(f"🔍 Vision Triggered: {user_msg}")
        return None


async def get_video_track(room: rtc.Room):
    """Fetch the first available remote video track from the room."""
    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if isinstance(track_publication.track, rtc.RemoteVideoTrack):
                print(f"🎥 Using video track {track_publication.track.sid}")
                return track_publication.track
    return None


async def entrypoint(ctx: JobContext):
    """Main function to connect to LiveKit and handle assistant logic."""

    await ctx.connect()
    print(f"✅ Connected to room: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="Your name is Alloy. You are a funny, witty AI with voice and vision. "
                        "Keep responses short and avoid emojis or complex punctuation."
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o")

    # Wrap OpenAI TTS with a Stream Adapter
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: rtc.VideoFrame | None = None

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=openai_tts,
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False):
        """Generate a response and optionally include the latest image."""
        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))
        response_stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(response_stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """Handle incoming chat messages."""
        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """Trigger when function calls complete."""
        if called_functions:
            user_msg = called_functions[0].call_info.arguments.get("user_msg")
            if user_msg:
                asyncio.create_task(_answer(user_msg, use_image=True))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hi there! How can I help?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)
        if video_track:
            async for event in rtc.VideoStream(video_track):
                latest_image = event.frame


async def connect_to_livekit():
    """Initialize and connect to LiveKit with a token."""
    token = generate_livekit_token()
    room = rtc.Room()
    options = rtc.RoomOptions(auto_subscribe=True)

    try:
        await room.connect(LIVEKIT_WS_URL, token=token, options=options)
        print("🎉 Successfully connected to LiveKit!")
        return room
    except Exception as e:
        print(f"❌ Failed to connect to LiveKit: {e}")
        return None


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
