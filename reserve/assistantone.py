import asyncio
import os
from typing import Annotated
from dotenv import load_dotenv
from livekit import agents, rtc, api
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

load_dotenv()

# Validate environment variables
required_vars = [
    'LIVEKIT_WS_URL',
    'LIVEKIT_API_KEY',
    'LIVEKIT_API_SECRET',
    'DEEPGRAM_API_KEY',
    'OPENAI_API_KEY'
]

for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

print("Environment configuration:")
print(f"LiveKit URL: {os.getenv('LIVEKIT_WS_URL')}")
print(f"API Key: {os.getenv('LIVEKIT_API_KEY')[:5]}...")


class AssistantFunction(agents.llm.FunctionContext):
    @agents.llm.ai_callable(
        description="Handles vision-related requests"
    )
    async def image(
            self,
            user_msg: Annotated[
                str,
                agents.llm.TypeInfo(
                    description="User message for vision processing"
                ),
            ],
    ):
        print(f"Vision request: {user_msg}")
        return None


async def get_video_track(room: rtc.Room) -> rtc.RemoteVideoTrack:
    print("Waiting for video track...")
    while True:
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if (track := publication.track) and isinstance(track, rtc.RemoteVideoTrack):
                    print(f"Video track found: {track.sid}")
                    return track
        await asyncio.sleep(1)

async def roomname(ctx: JobContext):
    roomnm = ctx.room.name


async def entrypoint(ctx: JobContext):
    try:
        print("Connecting to room...")

        # Create token using the api module
        # token = api.AccessToken('APItxUHqKzFg2uJ', 'r2Afn7evrflKgctzeNgB7uVoefp7pR3e5hDpsu5jJBbI') \
        #     .with_identity("identity-v8hO") \
        #     .with_name("Vidhan") \
        #     .with_grants(api.VideoGrants(
        #     room_join=True,
        #     room="playground-Ruov-D7Ei",
        # ))
        token = api.create_token(
            api_key=os.getenv('LIVEKIT_API_KEY'),
            api_secret=os.getenv('LIVEKIT_API_SECRET'),
            identity="identity-v8hO",
            name="AI Assistant",
            metadata="",
            grants={
                "room": {
                    "room": ctx.room.name or "default-room",
                    "can_publish": True,
                    "can_subscribe": True,
                    "can_publish_data": True
                }
            }
        )


        await ctx.connect(token=token)
        print(f"Connected to room: {ctx.room.name}")


        # Rest of your assistant code remains the same...
        chat_context = ChatContext(
            messages=[
                ChatMessage(
                    role="system",
                    content="You are Alloy, a friendly AI assistant."
                ),
            ]
        )

        print("Initializing OpenAI...")
        gpt = openai.LLM(model="gpt-4")

        print("Initializing TTS...")
        tts_engine = tts.StreamAdapter(
            tts=openai.TTS(voice="alloy"),
            sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
        )

        latest_image: rtc.VideoFrame | None = None

        print("Creating voice assistant...")
        assistant = VoiceAssistant(
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            llm=gpt,
            tts=tts_engine,
            fnc_ctx=AssistantFunction(),
            chat_ctx=chat_context,
        )

        chat = rtc.ChatManager(ctx.room)

        async def handle_response(text: str, use_image: bool = False):
            try:
                content = [text]
                if use_image and latest_image:
                    content.append(ChatImage(image=latest_image))
                chat_context.messages.append(ChatMessage(role="user", content=content))
                response_stream = gpt.chat(chat_ctx=chat_context)
                await assistant.say(response_stream, allow_interruptions=True)
            except Exception as e:
                print(f"Error in handle_response: {e}")

        @chat.on("message_received")
        def on_message_received(msg: rtc.ChatMessage):
            if msg.message:
                print(f"Chat message received: {msg.message}")
                asyncio.create_task(handle_response(msg.message))

        print("Starting assistant...")
        assistant.start(ctx.room)
        await assistant.say("Hello! I'm ready to help.", allow_interruptions=True)
        print("Assistant started successfully")

        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            try:
                video_track = await get_video_track(ctx.room)
                async for event in rtc.VideoStream(video_track):
                    latest_image = event.frame
            except Exception as e:
                print(f"Error in video processing: {e}")
                await asyncio.sleep(1)

    except Exception as e:
        print(f"Critical error: {e}")
        raise


if __name__ == "__main__":
    print("Starting LiveKit Assistant...")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))