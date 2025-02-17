# check_version.py
import importlib.metadata

try:
    livekit_version = importlib.metadata.version('livekit')
    print(f"LiveKit version: {livekit_version}")
except importlib.metadata.PackageNotFoundError:
    print("LiveKit is not installed")

try:
    livekit_agents_version = importlib.metadata.version('livekit-agents')
    print(f"LiveKit agents version: {livekit_agents_version}")
except importlib.metadata.PackageNotFoundError:
    print("LiveKit agents is not installed")
