import os
import logging
from typing import Any
from mcp.server.fastmcp import FastMCP
import httpx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ===== ADD THIS LOGGING CONFIGURATION HERE =====
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("mcp_server.log"),   # Logs go to this file
        logging.StreamHandler()                   # Also output logs in console/terminal
    ]
)
logger = logging.getLogger(__name__)
# ===============================================
# Initialize MCP server
mcp = FastMCP("audio-timbre-transfer")

@mcp.tool()
async def check_server_health(
    url: str = "http://oddly-uncommon-mastiff.ngrok-free.app/health"
) -> dict:
    """
    Checks the health endpoint of the server.

    Args:
        url: The health check endpoint URL. (Defaults to your FastAPI /health endpoint.)

    Returns:
        A dictionary with the HTTP status code and health response, or error details.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers={"ngrok-skip-browser-warning": "true"}, timeout=10.0)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.headers.get("Content-Type", "").startswith("application/json") else response.text
            }
    except Exception as e:
        return {
            "status_code": None,
            "error": str(e)
        }
    
@mcp.tool()
async def transfer_audio_timbre(
    audio_file_path: str,        # Path to the input raw audio file (.wav, .m4a, etc.)
    target_instrument: str,      # Target instrument (e.g., 'Violin', 'Flute', etc.)
    song_name: str = "unknown",  # Optional song name for naming output
    output_dir: str = "./converted_audios"  # Directory to save converted audio output
) -> str:
    """
    Convert audio to a different instrument timbre using DDSP.
    """
    try:
        audio_file_path = os.path.normpath(audio_file_path)
        logger.info(f"Received transfer_audio_timbre request: audio_file_path={audio_file_path}, "
                    f"target_instrument={target_instrument}, song_name={song_name}, output_dir={output_dir}")

        if not os.path.exists(audio_file_path):
            logger.error(f"Input audio file does not exist: {audio_file_path}")
            raise FileNotFoundError(f"Input audio file does not exist: {audio_file_path}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Read input audio file bytes and prepare multipart/form-data
        with open(audio_file_path, "rb") as audio_file:
            files = {
                "audio": (os.path.basename(audio_file_path), audio_file, "audio/wav")
            }
            data = {
                "instrument": target_instrument,
                "songName": song_name
            }
            headers = {
                "ngrok-skip-browser-warning": "true"
            }

            logger.info(f"Sending request to timbre transfer API for instrument: {target_instrument}")
            async with httpx.AsyncClient() as client:
                # Replace with your actual public ngrok tunnel URL
                response = await client.post(
                    "http://oddly-uncommon-mastiff.ngrok-free.app/api/v1/transfer-audio",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=300.0  # 5-minute timeout for audio processing
                )

            logger.info(f"Received response with status code: {response.status_code}")

        if response.status_code == 200:
            output_filename = f"converted_{song_name}_{target_instrument}.wav"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, "wb") as f_out:
                f_out.write(response.content)
            logger.info(f"Saved converted audio file to: {output_path}")
            return output_path
        else:
            logger.error(f"API call failed with status {response.status_code}: {response.text}")
            raise Exception(f"API call failed with status {response.status_code}: {response.text}")

    except Exception as e:
        logger.exception(f"Exception in transfer_audio_timbre: {e}")
        raise


@mcp.tool()
async def list_supported_instruments() -> list[str]:
    instruments = ['Violin', 'Flute', 'Trumpet', 'Tenor_Saxophone']
    logger.info(f"Listing supported instruments: {instruments}")
    return instruments


if __name__ == "__main__":
    logger.info("Starting MCP server: audio-timbre-transfer")
    mcp.run()
