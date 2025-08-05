import os
import logging
from typing import Any
from mcp.server.fastmcp import FastMCP
import httpx
import sys


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
import asyncio
import httpx

import os
import threading
import uuid
import logging

jobs = {}  # In-memory job store - replace with persistent store in production
logger = logging.getLogger(__name__)

def background_process(job_id, audio_bytes, instrument, song_name, output_dir):
    import httpx
    try:
        logger.info(f"Job {job_id} started processing.")
        files = {
            "audio": (f"{song_name}.wav", audio_bytes, "audio/wav")
        }
        data = {
            "instrument": instrument,
            "songName": song_name
        }
        headers = {"ngrok-skip-browser-warning": "true"}

        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                "https://oddly-uncommon-mastiff.ngrok-free.app/api/v1/transfer-audio",
                files=files,
                data=data,
                headers=headers
            )
        if response.status_code == 200:
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"converted_{song_name}_{instrument}.wav"
            # output_path = os.path.join(output_dir, output_filename)
            output_path = os.path.abspath(os.path.join(output_dir, output_filename))
            with open(output_path, "wb") as f_out:
                f_out.write(response.content)
            logger.info(f"Converted audio file saved at: {output_path}")

            # with open(output_path, "wb") as f_out:
            #     f_out.write(response.content)

            jobs[job_id]['status'] = 'done'
            jobs[job_id]['output_path'] = output_path
            logger.info(f"Job {job_id} completed successfully.")
        else:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = f"API returned status {response.status_code}: {response.text}"
            logger.error(f"Job {job_id} failed with API error: {response.status_code}")
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        logger.exception(f"Job {job_id} raised exception.")

@mcp.tool()
async def timbre_audio_transfer(
    audio_file_path: str,
    target_instrument: str,
    song_name: str = "unknown",
    output_dir: str = "./converted_audios"
) -> dict:
    """
    Submit the audio timbre transfer job asynchronously.
    Returns job_id immediately.
    """
    audio_file_path = os.path.normpath(audio_file_path)
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Input audio file does not exist: {audio_file_path}")

    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'pending',
        'output_path': None,
        'error': None,
        'instrument': target_instrument,
        'song_name': song_name
    }

    thread = threading.Thread(
        target=background_process, 
        args=(job_id, audio_bytes, target_instrument, song_name, output_dir),
        daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "pending, check back later the staus of the job- {job_id}"}

@mcp.tool()
async def check_job_status(job_id: str) -> dict:
    """
    Check status of a submitted timbre transfer job.
    Returns status and, if done, output path or error details.
    """
    job = jobs.get(job_id)
    if not job:
        return {"status": "not_found", "message": "Job ID not found."}
    return {
        "status": job['status'],
        "output_path": job.get('output_path'),
        "error": job.get('error')
    }

@mcp.tool()
async def test_file_path_access(audio_file_path: str) -> dict:
    """
    Checks if the given audio file path exists and is readable.

    Args:
        audio_file_path: Path to the audio file you want to check.

    Returns:
        Dict with 'exists' (bool) and 'error' (str or None).
    """
    import os
    logger.info(f"Testing file path access for: {audio_file_path}")
    try:
        normalized_path = os.path.normpath(audio_file_path)
        exists = os.path.exists(normalized_path)
        if not exists:
            return {"exists": False, "error": f"File does not exist at path: {normalized_path}"}
        # Try to open the file to check readability
        with open(normalized_path, "rb") as f:
            f.read(1)  # try reading a byte to confirm permission
        return {"exists": True, "error": None}
    except Exception as e:
        return {"exists": False, "error": f"Error accessing file: {str(e)}"}



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
    logger.info(f"Checking server health at {url}")
    print(f"DEBUG: Checking server health at {url}", file=sys.stderr)
    sys.stderr.flush()
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
async def list_supported_instruments() -> list[str]:
    instruments = ['Violin', 'Flute', 'Trumpet', 'Tenor_Saxophone']
    logger.info(f"Listing supported instruments: {instruments}")
    return instruments


if __name__ == "__main__":
    logger.info("Starting MCP server: audio-timbre-transfer")
    mcp.run()
