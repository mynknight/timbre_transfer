from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import os
import time
import uvicorn
from pyngrok import ngrok

from music_conversion import MusicConversion     # Your timbre transfer class
from vocal_extraction import VocalExtraction     # If you want vocal isolation

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is healthy and running"}

@app.post("/api/v1/transfer-audio")
async def transfer_audio(
    instrument: str = Form(...),
    audio: UploadFile = File(...),
    songName: str = Form('unknown')
):
    upload_dir = "uploads"

    
    # os.makedirs(upload_dir, exist_ok=True)
    # timestamp = str(int(time.time()))
    # raw_path = os.path.join(upload_dir, f"{timestamp}_{audio.filename}")
    # out_path = raw_path.replace('.wav', f'_styled_{instrument}.wav')

    # # Save uploaded audio file
    # with open(raw_path, "wb") as f:
    #     f.write(await audio.read())

    # # (Optional) Extract vocals first
    # vocal_extractor = VocalExtraction(raw_path, songName)
    # vocal_extractor.extract_vocal()
    # input_path = vocal_extractor.destination_path  # Or just use raw_path if not needed

    # # Run timbre transfer
    # music_converter = MusicConversion(songName, instrument, input_path)
    # processed_path = music_converter.load_song_and_extract_features()

    os.makedirs(upload_dir, exist_ok=True)
    timestamp = str(int(time.time()))
    raw_path = os.path.join(upload_dir, f"{timestamp}_{audio.filename}")
    out_path = raw_path.replace('.wav', f'_styled_{instrument}.wav')

    # Save uploaded audio file
    with open(raw_path, "wb") as f:
        f.write(await audio.read())

    # Skip the vocal extraction; use raw_path as input directly
    input_path = raw_path

    # Run timbre transfer
    music_converter = MusicConversion(songName, instrument, input_path)
    processed_path = music_converter.load_song_and_extract_features()

    return FileResponse(processed_path, media_type="audio/wav")

if __name__ == "__main__":
    port = 8000
    host = "0.0.0.0"

    # Open ngrok tunnel
    # public_url = ngrok.connect(port, bind_tls=True, subdomain="my-ddsp-app").public_url
    public_url = ngrok.connect(port).public_url
    print(f" * ngrok tunnel URL: {public_url}")

    # Start FastAPI server with uvicorn
    uvicorn.run("main:app", host=host, port=port, log_level="info")
