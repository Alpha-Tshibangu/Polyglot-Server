# server.py

import os
import requests
import replicate
from pydub import AudioSegment
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import re
import time
from pathlib import Path
from BunnyCDN.Storage import Storage
import asyncio
import socketio 
import uvicorn   

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS (Adjust 'allow_origins' as per your frontend's domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

# Wrap the FastAPI app with Socket.IO ASGI app
sio_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Global dictionary to track users
users = {}
rooms = {}

# Set Replicate API token from environment
replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
if not replicate_api_token:
    raise ValueError("Replicate API token not found in .env file")
# Initialize the async replicate client
replicate_client = replicate.Client(api_token=replicate_api_token)

# BunnyCDN credentials from environment
bunnycdn_api_key = os.getenv("BUNNYCDN_STORAGE_API_KEY")
bunnycdn_storage_zone = os.getenv("BUNNYCDN_STORAGE_ZONE")
bunnycdn_storage_region = os.getenv("BUNNYCDN_STORAGE_REGION")

if not all([bunnycdn_api_key, bunnycdn_storage_zone, bunnycdn_storage_region]):
    raise ValueError("BunnyCDN credentials not found in .env file")

# Debug: Confirm storage zone is loaded correctly
print(f"Using BunnyCDN Storage Zone: {bunnycdn_storage_zone}")
print(f"Using BunnyCDN Storage Region: {bunnycdn_storage_region}")

# Define the audio directory path
AUDIO_DIR = Path.cwd() / "audio"

# Create the audio directory if it doesn't exist
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
print(f"Audio directory is set to: {AUDIO_DIR.resolve()}")

async def generate_unique_id():
    # Implement a function to generate a unique ID (similar to the haiku function in the example)
    # For simplicity, you can use uuid or any other method
    from uuid import uuid4
    return str(uuid4())

# Function to sanitize filenames
def sanitize_filename(filename):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', filename)

# Function to validate and convert audio files to WAV
def convert_to_wav(input_path, output_path):
    try:
        # Detect format based on file extension
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext in ['.webm', '.ogg', '.wav', '.mp3', '.m4a']:
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
            print(f"Converted {input_path} to WAV format at {output_path}")
            return True
        else:
            print(f"Unsupported file format: {file_ext}")
            return False
    except Exception as e:
        print(f"Error during audio conversion: {e}")
        return False

# Initialize BunnyCDN storage
storage = Storage(bunnycdn_api_key, bunnycdn_storage_zone, bunnycdn_storage_region)

# Upload audio to BunnyCDN and return the public URL
def upload_to_bunnycdn(file_name, local_file_path):
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"{local_file_path} does not exist.")

    # Sanitize the file name
    sanitized_file_name = sanitize_filename(file_name)

    # Construct the upload URL using the storage zone from environment variables
    url = f"https://storage.bunnycdn.com/alpha-storage/{sanitized_file_name}"
    headers = {
        'AccessKey': bunnycdn_api_key,
        'Content-Type': 'application/octet-stream',
        'accept': 'application/json',
    }

    # Debug: Confirm upload details (Do not log AccessKey in production)
    print(f"Uploading to URL: {url}")
    file_size = os.path.getsize(local_file_path)
    print(f"Uploading file: {sanitized_file_name} (Size: {file_size} bytes)")

    try:
        # Upload the file
        with open(local_file_path, 'rb') as file:
            response = requests.put(url, headers=headers, data=file, timeout=60)

        # Debug: Print response status and body
        print(f"Upload Response Status: {response.status_code}")
        print(f"Upload Response Body: {response.text}")

        # Check if upload was successful
        if response.status_code in [200, 201]:
            print("Upload successful.")
        else:
            print(f"Upload failed with status code {response.status_code}: {response.text}")
            raise Exception(f"Upload failed: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request exception during upload: {e}")
        raise Exception(f"Upload failed: {str(e)}")

    # Return the public URL of the uploaded audio using the storage zone
    public_url = f"https://alpha-storage.b-cdn.net/{sanitized_file_name}"
    print(f"Public URL: {public_url}")
    return public_url, sanitized_file_name  # Return the filename as well

# Function to delete a file from BunnyCDN
def delete_from_bunnycdn(file_name):
    url = f"https://storage.bunnycdn.com/alpha-storage/{file_name}"
    headers = {
        'AccessKey': bunnycdn_api_key,
    }

    try:
        response = requests.delete(url, headers=headers, timeout=60)
        # Check if deletion was successful
        if response.status_code == 200:
            print(f"File {file_name} deleted successfully from BunnyCDN.")
        else:
            print(f"Delete failed with status code {response.status_code}: {response.text}")
            raise Exception(f"Delete failed: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request exception during delete: {e}")
        raise Exception(f"Delete failed: {str(e)}")

# Function to download translated audio
def download_translated_audio(translated_audio_url, output_file):
    print(f"Downloading translated audio from: {translated_audio_url}")
    try:
        response = requests.get(translated_audio_url, stream=True, timeout=60)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            print(f"Translated audio downloaded as {output_file}")
            return True
        else:
            print(f"Failed to download the translated audio: {response.status_code} {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Request exception during download: {e}")
        return False

# Language mapping dictionary
language_map = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
    # Add more mappings as needed
}

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print('Client connected:', sid)
    await sio.emit('me', sid, to=sid)

@sio.event
async def connect(sid, environ):
    print('Client connected:', sid)

@sio.event
async def init(sid):
    user_id = await generate_unique_id()
    users[user_id] = sid
    await sio.emit('init', {'id': user_id}, to=sid)

@sio.event
async def request(sid, data):
    to_sid = users.get(data['to'])
    if to_sid:
        from_user = next((user for user, s in users.items() if s == sid), None)
        await sio.emit('request', {'from': from_user}, to=to_sid)

@sio.event
async def call(sid, data):
    to_sid = users.get(data['to'])
    if to_sid:
        from_user = next((user for user, s in users.items() if s == sid), None)
        await sio.emit('call', {**data, 'from': from_user}, to=to_sid)
    else:
        await sio.emit('failed', to=sid)

@sio.event
async def end(sid, data):
    to_sid = users.get(data['to'])
    if to_sid:
        await sio.emit('end', to=to_sid)

@sio.event
async def joinRoom(sid, data):
    room_id = data['roomId']
    name = data['name']
    print(f'{name} is joining room {room_id}')
    await sio.enter_room(sid, room_id)
    if room_id not in rooms:
        rooms[room_id] = set()
    rooms[room_id].add(sid)
    users[sid] = {'name': name, 'room': room_id}
    
    # Notify the user that they've joined the room
    await sio.emit('roomJoined', {'roomId': room_id}, room=sid)
    
    # Notify other users in the room
    for user_sid in rooms[room_id]:
        if user_sid != sid:
            await sio.emit('user-connected', sid, room=user_sid)

@sio.event
async def sending_signal(sid, data):
    print(f"Sending signal from {sid} to {data['userToSignal']}")
    await sio.emit('user-joined', {'signal': data['signal'], 'callerID': data['callerID']}, room=data['userToSignal'])

@sio.event
async def returning_signal(sid, data):
    print(f"Returning signal from {sid} to {data['callerID']}")
    await sio.emit('receiving-returned-signal', {'signal': data['signal'], 'id': sid}, room=data['callerID'])

@sio.event
async def disconnect(sid):
    if sid in users:
        room_id = users[sid]['room']
        name = users[sid]['name']
        if room_id in rooms:
            rooms[room_id].remove(sid)
            print(f'{name} disconnected from room {room_id}')
            if len(rooms[room_id]) == 0:
                del rooms[room_id]
                print(f'Room {room_id} is now empty and has been deleted')
            else:
                await sio.emit('user-disconnected', sid, room=room_id)
        del users[sid]
    print('Client disconnected:', sid)

# Endpoint to translate audio
@app.post("/translate-audio")
async def translate_audio_endpoint(
    file: UploadFile = File(...),
    source_lang: str = "English",
    target_lang: str = "French"
):

    # Log the content type
    print(f"Uploaded file content type: {file.content_type}")

    # Define supported languages (extend as needed)
    supported_languages = ["English", "Spanish", "French", "German", "Chinese", "Japanese"]
    print(f"Received request with source_lang='{source_lang}', target_lang='{target_lang}'")

    # Validate languages
    if source_lang not in supported_languages or target_lang not in supported_languages:
        print("Unsupported language detected.")
        raise HTTPException(status_code=400, detail="Unsupported language.")

    # Initialize temporary file paths and BunnyCDN filenames
    temp_input_file = None
    temp_wav_file = None
    temp_output_file = None
    input_bunnycdn_filename = None
    output_bunnycdn_filename = None

    try:
        # Save uploaded file to the audio directory with a unique filename
        original_filename = file.filename
        file_base, file_ext = os.path.splitext(original_filename)
        unique_suffix = int(time.time())
        unique_file_name = f"{sanitize_filename(file_base)}_{unique_suffix}{file_ext}"
        save_path = AUDIO_DIR / unique_file_name

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        temp_input_file = save_path
        print(f"Saved uploaded file to: {temp_input_file}")

        # Convert the uploaded audio to WAV format
        temp_wav_file = AUDIO_DIR / f"{file_base}_{unique_suffix}.wav"
        conversion_success = convert_to_wav(temp_input_file, temp_wav_file)
        if not conversion_success:
            print("Audio conversion to WAV failed.")
            raise HTTPException(status_code=400, detail="Invalid or unsupported audio format.")

        # Upload the WAV audio to BunnyCDN
        audio_url, input_bunnycdn_filename = upload_to_bunnycdn(temp_wav_file.name, temp_wav_file)
        print(f"Audio uploaded to BunnyCDN: {audio_url}")

        # Define the Replicate model
        model = "adirik/seamless-expressive:fe1ce551597dee59a90f1fb418747c81214177f28c4e8728df96b06d2a2a6093"

        # Prepare input parameters
        input_params = {
            "audio_input": audio_url,
            "source_lang": source_lang,
            "target_lang": target_lang
        }

        # Send the audio URL to Replicate for translation
        print("Sending audio for translation to Replicate...")
        try:
            output = replicate_client.run(model, input=input_params)
            print(f"Translation completed. Output URL: {output['audio_out']}")
        except Exception as e:
            print(f"Error during Replicate translation: {e}")
            raise HTTPException(status_code=500, detail=f"Translation service failed: {str(e)}")

        # Download the translated audio
        translated_audio_url = output['audio_out']
        translated_file_name = f"translated_{unique_file_name}"
        temp_output_file = AUDIO_DIR / translated_file_name
        download_success = download_translated_audio(translated_audio_url, temp_output_file)
        if not download_success:
            print("Failed to download translated audio.")
            raise HTTPException(status_code=500, detail="Failed to download translated audio.")

        # Upload the translated audio to BunnyCDN to obtain a public URL
        translated_audio_public_url, output_bunnycdn_filename = upload_to_bunnycdn(temp_output_file.name, temp_output_file)
        print(f"Translated audio re-uploaded to BunnyCDN: {translated_audio_public_url}")

        # Return the translated audio public URL as a JSON response
        return JSONResponse(content={"translated_audio_url": translated_audio_public_url})

    except HTTPException as he:
        print(f"HTTPException occurred: {he.detail}")
        raise he
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary files
        if temp_input_file and temp_input_file.exists():
            temp_input_file.unlink()
            print(f"Deleted temporary input file: {temp_input_file}")
        if temp_wav_file and temp_wav_file.exists():
            temp_wav_file.unlink()
            print(f"Deleted temporary WAV file: {temp_wav_file}")
        if temp_output_file and temp_output_file.exists():
            temp_output_file.unlink()
            print(f"Deleted temporary output file: {temp_output_file}")

        # Delete the audio files from BunnyCDN
        if input_bunnycdn_filename:
            try:
                delete_from_bunnycdn(input_bunnycdn_filename)
            except Exception as e:
                print(f"Error deleting input file from BunnyCDN: {e}")
        if output_bunnycdn_filename:
            try:
                pass
                # Uncomment if you want to delete the translated file from BunnyCDN
                # delete_from_bunnycdn(output_bunnycdn_filename)
            except Exception as e:
                print(f"Error deleting output file from BunnyCDN: {e}")

# Endpoint for text translation
@app.post("/api/translate")
async def translate_text_endpoint(request: Request):
    try:
        data = await request.json()
        text = data.get('text')
        source_language = data.get('sourceLanguage')
        target_language = data.get('targetLanguage')

        if not all([text, source_language, target_language]):
            raise HTTPException(status_code=400, detail="Missing required fields: 'text', 'sourceLanguage', 'targetLanguage'.")

        # Map language names to the exact format expected by the model
        source_language_name = language_map.get(source_language.lower(), source_language)
        target_language_name = language_map.get(target_language.lower(), target_language)

        # Define the Replicate model for text translation
        model = "cjwbw/seamless_communication:668a4fec05a887143e5fe8d45df25ec4c794dd43169b9a11562309b2d45873b0"

        # Prepare input parameters
        input_params = {
            "task_name": "T2TT (Text to Text translation)",
            "input_text": text,
            "input_text_language": source_language_name,
            "max_input_audio_length": 60,
            "target_language_text_only": target_language_name
        }

        # Run the Replicate model
        output = replicate_client.run(model, input=input_params)

        # Return the translated text
        return JSONResponse(content={"translatedText": output['text_output']})

    except Exception as e:
        print(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(sio_app, host='0.0.0.0', port=5001)
