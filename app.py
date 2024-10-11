from flask import Flask, Response, render_template, request, jsonify, stream_with_context
import openai
import wave
import tempfile
import os
import numpy as np
import sounddevice as sd
import threading
import queue
import time
import json
import webrtcvad
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Initialize OpenAI client using environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set this in your environment

# Global variables
audio_queue = queue.Queue()
is_recording = False
sample_rate = 16000

# Voice Activity Detection
vad = webrtcvad.Vad(3)

def is_valid_transcription(transcription):
    min_length = 5
    max_length = 200
    banned_phrases = [
        'um', 'uh', 'ah', 'oh',
        'thanks for watching', 'thank you for watching', 'background noise'
    ]

    if len(transcription) < min_length or len(transcription) > max_length:
        return False

    for phrase in banned_phrases:
        if phrase in transcription:
            return False

    return True

def record_audio():
    global is_recording
    is_recording = True

    frame_duration_ms = 30
    frame_size = int(sample_rate * frame_duration_ms / 1000)

    speech_frames = []
    silence_frames = 0
    max_silence_frames = int(0.5 * 1000 / frame_duration_ms)  # 0.5 seconds of silence

    with sd.RawInputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        print("Recording started...")
        while is_recording:
            audio_frame = stream.read(frame_size)[0]
            if len(audio_frame) == 0:
                continue

            is_speech = vad.is_speech(audio_frame, sample_rate)

            if is_speech:
                speech_frames.append(audio_frame)
                silence_frames = 0
            else:
                if speech_frames:
                    silence_frames += 1
                    if silence_frames > max_silence_frames:
                        audio_data = b''.join(speech_frames)
                        audio_queue.put(audio_data)
                        speech_frames = []
                        silence_frames = 0
                else:
                    pass

            time.sleep(frame_duration_ms / 1000.0)

        if speech_frames:
            audio_data = b''.join(speech_frames)
            audio_queue.put(audio_data)

def transcribe_audio(filename):
    with open(filename, 'rb') as audio_file:
        try:
            transcription_response = openai.Audio.transcribe(
                model='whisper-1',
                file=audio_file,  # Pass the file object directly
                response_format='json',
                language='en'
            )
        except Exception as e:
            logging.error("Error during transcription", exc_info=True)
            return ""

    transcription = transcription_response.get('text', '').strip()
    print(transcription)
    return transcription

def generate_response(messages, system_prompt):
    try:
        # Prepare the conversation messages
        conversation = [{"role": "assistant", 'content': system_prompt },] + messages

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=conversation,
            stream=True,
        )

        for chunk in response:
            if 'choices' in chunk:
                if 'delta' in chunk.choices[0]:
                    delta = chunk.choices[0].delta
                    if 'content' in delta:
                        yield delta.content
    except Exception as e:
        logging.error("Error in generate_response", exc_info=True)
        yield "Sorry, there was an error generating the response."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    messages = data.get('messages')
    system_prompt = data.get('systemPrompt')

    def generate():
        for chunk in generate_response(messages, system_prompt):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/start_recording', methods=['POST'])
def start_recording_route():
    global is_recording
    if not is_recording:
        threading.Thread(target=record_audio, daemon=True).start()
        return jsonify({'status': 'Recording started'})
    return jsonify({'status': 'Already recording'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    is_recording = False
    return jsonify({'status': 'Recording stopped'})

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
                    with wave.open(temp_audio.name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit audio
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_data)

                    transcription = transcribe_audio(temp_audio.name)

                if transcription:
                    cleaned_transcription = transcription.strip().lower()
                    if is_valid_transcription(cleaned_transcription):
                        yield f"data: {json.dumps({'transcription': transcription})}\n\n"
                    else:
                        print("Transcription discarded due to common phrase or short length.")
            else:
                time.sleep(0.1)

    return Response(stream_with_context(event_stream()), content_type='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True)
