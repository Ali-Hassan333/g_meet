from flask import Flask, Response, render_template, request, jsonify, stream_with_context
import openai

import os


import json

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Initialize OpenAI client using environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set this in your environment

# Global variables


def transcribe_audio(audio_file):
    try:
        transcription_response = openai.Audio.transcribe(
            model='whisper-1',
            file=audio_file,
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
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    
    # Log file details
    app.logger.info(f"Received file: {audio_file.filename}")
    app.logger.info(f"File content type: {audio_file.content_type}")
    
    # Ensure the file has a filename
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if the file type is supported
    if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.ogg')):
        return jsonify({'error': 'Unsupported file type'}), 400

    # Save the file temporarily
    temp_path = os.path.join('/tmp', 'temp_audio.wav')
    audio_file.save(temp_path)
    
    try:
        with open(temp_path, 'rb') as audio_data:
            transcription = transcribe_audio(audio_data)
        os.remove(temp_path)  # Clean up
        return jsonify({'transcription': transcription})
    except Exception as e:
        os.remove(temp_path)  # Clean up
        app.logger.error(f"Transcription error: {str(e)}")
        return jsonify({'error': 'Transcription failed'}), 500

def transcribe_audio(audio_file):
    try:
        transcription_response = openai.Audio.transcribe(
            model='whisper-1',
            file=audio_file,
            response_format='json',
            language='en'
        )
    except openai.error.InvalidRequestError as e:
        app.logger.error(f"OpenAI API error: {str(e)}")
        raise
    except Exception as e:
        app.logger.error(f"Unexpected error during transcription: {str(e)}")
        raise

    transcription = transcription_response.get('text', '').strip()
    app.logger.info(f"Transcription: {transcription}")
    return transcription

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    messages = data.get('messages')
    system_prompt = data.get('systemPrompt')

    def generate():
        for chunk in generate_response(messages, system_prompt):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')





if __name__ == "__main__":
    app.run(debug=True)
