import os
import time
import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt
import io
import base64
import cv2
from PIL import Image
import soundfile as sf
import random
import json
from transformers import pipeline
import traceback
from gtts import gTTS  # Import Google Text-to-Speech
import tempfile  # For creating temporary files

# Initialize the models with better error handling
try:
    print("Loading emotion recognition model...")
    emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection", top_k=7)
    print("Emotion model loaded successfully")
except Exception as e:
    print(f"Error loading emotion model: {str(e)}")
    # Fallback to a simpler model or dummy function
    def emotion_classifier(image):
        return [{"label": "neutral", "score": 1.0}]

try:
    print("Loading story generation model...")
    story_generator = pipeline("text-generation", model="gpt2")
    print("Story model loaded successfully")
except Exception as e:
    print(f"Error loading story model: {str(e)}")
    # Fallback function
    def story_generator(prompt, max_length=200, do_sample=True, top_k=50):
        return [{"generated_text": f"{prompt} This is a simple story because the model couldn't be loaded."}]

# Skip audio model loading if causing issues
audio_classifier = None
print("Audio model skipped to improve stability")

class EmotionDrivenComposer:
    def __init__(self):
        # Load emotion mapping
        self.emotion_mapping = {
            "happy": {"tempo": 140, "scale": "major", "intensity": 0.9, "color": "yellow", "speech_rate": "slow", "pitch": "high"},
            "sad": {"tempo": 65, "scale": "minor", "intensity": 0.4, "color": "blue", "speech_rate": "slow", "pitch": "low"},
            "angry": {"tempo": 180, "scale": "minor", "intensity": 0.9, "color": "red", "speech_rate": "fast", "pitch": "high"},
            "surprised": {"tempo": 130, "scale": "lydian", "intensity": 0.8, "color": "green", "speech_rate": "fast", "pitch": "high"},
            "fearful": {"tempo": 160, "scale": "diminished", "intensity": 0.8, "color": "purple", "speech_rate": "fast", "pitch": "medium"},
            "neutral": {"tempo": 100, "scale": "major", "intensity": 0.5, "color": "gray", "speech_rate": "medium", "pitch": "medium"}
        }

        # Pre-load face cascade
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                print("Warning: Face cascade classifier could not be loaded!")
                # Fallback to a direct file path that might work
                alt_path = '/usr/local/lib/python3.x/dist-packages/cv2/data/haarcascade_frontalface_default.xml'
                if os.path.exists(alt_path):
                    self.face_cascade = cv2.CascadeClassifier(alt_path)
        except Exception as e:
            print(f"Error loading face cascade: {str(e)}")
            self.face_cascade = None

    def analyze_face(self, image):
        """Analyze an image to detect emotion with robust error handling"""
        if image is None:
            print("Error: No image provided")
            return {"error": "No image provided", "primary_emotion": "neutral"}

        try:
            # Print image info for debugging
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")

            # Ensure image is in the right format
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # Already RGB
                pass  # No conversion needed
            else:
                print(f"Unusual image format with shape {image.shape}")
                # Force conversion to RGB
                image = Image.fromarray(image).convert('RGB')
                image = np.array(image)

            # Make a copy to avoid modifying the original
            debug_image = image.copy()

            # Write debug image to disk for inspection
            cv2.imwrite('debug_input.jpg', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

            # Detect face using OpenCV
            if self.face_cascade is None:
                print("Face cascade not available, using full image")
                face_img = image
                primary_emotion = "neutral"  # Default

                # Try to run emotion detection on the full image
                try:
                    face_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    results = emotion_classifier(face_pil)

                    # Extract emotions
                    emotions = {}
                    for result in results:
                        emotions[result['label']] = result['score']

                    # Find primary emotion if we have results
                    if emotions:
                        primary_emotion = max(emotions, key=emotions.get)

                except Exception as e:
                    print(f"Error in emotion detection: {str(e)}")
                    emotions = {"neutral": 1.0}
                    primary_emotion = "neutral"

                return {
                    "primary_emotion": primary_emotion,
                    "emotions": emotions,
                    "face_image": image,  # Use full image
                    "full_image": image
                }

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

            if len(faces) == 0:
                print("No face detected, using full image")
                # If no face detected, use the full image
                face_img = image

                # Try emotion detection on full image
                face_pil = Image.fromarray(image)
                results = emotion_classifier(face_pil)

                # Extract emotions
                emotions = {}
                for result in results:
                    emotions[result['label']] = result['score']

                # Find primary emotion
                primary_emotion = max(emotions, key=emotions.get)
            else:
                # Extract the largest face
                x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
                face_img = image[y:y+h, x:x+w]

                # Draw rectangle for debugging
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imwrite('debug_face_detected.jpg', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

                # Analyze emotion using the model
                print("Converting face to PIL image")
                face_pil = Image.fromarray(face_img)
                print("Running emotion classifier")
                results = emotion_classifier(face_pil)
                print(f"Emotion results: {results}")

                # Extract emotions
                emotions = {}
                for result in results:
                    emotions[result['label']] = result['score']

                # Find primary emotion
                primary_emotion = max(emotions, key=emotions.get)
                print(f"Primary emotion: {primary_emotion}")

            return {
                "primary_emotion": primary_emotion,
                "emotions": emotions,
                "face_image": face_img,
                "full_image": image
            }
        except Exception as e:
            print(f"Error in analyze_face: {str(e)}")
            print(traceback.format_exc())
            # Return a default response that won't break downstream processing
            return {
                "primary_emotion": "neutral",
                "emotions": {"neutral": 1.0},
                "face_image": image if 'image' in locals() else None,
                "full_image": image if 'image' in locals() else None,
                "error": f"Error analyzing face: {str(e)}"
            }

    def generate_music(self, emotion_data):
        """Generate music based on detected emotion"""
        try:
            # Map emotion to musical parameters
            if isinstance(emotion_data, dict) and "error" in emotion_data:
                primary_emotion = "neutral"  # Default to neutral for errors
            else:
                primary_emotion = emotion_data["primary_emotion"].lower()

            print(f"Generating music for emotion: {primary_emotion}")

            # Get musical parameters
            mapping = self.emotion_mapping.get(primary_emotion, self.emotion_mapping["neutral"])
            tempo = mapping["tempo"]
            scale_type = mapping["scale"]
            intensity = mapping["intensity"]

            # Generate simple melody based on emotion
            notes = self._generate_melody_for_emotion(scale_type)

            # Create MIDI sequence
            sequence = self._create_midi_sequence(notes, tempo, intensity)

            # Convert to audio
            audio = self._sequence_to_audio(sequence)

            # Add audio normalization for consistent volume
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.9

            # Set sample rate
            sample_rate = 44100

            return (sample_rate, audio)
        except Exception as e:
            print(f"Error in generate_music: {str(e)}")
            print(traceback.format_exc())
            # Return a simple sine wave as fallback
            sample_rate = 44100
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, sample_rate * 2))
            return (sample_rate, audio)

    def _generate_melody_for_emotion(self, scale_type):
        """Generate a melody based on the emotional scale type"""
        base_notes = {"major": [60, 64, 67, 72],
                     "minor": [60, 63, 67, 72],
                     "diminished": [60, 63, 66, 69],
                     "lydian": [60, 65, 67, 72]}

        # Get base notes for this scale
        notes = base_notes.get(scale_type, [60, 64, 67, 72])

        # Generate a melody with these notes
        melody = []
        for _ in range(16):  # 16 notes for a short melody
            note = random.choice(notes)
            duration = random.choice([0.25, 0.5, 1.0])  # Quarter, half, or whole note
            velocity = random.randint(80, 120)
            melody.append((note, duration, velocity))

        return melody

    def _create_midi_sequence(self, notes, tempo, intensity):
        """Create a MIDI sequence from notes with the given tempo and intensity"""
        # Simple implementation - in a real system, use a proper MIDI library
        sequence = {"notes": [], "tempo": tempo}

        current_time = 0.0
        for note, duration, velocity in notes:
            # Adjust velocity based on intensity
            adjusted_velocity = int(velocity * intensity)
            sequence["notes"].append({
                "pitch": note,
                "start_time": current_time,
                "end_time": current_time + duration,
                "velocity": adjusted_velocity
            })
            current_time += duration

        return sequence

    def _sequence_to_audio(self, sequence):
        """Convert a sequence to audio data"""
        try:
            # Simple implementation - in a real system, use a proper audio library
            sample_rate = 44100
            duration = sum(note["end_time"] - note["start_time"] for note in sequence["notes"])
            samples = int(duration * sample_rate)
            audio = np.zeros(samples)

            for note in sequence["notes"]:
                # Generate simple sine wave for each note
                start_sample = int(note["start_time"] * sample_rate)
                end_sample = int(note["end_time"] * sample_rate)

                # Make sure we don't exceed array bounds
                if end_sample > samples:
                    end_sample = samples

                if start_sample >= end_sample:
                    continue

                t = np.linspace(0, note["end_time"] - note["start_time"], end_sample - start_sample, False)

                # Convert MIDI note to frequency
                frequency = 440.0 * (2.0 ** ((note["pitch"] - 69) / 12.0))

                # Make sure we don't exceed array bounds
                if start_sample < len(audio) and end_sample <= len(audio):
                    audio[start_sample:end_sample] += 0.1 * note["velocity"] / 127.0 * np.sin(2 * np.pi * frequency * t)

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            return audio
        except Exception as e:
            print(f"Error in _sequence_to_audio: {str(e)}")
            # Return a simple sine wave as fallback
            sample_rate = 44100
            return np.sin(2 * np.pi * 440 * np.linspace(0, 2, sample_rate * 2))

    def generate_story(self, emotion_data, audio_data=None):
        """Generate a story based on detected emotion and audio analysis"""
        try:
            # Extract emotion
            if isinstance(emotion_data, dict) and "error" in emotion_data:
                primary_emotion = "neutral"
            else:
                primary_emotion = emotion_data["primary_emotion"].lower()

            print(f"Generating story for emotion: {primary_emotion}")

            # Create a prompt for the story based on the emotion
            prompt_mapping = {
                "happy": f"Once upon a time, there was someone feeling joyful. Their day was filled with happiness because",
                "sad": f"In a quiet moment of reflection, Sadness washed over them as they remembered",
                "excited": f"The adventure was about to begin!  They couldn't contain their excitement when",
                "confused": f"Nothing made sense anymore. They stood there confused, trying to understand why",
                "thoughtful": f"Deep in thought,  They contemplated life and wondered about",
                "surprised": f"It came out of nowhere!  Their jaw dropped in surprise when they discovered",
                "angry": f"The frustration was building inside.  They were angry because",
                "calm": f"In that peaceful moment,  A sense of calm settled over them as",
                "anxious": f"Heart racing, palms sweating. Anxiety crept in when they thought about",
                "confident": f"Standing tall with determination, They felt confident knowing that",
                "neutral": f"On an ordinary day,  They went about their business while"
            }

            prompt = prompt_mapping.get(primary_emotion, "Once upon a time,")

            # If we have audio data, analyze it to add context
            if audio_data is not None:
                # In a real system, you would analyze the audio for additional context
                # For now, we'll just add some generic text based on the emotion
                audio_context = {
                    "happy": " The sounds of celebration filled the air with melodious tunes.",
                    "sad": " A mournful melody echoed through the empty spaces.",
                    "angry": " Harsh, discordant notes punctuated the tense atmosphere.",
                    "surprised": " Unexpected harmonies created moments of wonder and awe.",
                    "fearful": " Eerie whispers and unsettling tones seemed to follow every step.",
                    "neutral": " Balanced, calm sounds provided a backdrop for reflection."
                }

                prompt += audio_context.get(primary_emotion, "")

            print(f"Using story prompt: {prompt}")

            # Generate story with error handling
            try:
                story = story_generator(prompt, max_length=250, do_sample=True, top_k=50)
                print(f"Story generation succeeded: {len(story[0]['generated_text'])} chars")
                return story[0]["generated_text"]
            except Exception as e:
                print(f"Error in story_generator: {str(e)}")
                # Fallback to a simple story
                return f"{prompt} The journey continued with many twists and turns, each moment reflecting the {primary_emotion} emotion that started it all."

        except Exception as e:
            print(f"Error in generate_story: {str(e)}")
            print(traceback.format_exc())
            return f"Once upon a time, a story began. Due to technical difficulties, the rest is left to your imagination."

    def generate_story_speech(self, story_text, emotion_data):
        """Generate speech from the story text with emotion-appropriate settings"""
        try:
            # Extract emotion
            if isinstance(emotion_data, dict) and "error" in emotion_data:
                primary_emotion = "neutral"
            else:
                primary_emotion = emotion_data["primary_emotion"].lower()

            print(f"Generating speech for story with emotion: {primary_emotion}")

            # Get speech parameters based on emotion
            mapping = self.emotion_mapping.get(primary_emotion, self.emotion_mapping["neutral"])

            # The gTTS doesn't support direct speed/pitch control, but we can
            # select appropriate language/voice that might convey the emotion
            # In a production system, you'd use a more advanced TTS with emotion control

            # Create a temporary file to save the speech
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_filename = temp_file.name

            # Generate speech using gTTS
            tts = gTTS(text=story_text, lang='en', slow=(mapping["speech_rate"] == "slow"))
            tts.save(temp_filename)

            # Read the audio file
            data = sf.read(temp_filename)
            sample_rate = data[1] if isinstance(data, tuple) and len(data) > 1 else 24000
            audio_data = data[0] if isinstance(data, tuple) else data

            # Ensure audio_data is a numpy array
            if not isinstance(audio_data, np.ndarray):
                print(f"Warning: audio_data is not a numpy array, it's a {type(audio_data)}")
                audio_data = np.zeros(sample_rate * 2)  # 2 seconds of silence

            # Clean up the temporary file
            try:
                os.unlink(temp_filename)
            except:
                print(f"Warning: Could not delete temporary file {temp_filename}")

            return (sample_rate, audio_data)

        except Exception as e:
            print(f"Error in generate_story_speech: {str(e)}")
            print(traceback.format_exc())

            # Return a simple spoken message as fallback
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_filename = temp_file.name

                fallback_text = "Sorry, I couldn't generate speech for the story due to a technical issue."
                tts = gTTS(text=fallback_text, lang='en')
                tts.save(temp_filename)

                data = sf.read(temp_filename)
                sample_rate = data[1] if isinstance(data, tuple) and len(data) > 1 else 24000
                audio_data = data[0] if isinstance(data, tuple) else data

                # Ensure audio_data is a numpy array
                if not isinstance(audio_data, np.ndarray):
                    print(f"Warning: fallback audio_data is not a numpy array, it's a {type(audio_data)}")
                    audio_data = np.zeros(sample_rate * 2)  # 2 seconds of silence

                try:
                    os.unlink(temp_filename)
                except:
                    pass

                return (sample_rate, audio_data)
            except Exception as e:
                print(f"Critical error in fallback speech generation: {str(e)}")
                # If all else fails, return silence
                return (44100, np.zeros(44100))

    def create_visualization(self, emotion_data, audio_data=None):
        """Create a visualization based on the emotion and audio data"""
        try:
            # Extract emotion
            if isinstance(emotion_data, dict) and "error" in emotion_data:
                primary_emotion = "neutral"
                color = "gray"
            else:
                primary_emotion = emotion_data["primary_emotion"].lower()
                color = self.emotion_mapping.get(primary_emotion, {}).get("color", "gray")

            print(f"Creating visualization for emotion: {primary_emotion} with color {color}")

            # Create a simple visualization
            plt.figure(figsize=(10, 6))

            # If we have audio data, create a waveform visualization
            if audio_data is not None and isinstance(audio_data, tuple) and len(audio_data) == 2:
                # Unpack the tuple
                sample_rate, audio = audio_data

                # Ensure audio is a numpy array
                if not isinstance(audio, np.ndarray):
                    print(f"Warning: audio is not a numpy array, it's a {type(audio)}")
                    audio = np.zeros(sample_rate * 2)  # 2 seconds of silence

                # Plot only a sample of the audio to avoid overloading the plot
                max_samples = 10000
                if len(audio) > max_samples:
                    step = len(audio) // max_samples
                    audio_sample = audio[::step]
                else:
                    audio_sample = audio

                plt.plot(np.linspace(0, len(audio_sample)/sample_rate, len(audio_sample)), audio_sample, color=color)
                plt.title(f"Waveform for '{primary_emotion}' emotion")
            else:
                # Create a simple visualization based just on the emotion
                x = np.linspace(0, 10, 1000)
                # Generate different patterns based on the emotion
                if primary_emotion == "happy":
                    y = np.sin(x) * np.exp(-0.1 * x)
                elif primary_emotion == "sad":
                    y = np.sin(x) * np.exp(-0.3 * x)
                elif primary_emotion == "angry":
                    y = np.sin(2 * x) * (1 - np.exp(-0.1 * x))
                elif primary_emotion == "surprised":
                    y = np.sin(3 * x) * np.exp(-0.2 * x)
                else:
                    y = np.sin(x)

                plt.plot(x, y, color=color)
                plt.title(f"Visualization for '{primary_emotion}' emotion")

            plt.tight_layout()

            # Save to bytesIO
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            # Convert to base64 for embedding in HTML
            vis_base64 = base64.b64encode(buf.read()).decode('utf-8')

            return f"<img src='data:image/png;base64,{vis_base64}' width='100%'/>"
        except Exception as e:
            print(f"Error in create_visualization: {str(e)}")
            print(traceback.format_exc())

            # Create an extremely simple fallback visualization
            try:
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"Visualization for {primary_emotion}",
                         horizontalalignment='center', verticalalignment='center', fontsize=20)
                plt.axis('off')

                # Save to bytesIO
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close()

                # Convert to base64
                vis_base64 = base64.b64encode(buf.read()).decode('utf-8')
                return f"<img src='data:image/png;base64,{vis_base64}' width='100%'/>"
            except:
                return f"<div style='width:100%;height:200px;background-color:{color};display:flex;justify-content:center;align-items:center;'><p style='color:white;font-size:24px;'>Visualization for {primary_emotion}</p></div>"

# Create the Gradio interface
composer = EmotionDrivenComposer()

def process_webcam(image):
    """Main processing function with extensive error handling"""
    print("\n--- Starting image processing ---")
    print(f"Image received: {type(image)}")

    # Check if image is valid
    if image is None:
        print("ERROR: Received None image")
        return [None, "Error: No image received",
                "<div style='color:red;'>No image to visualize</div>",
                None, "Error: No image to process", None]

    try:
        print(f"Image shape: {image.shape}, type: {image.dtype}")
    except:
        print("ERROR: Image doesn't have shape attribute")
        return [None, "Error: Invalid image format",
                "<div style='color:red;'>Invalid image format</div>",
                None, "Error: Invalid image format", None]

    try:
        # Step 1: Analyze face for emotions
        print("Starting face analysis...")
        emotion_data = composer.analyze_face(image)
        print(f"Face analysis complete: {emotion_data.get('primary_emotion', 'unknown')}")

        # Step 2: Generate music based on detected emotion
        print("Starting music generation...")
        audio = composer.generate_music(emotion_data)
        print("Music generation complete")

        # Step 3: Create visualization
        print("Creating visualization...")
        visualization = composer.create_visualization(emotion_data, audio)
        print("Visualization complete")

        # Step 4: Generate story
        print("Generating story...")
        story = composer.generate_story(emotion_data, audio)
        print("Story generation complete")

        # Step 5: Generate speech for the story
        print("Generating speech for story...")
        story_speech = composer.generate_story_speech(story, emotion_data)
        print("Speech generation complete")

        print("--- Processing completed successfully ---\n")

        # Return all results
        return (
            emotion_data.get("face_image", image),  # Use original image as fallback
            emotion_data.get("primary_emotion", "unknown"),
            visualization,
            audio,
            story,
            story_speech
        )
    except Exception as e:
        print(f"ERROR in process_webcam: {str(e)}")
        print(traceback.format_exc())

        # Create fallback responses that won't further break the UI
        return (
            image,  # Return original image
            f"Error: {str(e)[:50]}...",  # Truncated error message
            "<div style='padding:20px;background-color:#f8d7da;color:#721c24;'>Visualization error</div>",
            (44100, np.sin(2 * np.pi * 440 * np.linspace(0, 2, 44100 * 2))),  # Simple beep sound
            f"Once upon a time, an error occurred: {str(e)[:100]}...",
            (44100, np.sin(2 * np.pi * 440 * np.linspace(0, 2, 44100 * 2)))  # Simple beep sound for TTS fallback
        )

# Set up UI with additional debug info and TTS output
with gr.Blocks(title="Emotion-Driven Music Composer") as demo:
    gr.Markdown("""
    # Emotion-Driven Music Composer
    This app captures your facial expression, detects your emotion, and generates music, a story, and narration that matches your emotional state.
    """)

    # Add debug information
    with gr.Accordion("Debug Info", open=False):
        debug_info = gr.Markdown(f"""
        - Running Gradio version: {gr.__version__}
        - OpenCV version: {cv2.__version__}
        - NumPy version: {np.__version__}
        - PIL version: {Image.__version__}
        - Face cascade loaded: {"Yes" if hasattr(composer, 'face_cascade') and composer.face_cascade is not None else "No"}
        - gTTS version: {"Installed" if 'gTTS' in globals() else "Not installed"}
        """)

    with gr.Row():
        with gr.Column():
            webcam_input = gr.Image(label="Webcam", type="numpy")
            webcam_button = gr.Button("Capture from Webcam", variant="primary")

            # Add additional upload option
            upload_button = gr.Button("Use Uploaded Image")

            # Add status message
            status = gr.Markdown("Status: Ready. Click 'Capture from Webcam' to start.")

        with gr.Column():
            face_output = gr.Image(label="Detected Face")
            emotion_output = gr.Text(label="Detected Emotion")

    with gr.Row():
        visualization_output = gr.HTML(label="Music Visualization")

    with gr.Row():
        audio_output = gr.Audio(label="Generated Music")
        story_output = gr.Text(label="Generated Story", lines=10)

    # Add TTS audio output
    with gr.Row():
        story_speech_output = gr.Audio(label="Story Narration")

    # Define process function with status updates
    def process_with_status(image):
        status.value = "Status: Processing image..."
        try:
            results = process_webcam(image)
            status.value = "Status: Processing complete!"
            return results + (status,)
        except Exception as e:
            status.value = f"Status: Error - {str(e)}"
            return [None, f"Error: {str(e)}", "<div>Error</div>", None, f"Error: {str(e)}", None, status]

    # Connect buttons
    webcam_button.click(
        fn=process_with_status,
        inputs=webcam_input,
        outputs=[face_output, emotion_output, visualization_output, audio_output, story_output, story_speech_output, status]
    )

    upload_button.click(
        fn=process_with_status,
        inputs=webcam_input,
        outputs=[face_output, emotion_output, visualization_output, audio_output, story_output, story_speech_output, status]
    )

if __name__ == "__main__":
    # Print debugging info at startup
    print("\n=== Emotion-Driven Music Composer with Text-to-Speech ===")
    print(f"Gradio version: {gr.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"PIL version: {Image.__version__}")
    print(f"Face cascade loaded: {hasattr(composer, 'face_cascade') and composer.face_cascade is not None}")
    print("Make sure you've installed gTTS with: pip install gtts")
    print("===================================\n")

    # Launch with debug settings
    demo.launch(debug=True, share=True)