
# 🎵 Emotion Composer

**Emotion Composer** is an AI-powered application that captures a user's facial emotion and generates custom music, a narrated story, and a visual waveform based on the detected emotion. It's an interactive creative experience using state-of-the-art models in computer vision, language generation, and audio synthesis.

---

##  What It Does

1. **Emotion Detection** – Captures an image (via webcam or upload) and detects the user's facial emotion using a Hugging Face model.
2. **Music Generation** – Generates a custom melody reflecting the user's emotional state.
3. **Story Creation** – Crafts a personalized short story based on the detected emotion using GPT-2.
4. **Text-to-Speech** – Narrates the generated story using Google's TTS.
5. **Visualization** – Displays a real-time waveform or abstract visualization matching the mood of the emotion.

---

## 🛠 AI Tools and Models Used

| Feature               | AI Tool / Model                                             |
|----------------------|-------------------------------------------------------------|
| Emotion Detection     | `dima806/facial_emotions_image_detection` (Hugging Face)   |
| Story Generation      | `gpt2` (Hugging Face)                                       |
| Text-to-Speech (TTS)  | `gTTS` (Google Text-to-Speech)                              |
| Visualization         | `matplotlib` + `NumPy` waveform plotting                    |

---

##  How to Run the Project

###  Option 1: Run Locally with Python

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/emotion-composer.git
   cd emotion-composer

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Run the app**:
   ```bash
   python app.py




