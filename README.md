# CodeWithKiroHckathon



# 🤟 Hindi Sign Language Recognition

A real-time sign language recognition system that translates hand gestures into **Hindi characters**, which are then converted into **English text and speech**.

---

## 🌟 Motivation

While researching sign language datasets, I noticed a lack of **linguistic diversity**—especially for Indian languages. Most datasets focused on American Sign Language (ASL), leaving speakers of other languages underrepresented.

So I created my own dataset from scratch, featuring **48 Hindi gestures**, extracted using **MediaPipe** for keypoint detection. This project is not just a technical challenge—it’s a mission to:

- Promote **diversity in AI** by including underrepresented languages
- Empower **disabled communities** by making communication more accessible
- Showcase **AI as a force for social good**, bridging gaps between languages and people

---

## 🛠️ Features

- 📷 Real-time hand gesture recognition using webcam
- 🧠 Custom CNN model trained on 48 Hindi gestures
- 🔤 Hindi-to-English transliteration
- 🔊 Text-to-speech output
- 🧩 Built with MediaPipe, TensorFlow, and Python

---

## 🤖 Built with Kiro

This project was developed using **Kiro**, an AI-powered development assistant that helped me structure, build, and optimize the entire pipeline.

### 🧠 How I Used Kiro

#### 1. Vibe Coding for Ideation
- Described my vision to Kiro in natural language
- Brainstormed features and refined the scope

#### 2. Spec-Driven Development
- Defined structured tasks and architecture
- Kiro generated modular design and task list

#### 3. Steering Files for Consistency
- Specified tech stack, coding style, and folder structure

#### 4. Impressive Code Generation
- CNN model for gesture classification
- Hindi-to-English mapping logic
- Text-to-speech integration
- Unit tests and performance optimizations

#### 5. Workflow Automation
- Task tracking and seamless session resumption

---

## 💡 Impact

This project is a step toward making **AI inclusive and accessible**. It demonstrates how AI can:

- Celebrate **linguistic diversity**
- Support **disabled communities**
- Build **bridges between cultures**

By combining **technical innovation** with **social purpose**, this project reflects the true power of AI.

---

## 📁 Project Structure

```
├── dataset/                # Custom Hindi gesture dataset
├── model/                  # CNN model and training scripts
├── mediapipe_utils/        # Keypoint extraction scripts
├── transliteration/        # Hindi to English mapping
├── tts/                    # Text-to-speech integration
├── app.py                  # Main application script
└── README.md               # Project documentation


## 📜 License

This project is open source and available under the [MIT License](LICENSE).
