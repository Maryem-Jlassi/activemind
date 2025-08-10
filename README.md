# 🧠 ActiveMind — AI Mental Performance Coach

**ActiveMind** is an AI-powered, voice-enabled mental performance coach for athletes.  
It delivers real-time, adaptive mental wellness guidance to help manage **stress, focus, and motivation** — anytime, anywhere.  
Unlike static wellness apps, ActiveMind’s conversational AI dynamically adapts to each athlete’s mood, age category ,needs, and performance state.

---

## 🚀 Features

- 🎙 **Real-Time Voice Coaching** — Natural, low-latency audio conversations using ElevenLabs Voice API.  
- 🤖 **Multi-Agent Architecture** — Three specialized agents:
  - **Data Gathering Agent** — Builds a personalized understanding of the athlete.
  - **Therapist Agent** — Asks context-aware mental performance questions.
  - **Motivational Agent** — Delivers tailored affirmations and exercises.
- 🧩 **Dynamic Conversation Flow** — No static datasets; sessions adapt in real time.
- 📂 **Progress Tracking** — SQLite database stores profiles and history for personalization.
- 🗣 **Speech-to-Text Integration** — Google Speech Recognition processes voice input 
- 🎧 **Audio Playback** — PyGame handles smooth voice output in the UI.
- 📱 **Mobile-Accessible UI** — React frontend with responsive design.

---

## 🎯 Problem & Challenge

Athletes often lack affordable, personalized mental coaching that is accessible on demand.  
ActiveMind addresses this by delivering an **AI-driven sports psychology partner** that feels like a real coach — without the need for pre-built datasets or expensive in-person sessions.

---

## 👥 Target Audience

- **Primary:** Athletes seeking real-time, tailored mental support before, during, or after training/competition.  
- **Secondary:** Coaches, sports psychologists, and wellness programs looking for scalable tools.

---


## 🛠 Tech Stack

**Frontend:**
- React  
- Tailwind CSS  
- Framer Motion  
- Lucide-React Icons

**Backend:**
- Flask + Flask-SocketIO  
- Python Multi-Agent System (crewai)
- SQLite (database)

**APIs & Libraries:**
- ElevenLabs Voice API (TTS)  
- Google Speech Recognition (STT)  
- PyGame (Audio Playback)  

---

## 🔮 Future Improvements

   -Add 3D animated avatar for a more immersive coaching presence.

   -Expand agent personalities for varied coaching styles.

   -Integrate biometric data (e.g., heart rate) for performance-based adjustments.

 ## 👏 Acknowledgments
 
  -ElevenLabs — Voice synthesis API

  -Flask-SocketIO — Real-time backend communication

  -Framer Motion & Tailwind — UI polish  
git clone https://github.com/YOUR_USERNAME/activemind.git
cd activemind
