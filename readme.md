# 🤖 AI Operating Assistant
### A voice-first personal AI that runs your computer — by **Shanu Kumari**

A real-time voice assistant that can hear, see, speak and **operate your computer**: open and close apps, run terminal commands, build full-stack projects, research the web, watch the news and weather, and take orders from your phone. It answers only to **your** voice.

Built on the Gemini Live API for native, low-latency voice. Works on Linux, Windows and macOS.

---

## ✨ Features

| | Feature | What it does |
|---|---|---|
| 🧠 | **Memory / Soul** | Remembers your preferences, past conversations and *how you like to be answered* (short, Hinglish, no "sir"…) — and uses it every session. |
| 🤖 | **Multi-agent system** | Simple commands run instantly; multi-step work is handed to a **Dev**, **Research**, **System** or **Task** agent that works in the background and reports back. |
| 💻 | **Developer agent** | Builds, runs, debugs and stops real projects: Python, React/Vite, Node/Express, MongoDB and full-stack **MERN** — verified (build / API health check) and auto-fixed. |
| 🖥️ | **System control** | Apps, files and folders, terminal commands, CPU/RAM/process monitoring. Destructive actions wait for an on-screen **CONFIRM**. |
| 🌐 | **World monitor** | News, real weather forecasts, AI/tech updates and important alerts (severe weather, major breaking news). |
| 📱 | **Mobile remote** | Control it from your phone: commands, voice, live PC status, start/cancel agent tasks, stop projects. |
| 🔒 | **Voice lock** | Speaker verification — only the enrolled owner's voice reaches the assistant; anyone else hears *"Sorry, you are not my owner."* |
| 🎨 | **Glass HUD** | Animated 3D avatar, live system cards, agent status, activity timeline, news, quick actions. |

---

## ⚡ Quick Start

```bash
git clone https://github.com/Shanu-Git2002/Ai-operating-assistant.git
cd Ai-operating-assistant
python setup.py        # installs the dependencies for your OS
python main.py
```

On first launch the setup screen asks for your **Gemini API key** (free from Google AI Studio). Everything else is optional and can be switched on from inside the app (⚙ / 🎛).

**Voice lock:** say *"enroll my voice"* and talk for ~20 seconds. It switches on automatically. *"Voice lock stricter / looser / off"* to adjust.

**Phone remote:** ⚙ → **Remote Control**, then scan the QR code with your phone.

---

## 🗣️ Things to try

- "Open YouTube in Chrome" · "Close the YouTube tab" · "Close Chrome"
- "Build a MERN notes app with login"
- "Compare the top 3 phones under 30,000 and tell me which to buy"
- "What's using my RAM?" · "Run git status in my project"
- "Give me today's briefing" · "Latest AI news"
- "What did we talk about last time?"

---

## 🔐 Privacy

Your API keys (`config/api_keys.json`), memory, conversation history and voiceprint stay on your machine and are git-ignored. The voice-lock model runs locally — your voice is never uploaded for verification.

---

## 📜 License & credits

**AI Operating Assistant** is developed and maintained by **Shanu Kumari**.

It is based on **[MARK LV](https://github.com/FatihMakes/Mark-LV)** by **FatihMakes**, used under the
**[Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** license — personal and
non-commercial use only. See [LICENSE](LICENSE).
