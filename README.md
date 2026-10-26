# SanviBot - Conversational AI Assistant

A sleek, self-branded conversational AI assistant powered by Google GenAI models, equipped with **multi-turn short-term memory**, **side-aligned chat bubbles** (user on the right, bot on the left), and **dynamic model selection**.

---

- **Claude & ChatGPT-Styled Dark Aesthetic**:
  - **Groups / Folders at Top**: Organize chats into custom folders (`Work`, `Personal`, `Projects`) placed directly above recent chats.
  - **Horizontal Three-Dots (`⋯`) Menu**:
    - 🗗 **Open in new window**: Opens the exact conversation in a separate window/tab.
    - 📌 **Pin / Unpin (P)**: Keep important chats anchored at the top.
    - ✏️ **Rename (R)**: Quick title editing with instant toast confirmation.
    - 📁 **Move to group (>)**: Seamlessly move chats between custom folders or ungroup.
    - 🗑️ **Delete (D)**: Instant deletion styled in red.
  - **Reference History Representation**: Clean chat list with `○` bullet indicators, active highlight, and horizontal `⋯` dots.
  - **Centered New Chat View**: Vertically centered initial screen ("What can I help with today?") with 4 instant suggestion cards.
  - **Conversation View**: User messages in right-aligned dark rounded bubbles; assistant responses formatted cleanly on the left.
  - **Floating Scroll Button (`↓`)**: Floating round button above the input bar to jump to latest messages.
  - **User Profile Badge**: `Shobhan · Free ▾` profile badge docked at the bottom of the sidebar.
  - **Bottom-Pinned Chat Bar**: Fixed `Write a message...` input bar with accuracy disclaimer.
  - **Zero Clutter**: Removed all memory turns counters and extraneous indicators.
- **Dynamic Model Selection**: Instant model switching (`gemini-3.5-flash-lite`, `gemini-3.5-flash`, `gemini-3.8-flash`, `gemini-flash-latest`).
- **Persistent Multi-Session Storage**: Conversations and folder assignments persist across restarts in `chat_sessions.json`.
- **Standalone Desktop App**: Launch borderless via `python ui.py` or through browser via `streamlit run chatbot.py`.

---

## Project Structure

```
chatbot/
├── .env                  # Environment variables (GOOGLE_API_KEY)
├── .venv/                # Python virtual environment with dependencies
├── bot.py                # Base reference script provided
├── chatbot.py            # Main application (Streamlit UI & CLI modes)
├── requirements.txt      # Project dependencies (google-genai, python-dotenv, streamlit)
├── environment.yml       # Conda environment definition (chatbot_env)
├── chatbot_prompt.txt    # Initial stateless chatbot requirements
├── memory_prompt.txt     # Short-term memory & Streamlit UI requirements
└── README.md             # Complete documentation and execution guide
```

---

## Getting Started

### 1. Prerequisites

- Python 3.11 or 3.12 (or active `.venv` created in project)
- A Google Gemini API Key in `.env`

### 2. Environment Setup

The virtual environment `.venv` is pre-configured in this repository. If you ever need to set it up again:

#### Standard Python Virtual Environment:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Conda Environment (Optional):
```bash
conda env create -f environment.yml
conda activate chatbot_env
```

### 3. API Key Configuration

Verify that `.env` contains your Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## Running the Application

### Option A: Desktop Application Window (Fastest & Recommended)

Launch directly with:
```powershell
python ui.py
```

- Automatically manages background server startup and port binding.
- Launches a **dedicated standalone application window** (App Mode) without browser tabs or address bars.
- Automatically handles Python virtual environment resolution (no manual activation required).
- Closing the window or pressing `Ctrl + C` in the terminal cleanly shuts down the application.

---

### Option B: Streamlit Web UI in Browser

Run with the Streamlit CLI:
```powershell
.\.venv\Scripts\streamlit.exe run chatbot.py
```

*Or launch via Python directly:*
```powershell
.\.venv\Scripts\python.exe chatbot.py
```

Once running, navigate to **`http://localhost:8501`**.

---

### Option B: Terminal CLI Mode

To interact via the command-line with short-term memory:
```powershell
.\.venv\Scripts\python.exe chatbot.py --cli
```

- Type your message and press **Enter**.
- Type `clear` to reset the short-term memory.
- Type `exit`, `quit`, or `q` to exit.
