"""
SanviBot - Self-Branded Conversational AI Assistant.
Accurately styled to mimic the reference Claude interface:
- Groups displayed cleanly as simple text headers (e.g. "Sanvi ▾" and "Ungrouped ▾")
- Chat history displayed as plain text lines with '○' bullets, strictly left-aligned
- Three dots (⋯) hidden by default, appearing smoothly on row hover
- Context menu options displayed as simple text rows
- Clean modal dialogs (@st.dialog) for Rename and Move to Group
- Active chat highlighted in a soft rounded dark rectangle (#38383a)
- Vertically centered chat window for initial "New Chat" state
- Top-down conversation flow with right-aligned user bubble and bottom-pinned input
- User profile footer (Shobhan · Free ▾)
- Floating scroll-to-bottom button (↓)
- No memory turn counter
Powered by Google GenAI SDK.
"""

import os
import sys
import copy
import time
import json
import uuid
import datetime
import threading
from typing import Generator, List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st
import streamlit.components.v1 as components
from tools import ALL_TOOLS, SYSTEM_INSTRUCTION, get_system_instruction, should_prefer_web_search

# Ensure user-scope Git/GitHub CLI are in PATH for Streamlit's Deploy button detection
for _extra_bin in (
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "cmd"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "GitHubCLI", "bin"),
):
    if _extra_bin and os.path.isdir(_extra_bin) and _extra_bin.lower() not in os.environ.get("PATH", "").lower():
        os.environ["PATH"] = _extra_bin + os.pathsep + os.environ.get("PATH", "")

# Load environment variables from .env
load_dotenv()

# Available models
AVAILABLE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]

DEFAULT_MODEL = "gemini-3.5-flash-lite"
SESSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_sessions.json")


def get_api_key() -> str:
    """Retrieve the Google API key from environment (.env) or Streamlit Cloud secrets."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
        except Exception:
            api_key = None
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Please add it to your .env file locally or Streamlit Secrets in the cloud:\n"
            "GOOGLE_API_KEY=your_gemini_api_key"
        )
    return api_key


@st.cache_resource(show_spinner=False)
def _get_cached_client(api_key: str) -> genai.Client:
    """Keep a single warm Google GenAI client and HTTP connection pool across all clicks."""
    return genai.Client(api_key=api_key)


def create_client() -> genai.Client:
    """Initialize and return the cached Google GenAI client."""
    return _get_cached_client(get_api_key())


_CHAT_GEN_CONFIG = types.GenerateContentConfig(
    system_instruction=get_system_instruction(force_web_search=False),
    tools=ALL_TOOLS,
)

_CHAT_GEN_CONFIG_WEB_SEARCH = types.GenerateContentConfig(
    system_instruction=get_system_instruction(force_web_search=True),
    tools=ALL_TOOLS,
)

_CHAT_GEN_CONFIG_NATIVE_SEARCH = types.GenerateContentConfig(
    system_instruction=(
        f"You are SanviBot, a helpful AI assistant. Current Date: {datetime.date.today().strftime('%B %d, %Y')}. "
        f"Always use Google Search to verify current officeholders, roles, news, prices, and time-sensitive facts "
        f"as of {datetime.date.today().year}, and answer directly and concisely."
    ),
    tools=[types.Tool(google_search=types.GoogleSearch())],
)

# --- PERSISTENT SESSION STORAGE (IN-MEMORY CACHE + NON-BLOCKING ASYNC DISK FLUSH) ---
_SAVE_LOCK = threading.Lock()
_SESSIONS_MEMORY_CACHE: Optional[Dict[str, Any]] = None
_LAST_DISK_MTIME: float = 0.0


def _flush_sessions_to_disk(snapshot: Dict[str, Any]):
    """Background worker that writes chat_sessions.json without blocking UI clicks."""
    global _LAST_DISK_MTIME
    with _SAVE_LOCK:
        try:
            tmp_file = SESSIONS_FILE + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            os.replace(tmp_file, SESSIONS_FILE)
            _LAST_DISK_MTIME = os.path.getmtime(SESSIONS_FILE)
        except Exception as e:
            try:
                with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=2, ensure_ascii=False)
                _LAST_DISK_MTIME = os.path.getmtime(SESSIONS_FILE)
            except Exception as inner_e:
                print(f"Error saving chat sessions: {inner_e or e}", file=sys.stderr)


def load_sessions_data() -> Dict[str, Any]:
    """Return in-memory chat sessions immediately; only read from disk on startup or external change."""
    global _SESSIONS_MEMORY_CACHE, _LAST_DISK_MTIME

    if "sessions_data" in st.session_state and isinstance(st.session_state["sessions_data"], dict):
        cached_sessions = st.session_state["sessions_data"].get("sessions", {})
        if "chat_photo_edit" not in cached_sessions:
            return st.session_state["sessions_data"]
        st.session_state.pop("sessions_data", None)
        _SESSIONS_MEMORY_CACHE = None

    if _SESSIONS_MEMORY_CACHE is not None:
        if "chat_photo_edit" not in _SESSIONS_MEMORY_CACHE.get("sessions", {}):
            try:
                curr_mtime = os.path.getmtime(SESSIONS_FILE) if os.path.exists(SESSIONS_FILE) else 0.0
                if curr_mtime <= _LAST_DISK_MTIME:
                    st.session_state["sessions_data"] = _SESSIONS_MEMORY_CACHE
                    return _SESSIONS_MEMORY_CACHE
            except Exception:
                st.session_state["sessions_data"] = _SESSIONS_MEMORY_CACHE
                return _SESSIONS_MEMORY_CACHE
        _SESSIONS_MEMORY_CACHE = None

    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "sessions" in data and isinstance(data["sessions"], dict):
                    data.setdefault("groups", ["Sanvi"])
                    for s in data["sessions"].values():
                        s.setdefault("group", None)
                        s.setdefault("pinned", False)

                    if data["sessions"]:
                        if data.get("active_chat_id") not in data["sessions"]:
                            data["active_chat_id"] = next(iter(data["sessions"]))
                        _SESSIONS_MEMORY_CACHE = data
                        _LAST_DISK_MTIME = os.path.getmtime(SESSIONS_FILE)
                        st.session_state["sessions_data"] = data
                        return data
        except Exception:
            pass

    # Create default session
    initial_id = "chat_default"
    initial_data = {
        "active_chat_id": initial_id,
        "groups": [],
        "sessions": {
            initial_id: {
                "id": initial_id,
                "title": "New Chat",
                "group": None,
                "created_at": time.time(),
                "updated_at": time.time(),
                "pinned": False,
                "messages": [],
            }
        },
    }
    _SESSIONS_MEMORY_CACHE = initial_data
    st.session_state["sessions_data"] = initial_data
    save_sessions_data(initial_data)
    return initial_data


def save_sessions_data(data: Dict[str, Any]):
    """Update in-memory session cache immediately (0ms) and flush to disk asynchronously."""
    global _SESSIONS_MEMORY_CACHE
    _SESSIONS_MEMORY_CACHE = data
    st.session_state["sessions_data"] = data
    try:
        snapshot = copy.deepcopy(data)
        threading.Thread(target=_flush_sessions_to_disk, args=(snapshot,), daemon=True).start()
    except Exception as e:
        print(f"Error scheduling session save: {e}", file=sys.stderr)


def get_ordered_chat_ids(sessions: Dict[str, Any], all_groups: List[str]) -> List[str]:
    """Return all chat session IDs in the exact order they are displayed in the sidebar."""
    ordered = []
    for grp in all_groups:
        grp_chats = [s for s in sessions.values() if s.get("group") == grp]
        grp_chats.sort(key=lambda s: (not s.get("pinned", False), -s.get("updated_at", 0)))
        for s in grp_chats:
            if s["id"] not in ordered:
                ordered.append(s["id"])
    ungrouped = [s for s in sessions.values() if not s.get("group") or s.get("group") not in all_groups]
    ungrouped.sort(key=lambda s: (not s.get("pinned", False), -s.get("updated_at", 0)))
    for s in ungrouped:
        if s["id"] not in ordered:
            ordered.append(s["id"])
    for s_id in sessions:
        if s_id not in ordered:
            ordered.append(s_id)
    return ordered


def format_history_for_sdk(messages: List[Dict[str, str]]) -> List[types.Content]:
    """Convert conversational message dictionaries to SDK Content objects."""
    history: List[types.Content] = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        history.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )
    return history


def stream_chat_response(
    client: genai.Client,
    history_messages: List[Dict[str, str]],
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> Generator[str, None, None]:
    """Call the LLM with short-term conversational memory and single-pass native Google Search grounding for live queries."""
    history = format_history_for_sdk(history_messages)

    # Fast single-pass path for online/time-sensitive queries: use native GoogleSearch() grounding directly
    # (1 streaming API call instead of Outer LLM -> Python tool -> Inner LLM -> Scrapers -> Outer LLM)
    if should_prefer_web_search(prompt):
        try:
            chat = client.chats.create(
                model=model,
                history=history,
                config=_CHAT_GEN_CONFIG_NATIVE_SEARCH,
            )
            response_stream = chat.send_message_stream(prompt)
            yielded_any = False
            for chunk in response_stream:
                if chunk.text:
                    yielded_any = True
                    yield chunk.text
            if yielded_any:
                return
        except Exception:
            # Fallback to Python web_search tool if the selected model does not support native GoogleSearch
            pass

        chat = client.chats.create(
            model=model,
            history=history,
            config=_CHAT_GEN_CONFIG_WEB_SEARCH,
        )
        for chunk in chat.send_message_stream(prompt):
            if chunk.text:
                yield chunk.text
        return

    chat = client.chats.create(
        model=model,
        history=history,
        config=_CHAT_GEN_CONFIG,
    )

    response_stream = chat.send_message_stream(prompt)
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text


# --- SINGLE-PASS INSTANT ACTION CALLBACKS (ELIMINATES DOUBLE RERUNS) ---

def _cb_select_chat(s_id: str):
    """Switch active chat in a single rerun pass."""
    sessions_data = load_sessions_data()
    if s_id not in sessions_data["sessions"]:
        return
    sessions_data["active_chat_id"] = s_id
    hist = st.session_state.get("nav_history", [])
    idx = st.session_state.get("nav_index", 0)
    if not hist:
        st.session_state["nav_history"] = [s_id]
        st.session_state["nav_index"] = 0
    elif 0 <= idx < len(hist) and hist[idx] != s_id:
        new_hist = hist[:idx + 1] + [s_id]
        st.session_state["nav_history"] = new_hist
        st.session_state["nav_index"] = len(new_hist) - 1
    save_sessions_data(sessions_data)


def _cb_toggle_pin(s_id: str):
    """Pin or unpin a chat in a single rerun pass."""
    sessions_data = load_sessions_data()
    s = sessions_data["sessions"].get(s_id)
    if s is not None:
        s["pinned"] = not s.get("pinned", False)
        save_sessions_data(sessions_data)


def _cb_trigger_rename(s_id: str):
    """Open rename chat modal in a single rerun pass."""
    st.session_state["trigger_rename_id"] = s_id


def _cb_trigger_move(s_id: str):
    """Open move/project modal in a single rerun pass."""
    st.session_state["trigger_move_id"] = s_id


def _cb_trigger_rename_folder(grp: str):
    """Open rename folder modal in a single rerun pass."""
    st.session_state["trigger_rename_folder"] = grp


def _cb_trigger_delete_folder(grp: str):
    """Open delete folder modal in a single rerun pass."""
    st.session_state["trigger_delete_folder"] = grp


def _cb_delete_chat(s_id: str):
    """Delete a chat and update active selection in a single rerun pass."""
    sessions_data = load_sessions_data()
    sessions = sessions_data["sessions"]
    if s_id not in sessions:
        return
    del sessions[s_id]
    if "nav_history" in st.session_state:
        st.session_state["nav_history"] = [c for c in st.session_state["nav_history"] if c != s_id]
        st.session_state["nav_index"] = min(
            st.session_state.get("nav_index", 0),
            max(0, len(st.session_state["nav_history"]) - 1),
        )
    if sessions_data["active_chat_id"] == s_id:
        if sessions:
            sessions_data["active_chat_id"] = next(iter(sessions))
        else:
            fresh_id = f"chat_{uuid.uuid4().hex[:8]}"
            sessions[fresh_id] = {
                "id": fresh_id,
                "title": "New Chat",
                "created_at": time.time(),
                "updated_at": time.time(),
                "pinned": False,
                "group": None,
                "messages": [],
            }
            sessions_data["active_chat_id"] = fresh_id
    save_sessions_data(sessions_data)
    st.session_state["pending_toast"] = ("Chat deleted", "🗑️")


def _cb_new_chat(group_name: Optional[str] = None):
    """Create a new chat in a single rerun pass."""
    sessions_data = load_sessions_data()
    sessions = sessions_data["sessions"]
    new_id = f"chat_{uuid.uuid4().hex[:8]}"
    sessions[new_id] = {
        "id": new_id,
        "title": "New Chat",
        "group": group_name,
        "created_at": time.time(),
        "updated_at": time.time(),
        "pinned": False,
        "messages": [],
    }
    sessions_data["active_chat_id"] = new_id
    save_sessions_data(sessions_data)
    hist = st.session_state.get("nav_history", [])
    idx = st.session_state.get("nav_index", 0)
    new_hist = hist[:idx + 1] + [new_id]
    st.session_state["nav_history"] = new_hist
    st.session_state["nav_index"] = len(new_hist) - 1


def _cb_nav_step(direction: str):
    """Navigate back/forward through chat history in a single rerun pass."""
    sessions_data = load_sessions_data()
    sessions = sessions_data["sessions"]
    all_groups = sessions_data.get("groups", ["Sanvi"])
    active_chat_id = sessions_data.get("active_chat_id")
    hist = st.session_state.get("nav_history", [active_chat_id] if active_chat_id else [])
    idx = st.session_state.get("nav_index", 0)
    ordered_ids = get_ordered_chat_ids(sessions, all_groups)
    target_id = None

    if direction == "prev":
        if idx > 0 and (idx - 1) < len(hist) and hist[idx - 1] in sessions:
            idx -= 1
            target_id = hist[idx]
            st.session_state["nav_index"] = idx
        elif ordered_ids:
            curr_idx = ordered_ids.index(active_chat_id) if active_chat_id in ordered_ids else 0
            prev_idx = (curr_idx - 1) % len(ordered_ids)
            target_id = ordered_ids[prev_idx]
            st.session_state["nav_history"] = [target_id] + hist
            st.session_state["nav_index"] = 0
    elif direction == "next":
        if (idx + 1) < len(hist) and hist[idx + 1] in sessions:
            idx += 1
            target_id = hist[idx]
            st.session_state["nav_index"] = idx
        elif ordered_ids:
            curr_idx = ordered_ids.index(active_chat_id) if active_chat_id in ordered_ids else 0
            next_idx = (curr_idx + 1) % len(ordered_ids)
            target_id = ordered_ids[next_idx]
            hist = hist[:idx + 1] + [target_id]
            st.session_state["nav_history"] = hist
            st.session_state["nav_index"] = len(hist) - 1

    if target_id and target_id in sessions and target_id != active_chat_id:
        sessions_data["active_chat_id"] = target_id
        save_sessions_data(sessions_data)


def _cb_submit_prompt(direct_prompt: Optional[str] = None, state_key: Optional[str] = None):
    """Queue user message before the rerun starts so user bubble + streaming happen in 1 single pass."""
    raw = direct_prompt if direct_prompt is not None else (st.session_state.get(state_key, "") if state_key else "")
    prompt_text = (raw or "").strip()
    if not prompt_text:
        return
    sessions_data = load_sessions_data()
    active_id = sessions_data.get("active_chat_id")
    current_session = sessions_data["sessions"].get(active_id)
    if not current_session:
        return
    if current_session.get("title") == "New Chat":
        clean_title = prompt_text.split("\n")[0]
        current_session["title"] = (clean_title[:32] + "...") if len(clean_title) > 32 else clean_title
    current_session.setdefault("messages", []).append({"role": "user", "content": prompt_text})
    current_session["updated_at"] = time.time()
    save_sessions_data(sessions_data)
    st.session_state["pending_prompt"] = prompt_text
    if state_key in ("hero_text_input", "hero_chat_input"):
        st.session_state[state_key] = ""


# --- SIMPLE TEXT MODAL DIALOGS ---

@st.dialog("✏️ Rename Chat")
def show_rename_modal(s_id: str, sessions_data: Dict[str, Any]):
    """Clean modal dialog to rename a chat without cluttering the sidebar."""
    session = sessions_data["sessions"].get(s_id)
    if not session:
        st.error("Session not found.")
        return
    curr_title = session.get("title", "")
    new_title = st.text_input("Enter new chat title", value=curr_title)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Title", type="primary", use_container_width=True):
            if new_title.strip():
                session["title"] = new_title.strip()
                session["updated_at"] = time.time()
                save_sessions_data(sessions_data)
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("📁 Move to Group")
def show_move_group_modal(s_id: str, sessions_data: Dict[str, Any], all_groups: List[str]):
    """Clean modal dialog to move a chat to a group folder or create a new one."""
    session = sessions_data["sessions"].get(s_id)
    if not session:
        st.error("Session not found.")
        return
    curr_grp = session.get("group")
    group_options = ["None (Ungrouped)"] + all_groups + ["➕ Create New Group..."]
    curr_idx = group_options.index(curr_grp) if curr_grp in group_options else 0
    selected_grp = st.selectbox("Select destination group", options=group_options, index=curr_idx)

    new_group_name = ""
    if selected_grp == "➕ Create New Group...":
        new_group_name = st.text_input("New Group Name", placeholder="e.g. Research, Work...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply Move", type="primary", use_container_width=True):
            if selected_grp == "➕ Create New Group...":
                grp_to_set = new_group_name.strip()
                if grp_to_set and grp_to_set not in all_groups:
                    all_groups.append(grp_to_set)
                    sessions_data["groups"] = all_groups
            elif selected_grp == "None (Ungrouped)":
                grp_to_set = None
            else:
                grp_to_set = selected_grp

            session["group"] = grp_to_set
            session["updated_at"] = time.time()
            save_sessions_data(sessions_data)
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("✏️ Rename Folder")
def show_rename_folder_modal(old_name: str, sessions_data: Dict[str, Any], all_groups: List[str]):
    """Clean modal dialog to rename a folder group."""
    new_name = st.text_input("Folder Name", value=old_name)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save", type="primary", use_container_width=True):
            clean_name = new_name.strip()
            if clean_name and clean_name != old_name:
                idx = all_groups.index(old_name) if old_name in all_groups else -1
                if idx >= 0:
                    all_groups[idx] = clean_name
                else:
                    all_groups.append(clean_name)
                sessions_data["groups"] = all_groups
                for s in sessions_data["sessions"].values():
                    if s.get("group") == old_name:
                        s["group"] = clean_name
                save_sessions_data(sessions_data)
                st.toast(f"Renamed folder to '{clean_name}'", icon="✏️")
                st.rerun()
            elif clean_name == old_name:
                st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("🗑️ Delete Folder")
def show_delete_folder_modal(grp_name: str, sessions_data: Dict[str, Any], all_groups: List[str]):
    """Clean modal dialog to delete a folder group."""
    st.write(f"Are you sure you want to delete the folder **{grp_name}**?")
    st.caption("Chats inside this folder will not be lost; they will be moved to Ungrouped.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Delete Folder", type="primary", use_container_width=True):
            if grp_name in all_groups:
                all_groups.remove(grp_name)
                sessions_data["groups"] = all_groups
            for s in sessions_data["sessions"].values():
                if s.get("group") == grp_name:
                    s["group"] = None
            save_sessions_data(sessions_data)
            st.toast(f"Folder '{grp_name}' deleted", icon="🗑️")
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def render_sidebar_chat_item(
    s: Dict[str, Any],
    sessions_data: Dict[str, Any],
    all_groups: List[str],
    active_chat_id: str,
    key_prefix: str = "",
):
    """
    Renders a chat item that appears strictly as plain, left-aligned text with a '○' bullet.
    The horizontal three-dots (⋯) are hidden by default and appear on row hover.
    Clicking ⋯ opens a clean, simple text context menu.
    """
    s_id = s["id"]
    is_active = (s_id == active_chat_id)
    title = s.get("title", "New Chat")

    # Clean plain title without any bullet prefix matching user's instruction
    display_title = title

    # Row layout: Title button with absolute floating three dots overlay
    container_key = f"chat_item_{'active_' if is_active else ''}{key_prefix}_{s_id}"
    with st.container(key=container_key):
        col_title, col_dots = st.columns([0.88, 0.12])

        with col_title:
            btn_type = "primary" if is_active else "secondary"
            st.button(
                display_title,
                key=f"btn_chat_{key_prefix}_{s_id}",
                type=btn_type,
                use_container_width=True,
                on_click=_cb_select_chat,
                args=(s_id,),
            )

        with col_dots:
            # Horizontal Three-Dots (⋯) Popover Menu
            with st.popover("⋯"):
                # 1. Open in new window (ONLY link that opens in a new tab)
                st.markdown(
                    f"""
                    <a href="?chat_id={s_id}" target="_blank" class="claude-menu-link">
                        <span class="item-left">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                                <polyline points="15 3 21 3 21 9"></polyline>
                                <line x1="10" y1="14" x2="21" y2="3"></line>
                            </svg>
                            <span>Open in new window</span>
                        </span>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

                # 2. Pin / Unpin (Single-pass instant update)
                is_pinned = s.get("pinned", False)
                pin_lbl = "Unpin" if is_pinned else "Pin"
                st.button(
                    f"📌 {pin_lbl}",
                    key=f"opt_pin_{key_prefix}_{s_id}",
                    use_container_width=True,
                    on_click=_cb_toggle_pin,
                    args=(s_id,),
                )

                # 3. Rename (Single-pass instant modal)
                st.button(
                    "✏️ Rename",
                    key=f"opt_ren_{key_prefix}_{s_id}",
                    use_container_width=True,
                    on_click=_cb_trigger_rename,
                    args=(s_id,),
                )

                # 4. Add to project
                st.button(
                    "📦 Add to project",
                    key=f"opt_proj_{key_prefix}_{s_id}",
                    use_container_width=True,
                    on_click=_cb_trigger_move,
                    args=(s_id,),
                )

                # 5. Move to group
                st.button(
                    "📁 Move to group",
                    key=f"opt_mov_{key_prefix}_{s_id}",
                    use_container_width=True,
                    on_click=_cb_trigger_move,
                    args=(s_id,),
                )

                st.markdown('<div class="popover-divider"></div>', unsafe_allow_html=True)

                # 6. Delete (Single-pass instant delete)
                st.button(
                    "🗑️ Delete",
                    key=f"opt_del_{key_prefix}_{s_id}",
                    use_container_width=True,
                    on_click=_cb_delete_chat,
                    args=(s_id,),
                )


def run_streamlit_app():
    """SanviBot interface matching reference screenshot."""
    st.set_page_config(
        page_title="SanviBot | AI Assistant",
        page_icon="✨",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Reference Screenshot-Matched Pixel-Perfect CSS
    sanvibot_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color-scheme: dark;
        }

        /* Streamlit Header adjustments */
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 2.2rem !important;
            z-index: 10 !important;
        }

        /* Main content container */
        .block-container {
            max-width: 860px !important;
            padding-top: 3.2rem !important;
            padding-bottom: 6.5rem !important;
        }

        /* Top header bar */
        .sanvibot-topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.6rem 0.2rem 1rem;
            margin-top: 0.2rem;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid rgba(128, 128, 128, 0.16);
        }

        .sanvibot-title {
            font-size: 1.15rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            letter-spacing: -0.01em;
            color: #f4f4f5;
        }

        .model-pill {
            font-size: 0.74rem;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 9999px;
            background: rgba(99, 102, 241, 0.12);
            color: #818cf8;
            border: 1px solid rgba(99, 102, 241, 0.25);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Sidebar Container */
        [data-testid="stSidebar"] {
            background-color: #17171a !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding: 0.6rem 0.7rem 1rem 0.7rem !important;
        }

        /* Tighten gap between sidebar blocks - eliminates extra lines/spaces */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 2px !important;
        }

        /* Chat History Navigation Bar: ‹  › Buttons (placed right above Groups / Ungrouped history) */
        div[class*="st-key-sidebar_nav"] {
            margin-top: 8px !important;
            margin-bottom: 0 !important;
        }
        div[class*="st-key-sidebar_nav"] [data-testid="stHorizontalBlock"],
        .sidebar-nav-container [data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            gap: 4px !important;
            padding: 0 !important;
            margin: 0 !important;
            min-height: unset !important;
            background: transparent !important;
        }
        div[class*="st-key-sidebar_nav"] [data-testid="stHorizontalBlock"] > div,
        .sidebar-nav-container [data-testid="stHorizontalBlock"] > div {
            position: static !important;
            flex: 0 0 28px !important;
            width: 28px !important;
            min-width: 28px !important;
            max-width: 28px !important;
            padding: 0 !important;
            margin: 0 !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }
        div[class*="st-key-sidebar_nav"] [data-testid="stHorizontalBlock"] > div:last-child,
        .sidebar-nav-container [data-testid="stHorizontalBlock"] > div:last-child {
            flex: 1 1 auto !important;
            width: auto !important;
            max-width: none !important;
        }
        div[class*="st-key-nav_arrow_"] button,
        [data-testid="stSidebar"] div[class*="st-key-nav_arrow_"] button,
        [data-testid="stSidebar"] button[key="nav_arrow_prev"],
        [data-testid="stSidebar"] button[key="nav_arrow_next"] {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            color: #a1a1aa !important;
            font-size: 1.25rem !important;
            line-height: 1 !important;
            font-weight: 500 !important;
            width: 28px !important;
            height: 28px !important;
            min-height: 28px !important;
            max-height: 28px !important;
            padding: 0 !important;
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 6px !important;
            cursor: pointer !important;
            transition: background-color 0.15s ease, color 0.15s ease !important;
        }
        div[class*="st-key-nav_arrow_"] button:hover,
        [data-testid="stSidebar"] div[class*="st-key-nav_arrow_"] button:hover {
            background-color: rgba(255, 255, 255, 0.12) !important;
            color: #ffffff !important;
        }
        div[class*="st-key-nav_arrow_"] button p,
        [data-testid="stSidebar"] div[class*="st-key-nav_arrow_"] button p {
            font-size: 1.25rem !important;
            line-height: 1 !important;
            margin: 0 !important;
            padding: 0 !important;
            color: inherit !important;
        }

        /* + New Button */
        div[class*="st-key-btn_new_chat"] button,
        div[class*="st-key-new_chat_btn_container"] button,
        .new-chat-btn button {
            background: transparent !important;
            background-color: transparent !important;
            color: #e4e4e7 !important;
            border: 1px solid rgba(255, 255, 255, 0.14) !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
            padding: 0.4rem 0.75rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            gap: 8px !important;
            font-size: 0.86rem !important;
            transition: all 0.15s ease !important;
            width: 100% !important;
            margin-bottom: 4px !important;
        }
        div[class*="st-key-btn_new_chat"] button:hover,
        div[class*="st-key-new_chat_btn_container"] button:hover,
        .new-chat-btn button:hover {
            background: rgba(255, 255, 255, 0.08) !important;
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: #ffffff !important;
            border-color: rgba(255, 255, 255, 0.25) !important;
        }
        div[class*="st-key-btn_new_chat"] button p,
        div[class*="st-key-new_chat_btn_container"] button p,
        .new-chat-btn button p {
            font-size: 0.86rem !important;
            color: inherit !important;
            text-align: left !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Search Input */
        [data-testid="stSidebar"] div[data-testid="stTextInput"] {
            margin-bottom: 4px !important;
        }
        [data-testid="stSidebar"] div[data-testid="stTextInput"] input {
            background-color: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
            color: #e4e4e7 !important;
            font-size: 0.84rem !important;
            padding: 0.35rem 0.7rem !important;
        }
        [data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus {
            border-color: rgba(255, 255, 255, 0.25) !important;
            background-color: rgba(255, 255, 255, 0.07) !important;
        }

        /* Group Headings: matching chat item vertical height and spacing (same distance as between chats) */
        div[class*="st-key-group_header_"],
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(div[class*="st-key-group_header_"]) {
            margin-top: 2px !important;
            margin-bottom: 1px !important;
            height: 28px !important;
            min-height: 28px !important;
            max-height: 28px !important;
            display: flex !important;
            align-items: center !important;
            padding: 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(div[class*="st-key-chat_item_"]) + [data-testid="stElementContainer"]:has(div[class*="st-key-group_header_"]) {
            margin-top: 12px !important;
        }
        div[class*="st-key-group_header_"] [data-testid="stHorizontalBlock"],
        .group-header-row [data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: space-between !important;
            padding: 0 4px !important;
            margin: 0 !important;
            height: 28px !important;
            min-height: 28px !important;
            max-height: 28px !important;
            background: transparent !important;
        }
        div[class*="st-key-group_header_"] [data-testid="stHorizontalBlock"] > div:first-child,
        .group-header-row [data-testid="stHorizontalBlock"] > div:first-child {
            flex: 1 1 auto !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        div[class*="st-key-group_header_"] [data-testid="stHorizontalBlock"] > div:last-child,
        .group-header-row [data-testid="stHorizontalBlock"] > div:last-child {
            position: static !important;
            flex: 0 0 22px !important;
            width: 22px !important;
            min-width: 22px !important;
            max-width: 22px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            margin: 0 !important;
            opacity: 0.85 !important;
            pointer-events: auto !important;
        }
        div[class*="st-key-group_header_"]:hover [data-testid="stHorizontalBlock"] > div:last-child,
        .group-header-row:hover [data-testid="stHorizontalBlock"] > div:last-child {
            opacity: 1 !important;
        }
        div[class*="st-key-group_header_"] [data-testid="stPopover"] button,
        .group-header-row [data-testid="stPopover"] button {
            background: transparent !important;
            background-color: transparent !important;
            color: #a1a1aa !important;
            font-size: 1.1rem !important;
            width: 22px !important;
            height: 22px !important;
            min-height: 22px !important;
            border-radius: 4px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            box-shadow: none !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
        }
        div[class*="st-key-group_header_"] [data-testid="stPopover"] button:hover,
        .group-header-row [data-testid="stPopover"] button:hover {
            background: rgba(255, 255, 255, 0.12) !important;
            background-color: rgba(255, 255, 255, 0.12) !important;
            color: #ffffff !important;
        }

        .simple-group-header {
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            color: #ffffff !important;
            font-size: 0.86rem !important;
            font-weight: 600 !important;
            height: 28px !important;
            line-height: 28px !important;
            padding: 0 4px !important;
            margin: 0 !important;
            letter-spacing: -0.01em !important;
            user-select: none !important;
        }
        .simple-group-header .group-title {
            display: flex !important;
            align-items: center !important;
            gap: 6px !important;
        }

        /* Sidebar Model Selection */
        [data-testid="stSidebar"] div[data-testid="stSelectbox"] {
            margin-bottom: 6px !important;
        }
        [data-testid="stSidebar"] div[data-testid="stSelectbox"] > div {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 8px !important;
            color: #e4e4e7 !important;
            font-size: 0.86rem !important;
        }
        [data-testid="stSidebar"] div[data-testid="stSelectbox"] > div:hover {
            border-color: rgba(255, 255, 255, 0.25) !important;
        }

        /* Voice input microphone button before send button */
        #voice-mic-btn {
            background: transparent !important;
            border: none !important;
            outline: none !important;
            cursor: pointer !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 32px !important;
            height: 32px !important;
            min-width: 32px !important;
            min-height: 32px !important;
            padding: 0 !important;
            margin: 0 4px 0 0 !important;
            border-radius: 50% !important;
            color: #a1a1aa !important;
            transition: all 0.15s ease !important;
            flex-shrink: 0 !important;
            align-self: center !important;
        }
        #voice-mic-btn:hover {
            color: #ffffff !important;
            background: rgba(255, 255, 255, 0.12) !important;
        }
        #voice-mic-btn.recording {
            color: #ef4444 !important;
            background: rgba(239, 68, 68, 0.18) !important;
            animation: micPulse 1.2s infinite ease-in-out !important;
        }
        @keyframes micPulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.18); opacity: 0.7; }
        }

        /* Hidden bridge iframe for voice script */
        iframe[title*="streamlit.components"] {
            height: 0px !important;
            min-height: 0px !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            position: absolute !important;
            visibility: hidden !important;
        }

        /* =========================================================
           CRITICAL: SIDEBAR PRIMARY BUTTON OVERRIDE (NEVER SHOW RED)
           ========================================================= */
        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
        [data-testid="stSidebar"] button[kind="primary"]:hover,
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stSidebar"] button[kind="primary"]:active,
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:active,
        [data-testid="stSidebar"] button[kind="primary"]:focus,
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:focus {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            color: #ffffff !important;
        }

        /* =========================================================
           CHAT ITEM ROWS (CLEAN, TEXT-BASED, STRICT LEFT-ALIGNED)
           Compact Claude Replica - Eliminates Extra Blank Lines
           ========================================================= */
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(div[class*="st-key-chat_item_"]),
        div[class*="st-key-chat_item_"],
        .chat-item-row {
            position: relative !important;
            width: 100% !important;
            margin: 1px 0 !important;
            padding: 0 !important;
            height: 28px !important;
            min-height: 28px !important;
            max-height: 28px !important;
        }

        div[class*="st-key-chat_item_"] [data-testid="stColumn"],
        div[class*="st-key-chat_item_"] [data-testid="stColumn"] [data-testid="stVerticalBlock"],
        div[class*="st-key-chat_item_"] [data-testid="stColumn"] [data-testid="stElementContainer"] {
            padding: 0 !important;
            margin: 0 !important;
            gap: 0 !important;
            min-height: 28px !important;
            height: 28px !important;
            max-height: 28px !important;
        }

        div[class*="st-key-chat_item_"] [data-testid="stHorizontalBlock"],
        .chat-item-row [data-testid="stHorizontalBlock"],
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) {
            position: relative !important;
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 0px !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 6px !important;
            transition: background 0.1s ease !important;
            width: 100% !important;
            min-height: 28px !important;
            height: 28px !important;
            max-height: 28px !important;
            box-sizing: border-box !important;
            cursor: pointer !important;
            overflow: hidden !important;
        }

        /* Inactive row hover */
        div[class*="st-key-chat_item_"] [data-testid="stHorizontalBlock"]:hover,
        .chat-item-row [data-testid="stHorizontalBlock"]:hover,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]):hover {
            background: rgba(255, 255, 255, 0.07) !important;
        }

        /* Active Chat Row: Soft dark rounded rectangle #38383a */
        div[class*="st-key-chat_item_active_"] [data-testid="stHorizontalBlock"],
        div[class*="st-key-chat_item_"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]),
        div[class*="st-key-chat_item_"] [data-testid="stHorizontalBlock"]:has([data-testid="stBaseButton-primary"]),
        .chat-item-row [data-testid="stHorizontalBlock"]:has(button[kind="primary"]),
        .chat-item-row [data-testid="stHorizontalBlock"]:has([data-testid="stBaseButton-primary"]),
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]),
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="stBaseButton-primary"]) {
            background: #38383a !important;
            background-color: #38383a !important;
            border-radius: 6px !important;
        }

        /* First column: Title button container */
        div[class*="st-key-chat_item_"] [data-testid="stHorizontalBlock"] > div:first-child,
        .chat-item-row [data-testid="stHorizontalBlock"] > div:first-child,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) > div:first-child {
            flex: 1 1 100% !important;
            min-width: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            padding: 0 !important;
            padding-right: 24px !important;
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
        }

        /* Strict Left Alignment on Title Button */
        div[class*="st-key-chat_item_"] .stButton,
        div[class*="st-key-chat_item_"] [data-testid="stButton"],
        .chat-item-row [data-testid="stHorizontalBlock"] .stButton,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) .stButton {
            width: 100% !important;
            display: flex !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div[class*="st-key-chat_item_"] button,
        div[class*="st-key-chat_item_"] [data-testid="stBaseButton-secondary"],
        div[class*="st-key-chat_item_"] [data-testid="stBaseButton-primary"],
        .chat-item-row [data-testid="stHorizontalBlock"] button,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) button {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            color: #d1d1d6 !important;
            font-size: 0.85rem !important;
            font-weight: 400 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            align-items: center !important;
            padding: 0 8px !important;
            margin: 0 !important;
            min-height: unset !important;
            height: 28px !important;
            line-height: 28px !important;
            width: 100% !important;
            display: flex !important;
            flex-direction: row !important;
            border-radius: 6px !important;
            cursor: pointer !important;
        }

        div[class*="st-key-chat_item_"] button div,
        div[class*="st-key-chat_item_"] button span,
        div[class*="st-key-chat_item_"] button [data-testid="stMarkdownContainer"],
        .chat-item-row [data-testid="stHorizontalBlock"] button div,
        .chat-item-row [data-testid="stHorizontalBlock"] button span,
        .chat-item-row [data-testid="stHorizontalBlock"] button [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) button div,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) button span,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) button [data-testid="stMarkdownContainer"] {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        div[class*="st-key-chat_item_"] button p,
        .chat-item-row [data-testid="stHorizontalBlock"] button p,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) button p {
            display: block !important;
            text-align: left !important;
            width: 100% !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            color: #d1d1d6 !important;
            font-size: 0.85rem !important;
            font-weight: 400 !important;
            line-height: 28px !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        div[class*="st-key-chat_item_active_"] button p,
        div[class*="st-key-chat_item_"] [data-testid="stBaseButton-primary"] p,
        div[class*="st-key-chat_item_"] button[kind="primary"] p,
        .chat-item-row [data-testid="stHorizontalBlock"]:has(button[kind="primary"]) button p,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]) button p {
            color: #ffffff !important;
            font-weight: 500 !important;
        }

        /* Second column: Three dots (Positioned absolute on the right edge) */
        div[class*="st-key-chat_item_"] [data-testid="stHorizontalBlock"] > div:last-child,
        .chat-item-row [data-testid="stHorizontalBlock"] > div:last-child,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) > div:last-child {
            position: absolute !important;
            right: 4px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            flex: 0 0 20px !important;
            width: 20px !important;
            max-width: 20px !important;
            min-width: 20px !important;
            height: 20px !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            padding: 0 !important;
            margin: 0 !important;
            z-index: 10 !important;
            opacity: 0 !important;
            pointer-events: none !important;
            transition: opacity 0.12s ease !important;
        }

        div[class*="st-key-chat_item_"]:hover [data-testid="stHorizontalBlock"] > div:last-child,
        div[class*="st-key-chat_item_"] [data-testid="stHorizontalBlock"]:hover > div:last-child,
        div[class*="st-key-chat_item_"] [data-testid="stHorizontalBlock"]:focus-within > div:last-child,
        div[class*="st-key-chat_item_"] [data-testid="stHorizontalBlock"]:has([aria-expanded="true"]) > div:last-child,
        .chat-item-row:hover [data-testid="stHorizontalBlock"] > div:last-child,
        .chat-item-row [data-testid="stHorizontalBlock"]:hover > div:last-child,
        .chat-item-row [data-testid="stHorizontalBlock"]:focus-within > div:last-child,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]):hover > div:last-child,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]):focus-within > div:last-child,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]):has([aria-expanded="true"]) > div:last-child {
            opacity: 1 !important;
            pointer-events: auto !important;
        }

        div[class*="st-key-chat_item_"] [data-testid="stPopover"],
        .chat-item-row [data-testid="stPopover"],
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) [data-testid="stPopover"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        div[class*="st-key-chat_item_"] [data-testid="stPopover"] button,
        .chat-item-row [data-testid="stPopover"] button,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) [data-testid="stPopover"] button {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            color: #9ca3af !important;
            font-size: 1.15rem !important;
            padding: 0 !important;
            margin: 0 !important;
            min-height: unset !important;
            height: 22px !important;
            width: 22px !important;
            min-width: 22px !important;
            line-height: 1 !important;
            border-radius: 4px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        div[class*="st-key-chat_item_"] [data-testid="stPopover"] button:hover,
        .chat-item-row [data-testid="stPopover"] button:hover,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) [data-testid="stPopover"] button:hover {
            color: #ffffff !important;
            background: rgba(255, 255, 255, 0.12) !important;
            background-color: rgba(255, 255, 255, 0.12) !important;
        }

        /* Explicitly format inner wrappers, markdown container, and p for the three dots so they are never clipped or pushed */
        div[class*="st-key-chat_item_"] [data-testid="stPopover"] button div:not([aria-hidden="true"]),
        div[class*="st-key-chat_item_"] [data-testid="stPopover"] button span:not([aria-hidden="true"]),
        div[class*="st-key-chat_item_"] [data-testid="stPopover"] button [data-testid="stMarkdownContainer"],
        .chat-item-row [data-testid="stPopover"] button div:not([aria-hidden="true"]),
        .chat-item-row [data-testid="stPopover"] button span:not([aria-hidden="true"]),
        .chat-item-row [data-testid="stPopover"] button [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) [data-testid="stPopover"] button div:not([aria-hidden="true"]),
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) [data-testid="stPopover"] button span:not([aria-hidden="true"]),
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) [data-testid="stPopover"] button [data-testid="stMarkdownContainer"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            height: 100% !important;
            line-height: 1 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        div[class*="st-key-chat_item_"] [data-testid="stPopover"] button p,
        .chat-item-row [data-testid="stPopover"] button p,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) [data-testid="stPopover"] button p {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            width: 100% !important;
            height: 100% !important;
            color: #9ca3af !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            line-height: 1 !important;
            letter-spacing: 0.5px !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
            white-space: nowrap !important;
        }
        div[class*="st-key-chat_item_"] [data-testid="stPopover"] button:hover p,
        .chat-item-row [data-testid="stPopover"] button:hover p,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) [data-testid="stPopover"] button:hover p {
            color: #ffffff !important;
        }

        /* Completely hide the chevron down-arrow in stPopover across all popovers (must beat chat_item button div specificity) */
        div[class*="st-key-chat_item_"] [data-testid="stPopover"] button [aria-hidden="true"],
        .chat-item-row [data-testid="stPopover"] button [aria-hidden="true"],
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) [data-testid="stPopover"] button [aria-hidden="true"],
        [data-testid="stSidebar"] [data-testid="stPopover"] [aria-hidden="true"],
        [data-testid="stSidebar"] [data-testid="stPopoverButton"] [aria-hidden="true"],
        [data-testid="stSidebar"] [data-testid="stPopover"] [data-testid="stIconMaterial"],
        [data-testid="stSidebar"] [data-testid="stPopover"] svg,
        [data-testid="stSidebar"] [data-testid="stPopover"] span:has(svg) {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            max-width: 0 !important;
            min-width: 0 !important;
            height: 0 !important;
            flex: 0 0 0px !important;
            overflow: hidden !important;
        }

        /* =========================================================
           CONTEXT MENU POPOVER CONTAINER (CLAUDE UI REPLICA)
           Strict Left-Aligned, Tight Vertical Spacing, No Gaps
           ========================================================= */
        div[data-testid="stPopoverBody"] {
            background-color: #1e1e21 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            padding: 4px !important;
            box-shadow: 0 10px 32px rgba(0, 0, 0, 0.75) !important;
            min-width: 190px !important;
            max-width: 220px !important;
        }

        /* Eliminate the vertical gap between items inside the popover */
        div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
            gap: 2px !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div[data-testid="stPopoverBody"] .stButton,
        div[data-testid="stPopoverBody"] [data-testid="stButton"] {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
        }

        /* All popover menu buttons styled as flat sleek items with zero refresh */
        div[data-testid="stPopoverBody"] button,
        div[data-testid="stPopoverBody"] [data-testid="stBaseButton-secondary"],
        div[data-testid="stPopoverBody"] [data-testid="stBaseButton-primary"] {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            color: #e4e4e7 !important;
            font-size: 0.85rem !important;
            font-weight: 400 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            align-items: center !important;
            padding: 4px 8px !important;
            margin: 0 !important;
            min-height: unset !important;
            height: 28px !important;
            line-height: 28px !important;
            width: 100% !important;
            border-radius: 6px !important;
            cursor: pointer !important;
            display: flex !important;
            flex-direction: row !important;
            transition: background-color 0.1s ease, color 0.1s ease !important;
        }

        div[data-testid="stPopoverBody"] button:hover,
        div[data-testid="stPopoverBody"] [data-testid="stBaseButton-secondary"]:hover {
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: #ffffff !important;
        }

        /* Force inner wrappers to strictly left-align */
        div[data-testid="stPopoverBody"] button > div,
        div[data-testid="stPopoverBody"] button div,
        div[data-testid="stPopoverBody"] button span,
        div[data-testid="stPopoverBody"] button [data-testid="stMarkdownContainer"] {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        div[data-testid="stPopoverBody"] button p {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
            width: 100% !important;
            gap: 8px !important;
            font-size: 0.85rem !important;
            font-weight: 400 !important;
            color: inherit !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 28px !important;
        }

        .claude-menu-link {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            gap: 8px !important;
            padding: 4px 8px !important;
            margin: 0 !important;
            height: 28px !important;
            box-sizing: border-box !important;
            color: #e4e4e7 !important;
            font-size: 0.85rem !important;
            font-weight: 400 !important;
            text-decoration: none !important;
            border-radius: 6px !important;
            transition: background-color 0.1s ease, color 0.1s ease !important;
            width: 100% !important;
        }
        .claude-menu-link:hover {
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: #ffffff !important;
            text-decoration: none !important;
        }
        .claude-menu-link .item-left {
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            color: inherit !important;
        }
        .claude-menu-link svg {
            color: #9ca3af !important;
            flex-shrink: 0 !important;
        }
        .claude-menu-link:hover svg {
            color: #ffffff !important;
        }

        /* Line separator in popover menus - tight symmetrical spacing */
        div[data-testid="stPopoverBody"] .popover-divider,
        div[data-testid="stPopoverBody"] [data-testid="stDivider"],
        div[data-testid="stPopoverBody"] [data-testid="stDivider"] hr,
        div[data-testid="stPopoverBody"] hr {
            height: 1px !important;
            min-height: 1px !important;
            max-height: 1px !important;
            border: none !important;
            background: rgba(255, 255, 255, 0.1) !important;
            margin: 3px 0 !important;
            margin-top: 3px !important;
            margin-bottom: 3px !important;
            padding: 0 !important;
            width: 100% !important;
            display: block !important;
            line-height: 1px !important;
        }

        div[data-testid="stPopoverBody"] [data-testid="stElementContainer"]:has(.popover-divider),
        div[data-testid="stPopoverBody"] [data-testid="stElementContainer"]:has([data-testid="stDivider"]),
        div[data-testid="stPopoverBody"] [data-testid="stElementContainer"]:has(hr) {
            margin: 3px 0 !important;
            margin-top: 3px !important;
            margin-bottom: 3px !important;
            padding: 0 !important;
            height: 1px !important;
            min-height: 1px !important;
            max-height: 1px !important;
            line-height: 1px !important;
        }

        /* Ensure all element containers inside popover body have zero excess margin */
        div[data-testid="stPopoverBody"] [data-testid="stElementContainer"] {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }

        /* Delete button container without extra spacing */
        div[data-testid="stPopoverBody"] [data-testid="stElementContainer"]:has(div[class*="opt_del"]),
        div[data-testid="stPopoverBody"] [data-testid="stElementContainer"]:has(div[class*="fld_del"]) {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            min-height: unset !important;
        }
        div[data-testid="stPopoverBody"] div[class*="opt_del"],
        div[data-testid="stPopoverBody"] div[class*="fld_del"] {
            /* Breathing room between the separator line and Delete */
            margin-top: 9px !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            min-height: unset !important;
        }

        /* "Open in new window" row: reserve the full row height so Pin does not overlap it */
        div[data-testid="stPopoverBody"] [data-testid="stElementContainer"]:has(.claude-menu-link),
        div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"]:has(.claude-menu-link) {
            height: 28px !important;
            min-height: 28px !important;
        }

        /* Delete button red highlight */
        div[data-testid="stPopoverBody"] div[class*="opt_del"] button,
        div[data-testid="stPopoverBody"] div[class*="fld_del"] button {
            color: #ef4444 !important;
        }
        div[data-testid="stPopoverBody"] div[class*="opt_del"] button p,
        div[data-testid="stPopoverBody"] div[class*="fld_del"] button p {
            color: #ef4444 !important;
        }
        div[data-testid="stPopoverBody"] div[class*="opt_del"] button:hover,
        div[data-testid="stPopoverBody"] div[class*="fld_del"] button:hover {
            background-color: rgba(239, 68, 68, 0.14) !important;
            color: #f87171 !important;
        }
        div[data-testid="stPopoverBody"] div[class*="opt_del"] button:hover p,
        div[data-testid="stPopoverBody"] div[class*="fld_del"] button:hover p {
            color: #f87171 !important;
        }

        /* Cooking indicator */
        .cooking-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            color: #d1d1d6;
            font-size: 0.90rem;
            font-weight: 500;
            margin: 4px 0;
            animation: cookingPulse 1.4s ease-in-out infinite;
        }
        @keyframes cookingPulse {
            0%, 100% { opacity: 0.6; transform: translateY(0); }
            50% { opacity: 1; transform: translateY(-1px); }
        }
        .cooking-spinner {
            font-size: 1rem;
            line-height: 1;
        }
        .cooking-text {
            color: #e4e4e7;
            font-family: inherit;
            letter-spacing: -0.01em;
        }

        /* User Message Bubble on Right (Claude Style) */
        .user-bubble-wrapper {
            display: flex;
            justify-content: flex-end;
            width: 100%;
            margin-bottom: 1.25rem;
        }
        .user-bubble-wrapper [data-testid="stChatMessage"] {
            background-color: #27272a !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 18px !important;
            padding: 0.85rem 1.25rem !important;
            max-width: 82% !important;
            margin-left: auto !important;
            color: #f4f4f5 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        }
        .user-bubble-wrapper [data-testid="stChatMessageAvatar"] {
            display: none !important;
        }

        /* Assistant Message on Left (Claude Style) */
        .bot-bubble-wrapper {
            margin-bottom: 1.5rem;
        }
        .bot-bubble-wrapper [data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            padding: 0.4rem 0 !important;
            color: #f4f4f5 !important;
            font-size: 0.98rem !important;
            line-height: 1.68 !important;
        }
        .bot-bubble-wrapper [data-testid="stChatMessageAvatar"] {
            display: none !important;
        }

        /* Centered New Chat State */
        .hero-center-box {
            margin-top: 15vh;
            margin-top: 15dvh;
            margin-bottom: 2rem;
            text-align: center;
        }
        .hero-sparkle {
            font-size: 2.5rem;
            margin-bottom: 0.6rem;
        }
        .hero-greeting {
            font-size: clamp(1.4rem, 1rem + 2vw, 1.95rem); /* 1.95rem on desktop; fluidly scales on mobile */
            font-weight: 700;
            color: #f4f4f5;
            letter-spacing: -0.02em;
            margin-bottom: 0.4rem;
        }
        .hero-subtitle {
            color: #71717a;
            font-size: 0.98rem;
            margin-bottom: 1.8rem;
        }

        /* Suppress any auto-generated heading anchor links */
        .hero-greeting a,
        .hero-center-box a,
        a.anchor-link {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        /* Centered Form Styling */
        [data-testid="stForm"] {
            border: none !important;
            background: transparent !important;
            padding: 0 !important;
        }
        [data-testid="stForm"] input {
            background-color: #212124 !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            border-radius: 28px !important;
            padding: 0.85rem 1.4rem !important;
            font-size: 1rem !important;
            color: #ffffff !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        }
        [data-testid="stForm"] button {
            border-radius: 28px !important;
            padding: 0.75rem 1.25rem !important;
            font-weight: 600 !important;
        }

        /* Floating Scroll-to-Bottom Button (↓) */
        .scroll-down-fab {
            position: fixed;
            bottom: 120px; /* fallback; exact position is aligned above the input by JS */
            right: calc(50% - 410px);
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: #27272a;
            color: #a1a1aa;
            border: 1px solid rgba(255, 255, 255, 0.14);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 15px;
            z-index: 999;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            transition: all 0.2s ease;
        }
        .scroll-down-fab:hover {
            background: #3f3f46;
            color: #ffffff;
            transform: translateY(-2px);
        }
        @media (max-width: 900px) {
            .scroll-down-fab {
                right: 20px;
            }
        }

        /* Bottom chat input styling */
        [data-testid="stChatInput"] {
            border-radius: 26px !important;
        }

        /* Bottom disclaimer caption */
        .bottom-disclaimer {
            text-align: center;
            font-size: 0.74rem;
            color: #71717a;
            margin-top: 0.4rem;
        }

        /* Prevent long URLs, code blocks, or tables from overflowing chat bubbles on any screen */
        [data-testid="stChatMessage"] {
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
        }
        [data-testid="stChatMessage"] pre {
            overflow-x: auto !important;
            max-width: 100% !important;
        }
        [data-testid="stChatMessage"] table {
            display: block !important;
            overflow-x: auto !important;
            max-width: 100% !important;
        }

        /* =========================================================
           MOBILE & TOUCH RESPONSIVENESS (ZERO DESKTOP REGRESSION)
           ========================================================= */
        @media (hover: none), (pointer: coarse) {
            /* Always show three-dots menu trigger on touch devices where hover does not exist */
            div[class*="st-key-chat_item_"] [data-testid="stHorizontalBlock"] > div:last-child,
            .chat-item-row [data-testid="stHorizontalBlock"] > div:last-child,
            [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.stButton):has([data-testid="stPopover"]) > div:last-child {
                opacity: 0.85 !important;
                pointer-events: auto !important;
            }
        }

        @media (max-width: 768px) {
            .block-container {
                max-width: 100% !important;
                padding-top: 2.3rem !important;
                padding-left: 0.85rem !important;
                padding-right: 0.85rem !important;
                padding-bottom: 5.5rem !important;
            }

            .sanvibot-topbar-left {
                padding-top: 0.1rem !important;
            }
            .sanvibot-title {
                font-size: 1.02rem !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                max-width: 82vw !important;
            }

            /* Compact hero section on mobile so greeting, input, and chips fit comfortably */
            .hero-center-box {
                margin-top: 4vh !important;
                margin-top: 4dvh !important;
                margin-bottom: 1.1rem !important;
            }
            .hero-sparkle {
                font-size: 2rem !important;
                margin-bottom: 0.35rem !important;
            }
            .hero-subtitle {
                font-size: 0.88rem !important;
                margin-bottom: 1rem !important;
                padding: 0 0.5rem !important;
            }

            /* Collapse empty desktop spacer columns in message rows on mobile */
            [data-testid="stHorizontalBlock"]:has(.user-bubble-wrapper) > [data-testid="stColumn"]:first-child,
            [data-testid="stHorizontalBlock"]:has(.bot-bubble-wrapper) > [data-testid="stColumn"]:last-child {
                display: none !important;
            }
            [data-testid="stHorizontalBlock"]:has(.user-bubble-wrapper) > [data-testid="stColumn"]:last-child,
            [data-testid="stHorizontalBlock"]:has(.bot-bubble-wrapper) > [data-testid="stColumn"]:first-child {
                width: 100% !important;
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }

            .user-bubble-wrapper {
                margin-bottom: 0.9rem !important;
            }
            .user-bubble-wrapper [data-testid="stChatMessage"] {
                max-width: 90% !important;
                padding: 0.7rem 1rem !important;
                font-size: 0.94rem !important;
            }
            .bot-bubble-wrapper {
                margin-bottom: 1.1rem !important;
            }
            .bot-bubble-wrapper [data-testid="stChatMessage"] {
                font-size: 0.94rem !important;
                line-height: 1.6 !important;
            }

            /* Prevent Streamlit from stacking sidebar horizontal rows (‹ ›, group headers, chat items) on <640px */
            [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
                flex-direction: row !important;
                flex-wrap: nowrap !important;
            }
            div[class*="st-key-sidebar_nav"] [data-testid="stHorizontalBlock"] > div {
                flex: 0 0 28px !important;
                width: 28px !important;
                min-width: 28px !important;
                max-width: 28px !important;
            }
            div[class*="st-key-group_header_"] [data-testid="stHorizontalBlock"] > div:first-child {
                min-width: 0 !important;
                width: auto !important;
            }
            div[class*="st-key-group_header_"] [data-testid="stHorizontalBlock"] > div:last-child {
                flex: 0 0 22px !important;
                width: 22px !important;
                min-width: 22px !important;
                max-width: 22px !important;
            }

            /* Keep three-dots visible on mobile viewports */
            div[class*="st-key-chat_item_"] [data-testid="stHorizontalBlock"] > div:last-child {
                opacity: 0.85 !important;
                pointer-events: auto !important;
            }

            /* 16px input font-size on mobile prevents iOS Safari / Android Chrome forced page zoom on focus */
            [data-testid="stChatInput"] textarea,
            [data-testid="stSidebar"] div[data-testid="stTextInput"] input {
                font-size: 16px !important;
            }

            .scroll-down-fab {
                right: 14px !important;
                width: 30px !important;
                height: 30px !important;
            }
        }
    </style>
    """
    st.markdown(sanvibot_css, unsafe_allow_html=True)

    # Load persistent chat sessions and folder groups
    sessions_data = load_sessions_data()
    sessions = sessions_data["sessions"]
    all_groups = sessions_data.get("groups", ["Sanvi"])

    # Display any queued toast from single-pass callbacks
    if st.session_state.get("pending_toast"):
        t_msg, t_icon = st.session_state.pop("pending_toast")
        st.toast(t_msg, icon=t_icon)

    # Check for modal triggers
    if st.session_state.get("trigger_rename_id"):
        target_id = st.session_state.pop("trigger_rename_id")
        show_rename_modal(target_id, sessions_data)

    if st.session_state.get("trigger_move_id"):
        target_id = st.session_state.pop("trigger_move_id")
        show_move_group_modal(target_id, sessions_data, all_groups)

    if st.session_state.get("trigger_rename_folder"):
        fld_name = st.session_state.pop("trigger_rename_folder")
        show_rename_folder_modal(fld_name, sessions_data, all_groups)

    if st.session_state.get("trigger_delete_folder"):
        fld_name = st.session_state.pop("trigger_delete_folder")
        show_delete_folder_modal(fld_name, sessions_data, all_groups)

    # Check if a chat_id or context menu action was passed in URL query params
    query_params = st.query_params
    if "action" in query_params:
        act = query_params.get("action")
        target_id = query_params.get("target")
        st.query_params.clear()
        if target_id and target_id in sessions:
            if act == "pin":
                sessions[target_id]["pinned"] = not sessions[target_id].get("pinned", False)
                save_sessions_data(sessions_data)
                st.rerun()
            elif act == "rename":
                st.session_state["trigger_rename_id"] = target_id
                st.rerun()
            elif act in ("move", "project"):
                st.session_state["trigger_move_id"] = target_id
                st.rerun()
            elif act == "delete":
                del sessions[target_id]
                if "nav_history" in st.session_state:
                    st.session_state["nav_history"] = [c for c in st.session_state["nav_history"] if c != target_id]
                    st.session_state["nav_index"] = min(st.session_state.get("nav_index", 0), max(0, len(st.session_state["nav_history"]) - 1))
                if sessions_data["active_chat_id"] == target_id:
                    if sessions:
                        sessions_data["active_chat_id"] = next(iter(sessions))
                    else:
                        fresh_id = f"chat_{uuid.uuid4().hex[:8]}"
                        sessions[fresh_id] = {
                            "id": fresh_id,
                            "title": "New Chat",
                            "created_at": time.time(),
                            "updated_at": time.time(),
                            "pinned": False,
                            "group": None,
                            "messages": [],
                        }
                        sessions_data["active_chat_id"] = fresh_id
                save_sessions_data(sessions_data)
                st.toast("Chat deleted", icon="🗑️")
                st.rerun()

    if "chat_id" in query_params and query_params["chat_id"] in sessions:
        sessions_data["active_chat_id"] = query_params["chat_id"]
        st.query_params.clear()
        save_sessions_data(sessions_data)

    active_chat_id = sessions_data["active_chat_id"]

    # Ensure active session exists
    if active_chat_id not in sessions:
        active_chat_id = next(iter(sessions))
        sessions_data["active_chat_id"] = active_chat_id

    # Initialize or synchronize navigation history stack
    if "nav_history" not in st.session_state or not st.session_state["nav_history"]:
        st.session_state["nav_history"] = [active_chat_id]
        st.session_state["nav_index"] = 0
    else:
        idx = st.session_state.get("nav_index", 0)
        hist = st.session_state.get("nav_history", [])
        if not (0 <= idx < len(hist) and hist[idx] == active_chat_id):
            if active_chat_id in hist:
                st.session_state["nav_index"] = hist.index(active_chat_id)
            else:
                hist.append(active_chat_id)
                st.session_state["nav_history"] = hist
                st.session_state["nav_index"] = len(hist) - 1

    # Handle Top-Left Navigation Arrows (‹ / ›)
    if "nav" in query_params:
        nav_dir = query_params.get("nav")
        st.query_params.clear()
        ordered_ids = get_ordered_chat_ids(sessions, all_groups)
        hist = st.session_state.get("nav_history", [active_chat_id])
        idx = st.session_state.get("nav_index", 0)

        target_id = None
        if nav_dir == "prev":
            # 1. Back through visited selection history
            if idx > 0 and (idx - 1) < len(hist) and hist[idx - 1] in sessions:
                idx -= 1
                target_id = hist[idx]
                st.session_state["nav_index"] = idx
            # 2. If at start of selection stack, navigate to previous chat in sidebar list
            elif ordered_ids:
                curr_idx = ordered_ids.index(active_chat_id) if active_chat_id in ordered_ids else 0
                prev_idx = (curr_idx - 1) % len(ordered_ids)
                target_id = ordered_ids[prev_idx]
                st.session_state["nav_history"] = [target_id] + hist
                st.session_state["nav_index"] = 0
        elif nav_dir == "next":
            # 1. Forward through visited selection history
            if (idx + 1) < len(hist) and hist[idx + 1] in sessions:
                idx += 1
                target_id = hist[idx]
                st.session_state["nav_index"] = idx
            # 2. If at end of selection stack, navigate to next chat in sidebar list
            elif ordered_ids:
                curr_idx = ordered_ids.index(active_chat_id) if active_chat_id in ordered_ids else 0
                next_idx = (curr_idx + 1) % len(ordered_ids)
                target_id = ordered_ids[next_idx]
                hist = hist[:idx + 1] + [target_id]
                st.session_state["nav_history"] = hist
                st.session_state["nav_index"] = len(hist) - 1

        if target_id and target_id in sessions and target_id != active_chat_id:
            sessions_data["active_chat_id"] = target_id
            save_sessions_data(sessions_data)
            st.rerun()

    current_session = sessions[active_chat_id]
    current_messages = current_session.get("messages", [])

    # Check API Key
    try:
        client = create_client()
        api_ready = True
    except Exception as e:
        api_ready = False
        api_error_msg = str(e)

    # --- SIDEBAR (MATCHING REFERENCE CLAUDE SCREENSHOT) ---
    with st.sidebar:
        # Model Selection Dropdown
        default_model_idx = AVAILABLE_MODELS.index(st.session_state.get("selected_model", DEFAULT_MODEL)) if st.session_state.get("selected_model", DEFAULT_MODEL) in AVAILABLE_MODELS else 0
        selected_model = st.selectbox(
            "Model",
            options=AVAILABLE_MODELS,
            index=default_model_idx,
            key="sidebar_model_select",
            label_visibility="collapsed",
        )
        st.session_state["selected_model"] = selected_model
        st.markdown("<div style='margin-bottom: 0.35rem;'></div>", unsafe_allow_html=True)

        # 1. "+ New" Button
        with st.container(key="new_chat_btn_container"):
            st.button(
                "➕ New",
                key="btn_new_chat",
                use_container_width=True,
                on_click=_cb_new_chat,
                args=(None,),
            )

        st.markdown("<div style='margin-top: 0.4rem;'></div>", unsafe_allow_html=True)

        # 2. Search Input
        search_query = st.text_input(
            "Search",
            placeholder="🔍 Search...",
            label_visibility="collapsed",
        )

        # 3. Chat History Navigation Arrows: ‹  › (placed after Search and directly above Groups / Ungrouped chats)
        with st.container(key="sidebar_nav"):
            col_arr1, col_arr2, col_arr_space = st.columns([0.14, 0.14, 0.72])
            with col_arr1:
                st.button(
                    "‹",
                    key="nav_arrow_prev",
                    help="Previous chat",
                    on_click=_cb_nav_step,
                    args=("prev",),
                )

            with col_arr2:
                st.button(
                    "›",
                    key="nav_arrow_next",
                    help="Next chat",
                    on_click=_cb_nav_step,
                    args=("next",),
                )

        # --- GROUPS DISPLAYED AS SIMPLE HEADINGS (e.g. "Sanvi ▾" and "Ungrouped ▾") ---
        # 1. Primary Groups (e.g. Sanvi)
        for grp in all_groups:
            grp_chats = [s for s in sessions.values() if s.get("group") == grp]
            grp_chats = sorted(grp_chats, key=lambda s: (not s.get("pinned", False), -s.get("updated_at", 0)))

            if search_query.strip():
                grp_chats = [c for c in grp_chats if search_query.strip().lower() in c.get("title", "").lower()]

            # Simple Text Group Heading with three lines button (≡) on right
            with st.container(key=f"group_header_{grp}"):
                col_grp_title, col_grp_opts = st.columns([0.86, 0.14])
                with col_grp_title:
                    st.markdown(
                        f"""
                        <div class="simple-group-header">
                            <div class="group-title">
                                <span>{grp}</span>
                                <span style="color: #71717a; font-size: 0.75rem;">▾</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_grp_opts:
                    with st.popover("≡", help="Folder options"):
                        st.button(
                            "➕ New chat in folder",
                            key=f"fld_new_{grp}",
                            use_container_width=True,
                            on_click=_cb_new_chat,
                            args=(grp,),
                        )

                        st.button(
                            "✏️ Rename folder",
                            key=f"fld_ren_{grp}",
                            use_container_width=True,
                            on_click=_cb_trigger_rename_folder,
                            args=(grp,),
                        )

                        st.markdown('<div class="popover-divider"></div>', unsafe_allow_html=True)

                        st.button(
                            "🗑️ Delete folder",
                            key=f"fld_del_{grp}",
                            use_container_width=True,
                            on_click=_cb_trigger_delete_folder,
                            args=(grp,),
                        )

            for s in grp_chats:
                render_sidebar_chat_item(
                    s=s,
                    sessions_data=sessions_data,
                    all_groups=all_groups,
                    active_chat_id=active_chat_id,
                    key_prefix=f"grp_{grp}",
                )

        # 2. Ungrouped Chats Section (matching "Ungrouped ▾" in screenshot)
        ungrouped_chats = [s for s in sessions.values() if not s.get("group") or s.get("group") not in all_groups]
        ungrouped_chats = sorted(ungrouped_chats, key=lambda s: (not s.get("pinned", False), -s.get("updated_at", 0)))

        if search_query.strip():
            ungrouped_chats = [c for c in ungrouped_chats if search_query.strip().lower() in c.get("title", "").lower()]

        if ungrouped_chats:
            with st.container(key="group_header_ungrouped"):
                st.markdown(
                    """
                    <div class="simple-group-header">
                        <div class="group-title">
                            <span>Ungrouped</span>
                            <span style="color: #71717a; font-size: 0.75rem;">▾</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            for s in ungrouped_chats:
                render_sidebar_chat_item(
                    s=s,
                    sessions_data=sessions_data,
                    all_groups=all_groups,
                    active_chat_id=active_chat_id,
                    key_prefix="ungrouped",
                )

    # --- TOPBAR ---
    active_title = current_session.get("title", "New Chat")
    st.markdown(
        f"""
        <div class="sanvibot-topbar-left" style="display: flex; align-items: center; gap: 8px; padding-top: 0.2rem;">
            <span class="sanvibot-title" style="font-size: 1.15rem; font-weight: 600; color: #f4f4f5;">{active_title}</span>
            <span style="color: #71717a; font-size: 0.82rem;">▾</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_ready:
        st.error(f"⚠️ **Configuration Error**: {api_error_msg}")
        st.info("Please set `GOOGLE_API_KEY` inside `.env` in the project root directory and reload.")
        return

    # Check if a prompt is pending AI generation
    pending_prompt = st.session_state.get("pending_prompt")

    # --- CASE 1: INITIAL NEW CHAT (VERTICALLY CENTERED IN THE MIDDLE OF PAGE) ---
    if not current_messages and not pending_prompt:
        st.markdown(
            """
            <div class="hero-center-box">
                <div class="hero-sparkle">✨</div>
                <div class="hero-greeting">What can I help with today?</div>
                <p class="hero-subtitle">Ask anything, draft content, summarize documents, or write code.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Centered chat input in the middle of the screen (same structure as active chat: microphone + arrow send button)
        with st.container(key="hero_chat_input_box"):
            st.chat_input(
                "Write a message...",
                key="hero_chat_input",
                on_submit=_cb_submit_prompt,
                kwargs={"state_key": "hero_chat_input"},
            )

        # Suggestion Chips under the centered input box
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.button(
                "💡 Brainstorm creative ideas",
                use_container_width=True,
                on_click=_cb_submit_prompt,
                kwargs={"direct_prompt": "Give me 5 creative ideas for a productive weekend project."},
            )
            st.button(
                "🔍 Summarize key concepts",
                use_container_width=True,
                on_click=_cb_submit_prompt,
                kwargs={"direct_prompt": "Explain the key differences between synchronous and asynchronous programming simply."},
            )

        with col_c2:
            st.button(
                "📝 Help me draft a message",
                use_container_width=True,
                on_click=_cb_submit_prompt,
                kwargs={"direct_prompt": "Help me draft a concise, polite follow-up email to a colleague."},
            )
            st.button(
                "💻 Write & debug code",
                use_container_width=True,
                on_click=_cb_submit_prompt,
                kwargs={"direct_prompt": "Write a Python script to monitor a folder and organize incoming files by extension."},
            )

    # --- CASE 2: ACTIVE CONVERSATION (TOP-DOWN FLOW WITH FIXED BOTTOM INPUT) ---
    else:
        messages_container = st.container()

        # Floating scroll-to-bottom button (↓) matching the reference screenshot
        st.markdown(
            """
            <div class="scroll-down-fab" title="Scroll to bottom">
                ↓
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Fixed bottom-pinned input matching reference screenshot (single-pass on_submit)
        st.chat_input(
            "Write a message...",
            key="main_chat_input",
            on_submit=_cb_submit_prompt,
            kwargs={"state_key": "main_chat_input"},
        )
        st.markdown("<div class='bottom-disclaimer'>SanviBot is AI and can make mistakes. Please double-check responses.</div>", unsafe_allow_html=True)

        with messages_container:
            # If a prompt is pending generation, render previous messages first
            msgs_to_render = current_messages[:-1] if pending_prompt else current_messages
            for msg in msgs_to_render:
                if msg["role"] == "user":
                    col_spacer, col_user = st.columns([1.5, 8.5])
                    with col_user:
                        st.markdown('<div class="user-bubble-wrapper">', unsafe_allow_html=True)
                        with st.chat_message("user"):
                            st.markdown(msg["content"])
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    col_bot, col_spacer = st.columns([8.5, 1.5])
                    with col_bot:
                        st.markdown('<div class="bot-bubble-wrapper">', unsafe_allow_html=True)
                        with st.chat_message("assistant"):
                            st.markdown(msg["content"])
                        st.markdown('</div>', unsafe_allow_html=True)

            # If pending AI generation, render the newly sent user message immediately above the chat input
            # and display "Cooking..." until the response stream begins (all in 1 single pass, zero extra rerun)
            if pending_prompt:
                # 1. Render immediate user message bubble on right
                col_spacer, col_user = st.columns([1.5, 8.5])
                with col_user:
                    st.markdown('<div class="user-bubble-wrapper">', unsafe_allow_html=True)
                    with st.chat_message("user"):
                        st.markdown(pending_prompt)
                    st.markdown('</div>', unsafe_allow_html=True)

                # 2. Render assistant bubble on left with Cooking... indicator
                col_bot, col_spacer = st.columns([8.5, 1.5])
                with col_bot:
                    st.markdown('<div class="bot-bubble-wrapper">', unsafe_allow_html=True)
                    with st.chat_message("assistant"):
                        status_placeholder = st.empty()
                        status_placeholder.markdown(
                            '<div class="cooking-indicator"><span class="cooking-spinner">🍳</span><span class="cooking-text">Cooking...</span></div>',
                            unsafe_allow_html=True,
                        )

                        def stream_with_status_clear(gen, placeholder):
                            first = True
                            for chunk in gen:
                                if first:
                                    placeholder.empty()
                                    first = False
                                yield chunk

                        try:
                            history_for_gemini = current_messages[:-1]
                            response_generator = stream_chat_response(
                                client=client,
                                history_messages=history_for_gemini,
                                prompt=pending_prompt,
                                model=selected_model,
                            )
                            full_response = st.write_stream(stream_with_status_clear(response_generator, status_placeholder))
                        except Exception as e:
                            status_placeholder.empty()
                            full_response = f"⚠️ Error generating response: {e}"
                            st.error(full_response)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Store assistant response in memory and flush to disk asynchronously without forcing a second rerun
                current_messages.append({"role": "assistant", "content": full_response})
                current_session["updated_at"] = time.time()
                st.session_state.pop("pending_prompt", None)
                save_sessions_data(sessions_data)

    # Voice input microphone component (inserted directly before the Send button)
    # The script is loaded through a zero-height components iframe and injected into the
    # main page, because st.html does not execute <script> tags reliably.
    _sanvi_ui_js = """
        (function() {
            if (window.__sanviUiLoaded) return;
            window.__sanviUiLoaded = true;
            function getAppWin() {
                try {
                    if (document.querySelector('[data-testid="stApp"]') || document.querySelector('[data-testid="stChatInput"]')) {
                        return window;
                    }
                    if (window.parent && window.parent !== window && window.parent.document &&
                        (window.parent.document.querySelector('[data-testid="stApp"]') || window.parent.document.querySelector('[data-testid="stChatInput"]'))) {
                        return window.parent;
                    }
                } catch(e) {}
                return window;
            }
            function getAppDoc() {
                try { return getAppWin().document; } catch(e) { return document; }
            }
            function setupVoiceMic() {
                try {
                    const doc = getAppDoc();
                    if (!doc) return;
                    const chatInput = doc.querySelector('[data-testid="stChatInput"]');
                    if (!chatInput) return;
                    const sendBtn = chatInput.querySelector('button[data-testid="stChatInputSubmitButton"]');
                    if (!sendBtn) return;

                    if (chatInput.querySelector('#voice-mic-btn')) return;

                    const micBtn = doc.createElement('button');
                    micBtn.id = 'voice-mic-btn';
                    micBtn.type = 'button';
                    micBtn.title = 'Voice input (Click to speak)';
                    micBtn.setAttribute('aria-label', 'Voice input');
                    micBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>';

                    let recognition = null;
                    let isRecording = false;

                    micBtn.addEventListener('click', function(e) {
                        e.preventDefault();
                        e.stopPropagation();

                        const win = getAppWin();
                        const SpeechRec = win.SpeechRecognition || win.webkitSpeechRecognition || window.SpeechRecognition || window.webkitSpeechRecognition;
                        if (!SpeechRec) {
                            alert("Speech recognition is not supported in this browser. Please use Google Chrome, Microsoft Edge, or Safari.");
                            return;
                        }

                        if (isRecording) {
                            if (recognition) {
                                try { recognition.stop(); } catch(err) {}
                            }
                            isRecording = false;
                            micBtn.classList.remove('recording');
                            micBtn.title = 'Voice input (Click to speak)';
                            return;
                        }

                        try {
                            recognition = new SpeechRec();
                            recognition.continuous = true;
                            recognition.interimResults = true;
                            recognition.lang = navigator.language || 'en-US';

                            const textarea = chatInput.querySelector('textarea[data-testid="stChatInputTextArea"]');
                            let baseText = textarea ? textarea.value : '';
                            if (baseText && !baseText.endsWith(' ')) baseText += ' ';

                            recognition.onstart = function() {
                                isRecording = true;
                                micBtn.classList.add('recording');
                                micBtn.title = 'Listening... Click to stop';
                            };

                            recognition.onresult = function(event) {
                                let finalTranscript = '';
                                let interimTranscript = '';
                                for (let i = event.resultIndex; i < event.results.length; ++i) {
                                    if (event.results[i].isFinal) {
                                        finalTranscript += event.results[i][0].transcript;
                                    } else {
                                        interimTranscript += event.results[i][0].transcript;
                                    }
                                }
                                const currentSpoken = (finalTranscript || interimTranscript).trim();
                                if (textarea && currentSpoken) {
                                    const updated = baseText + currentSpoken;
                                    const nativeSetter = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value').set;
                                    nativeSetter.call(textarea, updated);
                                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                                    textarea.dispatchEvent(new Event('change', { bubbles: true }));
                                }
                            };

                            recognition.onerror = function(event) {
                                console.warn("Speech recognition error:", event.error);
                                isRecording = false;
                                micBtn.classList.remove('recording');
                                micBtn.title = 'Voice input (Click to speak)';
                            };

                            recognition.onend = function() {
                                isRecording = false;
                                micBtn.classList.remove('recording');
                                micBtn.title = 'Voice input (Click to speak)';
                            };

                            recognition.start();
                        } catch(err) {
                            console.error("Speech recognition error:", err);
                            isRecording = false;
                            micBtn.classList.remove('recording');
                            micBtn.title = 'Voice input (Click to speak)';
                        }
                    });

                    sendBtn.parentNode.insertBefore(micBtn, sendBtn);
                } catch(err) {
                    console.error("setupVoiceMic error:", err);
                }
            }
            function setupScrollFab() {
                try {
                    const doc = getAppDoc();
                    const win = getAppWin();
                    const fab = doc.querySelector('.scroll-down-fab');
                    if (!fab) return;

                    if (!fab.dataset.bound) {
                        fab.dataset.bound = '1';
                        fab.addEventListener('click', function() {
                            // Find the real scrolling container of the chat (Streamlit versions differ)
                            let scroller = null;
                            let node = doc.querySelector('[data-testid="stChatMessage"]') || doc.querySelector('[data-testid="stMain"]');
                            while (node && node !== doc.body) {
                                const oy = win.getComputedStyle(node).overflowY;
                                if ((oy === 'auto' || oy === 'scroll') && node.scrollHeight > node.clientHeight) {
                                    scroller = node;
                                    break;
                                }
                                node = node.parentElement;
                            }
                            if (!scroller) scroller = doc.scrollingElement || doc.documentElement;
                            scroller.scrollTo({ top: scroller.scrollHeight + 1000, behavior: 'smooth' });
                        });
                    }

                    // Align the arrow just above the right edge of the chat input (only mutate DOM when coords change)
                    const chatInput = doc.querySelector('[data-testid="stChatInput"]');
                    if (chatInput) {
                        const r = chatInput.getBoundingClientRect();
                        if (r.width > 0) {
                            const newRight = Math.round(win.innerWidth - r.right + 8) + 'px';
                            const newBottom = Math.round(win.innerHeight - r.top + 12) + 'px';
                            if (fab.dataset.lastRight !== newRight) {
                                fab.style.right = newRight;
                                fab.dataset.lastRight = newRight;
                            }
                            if (fab.dataset.lastBottom !== newBottom) {
                                fab.style.bottom = newBottom;
                                fab.dataset.lastBottom = newBottom;
                            }
                        }
                    }
                } catch(err) {
                    console.error("setupScrollFab error:", err);
                }
            }
            function setupInstantClickFeedback() {
                try {
                    const doc = getAppDoc();
                    if (!doc || doc.body.dataset.sanviInstantBound === '1') return;
                    doc.body.dataset.sanviInstantBound = '1';

                    if (!doc.getElementById('sanvi-optimistic-css')) {
                        const styleEl = doc.createElement('style');
                        styleEl.id = 'sanvi-optimistic-css';
                        styleEl.textContent =
                            '[data-testid="stSidebar"].sanvi-opt-switching div[class*="st-key-chat_item_"] > [data-testid="stHorizontalBlock"] { background: transparent !important; }' +
                            '[data-testid="stSidebar"].sanvi-opt-switching div[class*="st-key-chat_item_"].sanvi-opt-target > [data-testid="stHorizontalBlock"] { background: #38383a !important; border-radius: 8px !important; }';
                        doc.head.appendChild(styleEl);
                    }

                    let optTimer = null;

                    // 0ms optimistic visual response on pointerdown/click before Streamlit WebSocket roundtrip completes
                    doc.addEventListener('pointerdown', function(e) {
                        const target = e.target;
                        if (!target || !target.closest) return;

                        // 1. Sidebar chat row selection: highlight row & update topbar title immediately in 0ms
                        const chatRow = target.closest('[data-testid="stSidebar"] div[class*="st-key-chat_item_"]');
                        const chatBtn = target.closest('.stButton button');
                        const inPopoverTrigger = target.closest('[data-testid="stPopover"]');
                        if (chatRow && chatBtn && !inPopoverTrigger) {
                            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
                            if (sidebar) {
                                sidebar.querySelectorAll('.sanvi-opt-target').forEach(function(el) {
                                    el.classList.remove('sanvi-opt-target');
                                });
                                chatRow.classList.add('sanvi-opt-target');
                                sidebar.classList.add('sanvi-opt-switching');
                                if (optTimer) clearTimeout(optTimer);
                                optTimer = setTimeout(function() {
                                    sidebar.classList.remove('sanvi-opt-switching');
                                    chatRow.classList.remove('sanvi-opt-target');
                                }, 600);
                            }
                            const titleText = (chatBtn.textContent || '').trim();
                            const topTitle = doc.querySelector('.sanvibot-title');
                            if (topTitle && titleText) {
                                topTitle.textContent = titleText;
                            }
                            // On mobile viewports (<= 768px), auto-close the slide-over sidebar drawer after selecting a chat
                            const win = getAppWin();
                            if (win && win.innerWidth <= 768) {
                                setTimeout(function() {
                                    const closeBtn = doc.querySelector('[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarHeader"] button');
                                    if (closeBtn) {
                                        try { closeBtn.click(); } catch(err) {}
                                    }
                                }, 90);
                            }
                        }
                    }, { passive: true, capture: true });

                    doc.addEventListener('click', function(e) {
                        const target = e.target;
                        if (!target || !target.closest) return;

                        // 2. Popover menu action click: close popover menu visually in 0ms
                        const popBody = target.closest('[data-testid="stPopoverBody"]');
                        const actionBtn = target.closest('.stButton button');
                        if (popBody && actionBtn) {
                            const popWrapper = popBody.closest('[data-baseweb="popover"]') || popBody;
                            popWrapper.style.opacity = '0';
                            popWrapper.style.pointerEvents = 'none';
                            popWrapper.style.transition = 'opacity 0.05s ease';
                        }
                    }, { passive: true, capture: true });
                } catch(err) {
                    console.error("setupInstantClickFeedback error:", err);
                }
            }
            window.setInterval(function() { setupVoiceMic(); setupScrollFab(); setupInstantClickFeedback(); }, 250);
            setupVoiceMic();
            setupScrollFab();
            setupInstantClickFeedback();
        })();
    """
    _sanvi_ui_loader = (
        "<script>"
        "try {"
        "var p = window.parent; var d = p.document;"
        "if (!p.__sanviUiInjected) {"
        "p.__sanviUiInjected = true;"
        "var s = d.createElement('script');"
        "s.textContent = " + json.dumps(_sanvi_ui_js).replace("</", "<\\/") + ";"
        "d.head.appendChild(s);"
        "}"
        "} catch (e) { console.error('SanviBot UI loader error:', e); }"
        "</script>"
    )
    components.html(_sanvi_ui_loader, height=0)


def main():
    """Main entry point. Automatically runs SanviBot Streamlit UI or CLI mode."""
    if "--cli" in sys.argv:
        run_cli()
    else:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            in_streamlit = get_script_run_ctx() is not None
        except Exception:
            in_streamlit = False

        if in_streamlit:
            run_streamlit_app()
        else:
            print("Starting Streamlit Web UI for SanviBot...")
            import streamlit.web.cli as stcli
            sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
            sys.exit(stcli.main())


if __name__ == "__main__":
    main()
