"""
SanviBot Tools Module:
1. Browser & Live Online Information Tools (web_search, fetch_webpage, open_in_browser)
2. Laptop OS Folder & File Tools (get_system_paths, list_directory, search_files,
   read_file_content, write_file_content, create_directory, move_or_rename_path,
   copy_path, delete_path, open_in_explorer)
"""

import os
import re
import json
import time
import shutil
import fnmatch
import datetime
import webbrowser
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
from google import genai
from google.genai import types

WORKSPACE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
HOME_DIR = Path.home()
ONEDRIVE_DIR = HOME_DIR / "OneDrive"

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Persistent HTTP session for connection pooling (avoids repeated TCP/TLS handshakes)
_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update(_HTTP_HEADERS)

# In-memory TTL cache for web search queries: {normalized_query: (timestamp, result_str)}
_WEB_SEARCH_CACHE: Dict[str, Tuple[float, str]] = {}
_WEB_SEARCH_CACHE_TTL = 300.0  # 5 minutes


class _HTMLTextExtractor(HTMLParser):
    """Lightweight HTML-to-text parser that strips scripts/styles and extracts title + body text."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "meta", "link"}
    _BLOCK_TAGS = {"p", "div", "section", "article", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: List[str] = []
        self.text_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        t = tag.lower()
        if t == "title":
            self._in_title = True
        elif t in self._SKIP_TAGS:
            self._skip_depth += 1
        elif t in self._BLOCK_TAGS and self._skip_depth == 0:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if t == "title":
            self._in_title = False
        elif t in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif t in self._BLOCK_TAGS and self._skip_depth == 0:
            self.text_parts.append("\n")

    def handle_data(self, data: str):
        if self._in_title:
            self.title_parts.append(data.strip())
        elif self._skip_depth == 0:
            cleaned = re.sub(r"[ \t\r\f\v]+", " ", data)
            if cleaned.strip():
                self.text_parts.append(cleaned)

    def get_title(self) -> str:
        return " ".join(p for p in self.title_parts if p).strip()

    def get_text(self) -> str:
        raw = "".join(self.text_parts)
        lines = [line.strip() for line in raw.splitlines()]
        non_empty = [line for line in lines if line]
        return "\n".join(non_empty)


def _format_size(num_bytes: int) -> str:
    """Format byte size into human-readable string."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def _get_known_folders() -> Dict[str, Path]:
    """Return mapping of standard laptop folder aliases to actual paths (preferring OneDrive when active)."""
    folders: Dict[str, Path] = {
        "home": HOME_DIR,
        "~": HOME_DIR,
        "onedrive": ONEDRIVE_DIR if ONEDRIVE_DIR.exists() else HOME_DIR,
        "workspace": WORKSPACE_DIR,
        "project": WORKSPACE_DIR,
    }
    for name in ["Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos"]:
        od_candidate = ONEDRIVE_DIR / name
        home_candidate = HOME_DIR / name
        if od_candidate.exists():
            folders[name.lower()] = od_candidate
        else:
            folders[name.lower()] = home_candidate
    # Always map Downloads to HOME_DIR/Downloads if OneDrive/Downloads doesn't exist
    if (HOME_DIR / "Downloads").exists():
        folders["downloads"] = HOME_DIR / "Downloads"
    return folders


def resolve_user_path(path_str: str) -> Path:
    """Resolve user path strings including shortcuts like 'Desktop', 'Documents', 'Downloads', 'OneDrive', '~'."""
    if not path_str or not path_str.strip() or path_str.strip() == ".":
        return WORKSPACE_DIR

    cleaned = os.path.expandvars(path_str.strip().strip('"').strip("'"))
    known = _get_known_folders()

    low = cleaned.lower().replace("\\", "/").strip("/")
    if low in known:
        return known[low].resolve()

    # Handle relative paths starting with a known folder name, e.g. "Documents/notes.txt" or "Desktop/test"
    parts = cleaned.replace("\\", "/").split("/")
    first_key = parts[0].lower()
    if first_key in known and len(parts) > 1:
        sub_rel = Path(*parts[1:])
        candidate = known[first_key] / sub_rel
        # Also check non-OneDrive / OneDrive counterpart if it exists there
        if not candidate.exists() and first_key in ("desktop", "documents", "pictures"):
            alt_base = HOME_DIR / parts[0].capitalize()
            alt_candidate = alt_base / sub_rel
            if alt_candidate.exists():
                return alt_candidate.resolve()
        return candidate.resolve()

    expanded = Path(os.path.expanduser(cleaned))
    if expanded.is_absolute():
        # If user or LLM passes C:\Users\Admin\Documents\... but files are in C:\Users\Admin\OneDrive\Documents\...
        if not expanded.exists() and ONEDRIVE_DIR.exists():
            try:
                rel_to_home = expanded.relative_to(HOME_DIR)
                od_alt = ONEDRIVE_DIR / rel_to_home
                if od_alt.exists():
                    return od_alt.resolve()
            except ValueError:
                pass
        return expanded.resolve()

    return (WORKSPACE_DIR / expanded).resolve()


# ==============================================================================
# 1. BROWSER & LIVE ONLINE INFORMATION TOOLS
# ==============================================================================

def _search_google_news_rss(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Fetch live news and recent articles via Google News RSS using a warm HTTP session."""
    encoded = urllib.parse.quote_plus(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    resp = _HTTP_SESSION.get(rss_url, timeout=3.5)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    results: List[Dict[str, str]] = []
    for item in root.findall(".//item")[:max_results]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else ""
        if title:
            results.append({
                "title": title,
                "source": source,
                "published": pub_date,
                "url": link,
            })
    return results


def _search_duckduckgo_html(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Fetch general web search snippets via DuckDuckGo HTML endpoint using a warm HTTP session."""
    resp = _HTTP_SESSION.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        timeout=3.5,
    )
    resp.raise_for_status()
    html = resp.text

    # Extract result blocks
    results: List[Dict[str, str]] = []
    link_pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for idx, (href, raw_title) in enumerate(links[:max_results]):
        clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()
        raw_snip = snippets[idx] if idx < len(snippets) else ""
        clean_snip = re.sub(r"<[^>]+>", "", raw_snip).strip()
        # Unwrap DuckDuckGo redirect URL if present
        if "uddg=" in href:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            if "uddg" in parsed:
                href = parsed["uddg"][0]
        results.append({
            "title": clean_title,
            "snippet": clean_snip,
            "url": href,
        })
    return results


_SEARCH_CLIENT = None


def _get_search_client():
    """Return a cached genai.Client for grounded web search sub-calls."""
    global _SEARCH_CLIENT
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    if _SEARCH_CLIENT is None:
        _SEARCH_CLIENT = genai.Client(api_key=api_key)
    return _SEARCH_CLIENT


def _fetch_grounded_summary(query: str, now_str: str) -> str:
    """Fast grounded web search using active 2026 flash-lite/flash models (no retired 2.0/2.5 models)."""
    client = _get_search_client()
    if client is None:
        return ""
    for search_model in ("gemini-3.5-flash-lite", "gemini-flash-latest"):
        try:
            res = client.models.generate_content(
                model=search_model,
                contents=f"Current date: {now_str}. Provide a concise, factual answer with dates for: {query}",
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            if res and res.text:
                return f"--- Grounded Web Summary ({search_model}) ---\n{res.text.strip()}"
        except Exception:
            continue
    return ""


def web_search(query: str, max_results: int = 5) -> str:
    """Search the live internet for up-to-date information, current officeholders (CM, PM, President, CEO, etc.),
    today's news, elections, current events, mutable facts, weather, sports, or prices.

    ALWAYS call this tool BEFORE answering any question about who currently holds a political/corporate/sports position
    or any fact that can change over time, even if the user did not explicitly say 'current' or 'search online'.
    Do NOT call this tool for timeless coding, math, creative writing, or fixed historical facts.

    Args:
        query: The search query string to look up online.
        max_results: Maximum number of search results to return (default 5).
    """
    norm_q = " ".join(query.strip().lower().split())
    now_ts = time.time()
    cached = _WEB_SEARCH_CACHE.get(norm_q)
    if cached and (now_ts - cached[0]) < _WEB_SEARCH_CACHE_TTL:
        return cached[1]

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections: List[str] = [f"Search Query: {query}\nTimestamp: {now_str}"]

    # Run Grounded Search, DuckDuckGo HTML, and Google News RSS concurrently in parallel threads
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_grounded = pool.submit(_fetch_grounded_summary, query, now_str)
        fut_ddg = pool.submit(_search_duckduckgo_html, query, max_results)
        fut_news = pool.submit(_search_google_news_rss, query, max_results)

        try:
            grounded_text = fut_grounded.result(timeout=6.0)
            if grounded_text:
                sections.append(grounded_text)
        except Exception:
            pass

        try:
            ddg_results = fut_ddg.result(timeout=4.0)
            if ddg_results:
                ddg_lines = ["--- Live Web Search Results ---"]
                for i, item in enumerate(ddg_results, 1):
                    ddg_lines.append(f"{i}. {item['title']}\n   URL: {item['url']}\n   Snippet: {item['snippet']}")
                sections.append("\n".join(ddg_lines))
        except Exception:
            pass

        try:
            news_results = fut_news.result(timeout=4.0)
            if news_results:
                news_lines = ["--- Latest News Headlines (Google News RSS) ---"]
                for i, item in enumerate(news_results, 1):
                    src = f" [{item['source']}]" if item.get("source") else ""
                    pub = f" ({item['published']})" if item.get("published") else ""
                    news_lines.append(f"{i}. {item['title']}{src}{pub}\n   URL: {item['url']}")
                sections.append("\n".join(news_lines))
        except Exception:
            pass

    output = "\n\n".join(sections)
    _WEB_SEARCH_CACHE[norm_q] = (now_ts, output)
    return output


def fetch_webpage(url: str, max_chars: int = 10000) -> str:
    """Fetch and read the text content of a webpage URL.

    Args:
        url: Full URL (http:// or https://) of the webpage to read.
        max_chars: Maximum number of text characters to return (default 10000).
    """
    target_url = url.strip()
    if not target_url.startswith(("http://", "https://")):
        target_url = "https://" + target_url

    try:
        resp = _HTTP_SESSION.get(target_url, timeout=6)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type.lower() or "<html" in resp.text[:500].lower():
            extractor = _HTMLTextExtractor()
            extractor.feed(resp.text)
            title = extractor.get_title()
            body = extractor.get_text()
            if len(body) > max_chars:
                body = body[:max_chars] + "\n...[truncated]..."
            return f"URL: {resp.url}\nTitle: {title}\n\nContent:\n{body}"
        else:
            text = resp.text[:max_chars]
            return f"URL: {resp.url}\nContent-Type: {content_type}\n\nContent:\n{text}"
    except Exception as e:
        return f"Error fetching webpage '{target_url}': {e}"


def open_in_browser(url_or_query: str) -> str:
    """Open a website URL or search query in the user's laptop web browser.

    Args:
        url_or_query: Either a website URL (e.g. 'https://github.com' or 'youtube.com') or a search phrase.
    """
    raw = url_or_query.strip()
    if not raw:
        return "Error: Empty URL or query provided."

    if raw.startswith(("http://", "https://")):
        target = raw
    elif " " not in raw and "." in raw:
        target = "https://" + raw
    else:
        target = f"https://www.google.com/search?q={urllib.parse.quote_plus(raw)}"

    try:
        webbrowser.open(target)
        return f"Opened in browser: {target}"
    except Exception as e:
        return f"Error opening browser for '{target}': {e}"


# ==============================================================================
# 2. LAPTOP OS FOLDER & FILE TOOLS (FULL READ / WRITE / DELETE)
# ==============================================================================

def get_system_paths() -> str:
    """Return the user's key laptop directory paths (Home, Desktop, Documents, Downloads, OneDrive, Pictures, Workspace)."""
    known = _get_known_folders()
    result = {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in known.items()
        if name != "~"
    }
    return json.dumps(result, indent=2)


def _fast_dir_size(folder: Path, max_files: int = 8000) -> int:
    """Compute directory size rapidly via os.scandir with a file count cap to prevent hanging."""
    total = 0
    seen = 0
    stack = [str(folder)]
    while stack and seen < max_files:
        curr = stack.pop()
        try:
            with os.scandir(curr) as it:
                for entry in it:
                    seen += 1
                    if seen >= max_files:
                        break
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        else:
                            total += entry.stat(follow_symlinks=False).st_size
                    except Exception:
                        continue
        except Exception:
            continue
    return total


def list_directory(path: str = ".", include_folder_sizes: bool = False) -> str:
    """List all folders and files inside a directory on the user's laptop.
    Set include_folder_sizes=True when the user asks which folder occupies the most disk space or asks for folder sizes.

    Args:
        path: Folder path or shortcut ('Desktop', 'Documents', 'Downloads', 'OneDrive', '~', '.', or an absolute path).
        include_folder_sizes: If True, calculates and sorts folders by their total disk size.
    """
    try:
        target = resolve_user_path(path)
        if not target.exists():
            return f"Error: Directory does not exist: {target}"
        if not target.is_dir():
            return f"Error: Path is a file, not a directory: {target}"

        entries = list(target.iterdir())
        dirs = sorted([e for e in entries if e.is_dir()], key=lambda p: p.name.lower())
        files = sorted([e for e in entries if not e.is_dir()], key=lambda p: p.name.lower())

        lines = [
            f"Directory: {target}",
            f"Total: {len(dirs)} folder(s), {len(files)} file(s)",
            "-" * 60,
        ]
        if include_folder_sizes and dirs:
            dir_sizes = [(d, _fast_dir_size(d)) for d in dirs]
            dir_sizes.sort(key=lambda item: -item[1])
            for d, sz in dir_sizes:
                try:
                    mtime = datetime.datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    mtime = "unknown"
                lines.append(f"[DIR]  {d.name:<40} ({_format_size(sz)}, modified: {mtime})")
        else:
            for d in dirs:
                try:
                    mtime = datetime.datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    mtime = "unknown"
                lines.append(f"[DIR]  {d.name:<40} (modified: {mtime})")

        for f in files:
            try:
                st_info = f.stat()
                size_str = _format_size(st_info.st_size)
                mtime = datetime.datetime.fromtimestamp(st_info.st_mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                size_str = "?"
                mtime = "unknown"
            lines.append(f"[FILE] {f.name:<40} ({size_str}, modified: {mtime})")

        if not dirs and not files:
            lines.append("(Directory is empty)")

        return "\n".join(lines)
    except Exception as e:
        return f"Error listing directory '{path}': {e}"


def search_files(
    directory: str = "~",
    pattern: str = "*",
    recursive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for files or folders matching a name or wildcard pattern inside a laptop directory.

    Args:
        directory: Starting folder ('Desktop', 'Documents', 'Downloads', 'OneDrive', '~', '.', or full path).
        pattern: Filename substring or glob pattern (e.g. '*.pdf', '*report*', '*.py').
        recursive: Whether to search subfolders recursively (default True).
        max_results: Maximum number of matches to return (default 50).
    """
    try:
        root_dir = resolve_user_path(directory)
        if not root_dir.exists() or not root_dir.is_dir():
            return f"Error: Directory not found: {root_dir}"

        pat = pattern.strip() or "*"
        if "*" not in pat and "?" not in pat:
            pat = f"*{pat}*"
        pat_lower = pat.lower()

        skip_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox", ".mypy_cache"}
        matches: List[str] = []

        if not recursive:
            for entry in root_dir.iterdir():
                if fnmatch.fnmatch(entry.name.lower(), pat_lower):
                    kind = "DIR" if entry.is_dir() else "FILE"
                    matches.append(f"[{kind}] {entry}")
                    if len(matches) >= max_results:
                        break
        else:
            for curr_root, dirs, files in os.walk(root_dir):
                dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith("$")]
                for d in dirs:
                    if fnmatch.fnmatch(d.lower(), pat_lower):
                        matches.append(f"[DIR]  {os.path.join(curr_root, d)}")
                        if len(matches) >= max_results:
                            break
                if len(matches) >= max_results:
                    break
                for f in files:
                    if fnmatch.fnmatch(f.lower(), pat_lower):
                        full_p = os.path.join(curr_root, f)
                        try:
                            sz = _format_size(os.path.getsize(full_p))
                        except Exception:
                            sz = "?"
                        matches.append(f"[FILE] {full_p} ({sz})")
                        if len(matches) >= max_results:
                            break
                if len(matches) >= max_results:
                    break

        if not matches:
            return f"No files or folders matching '{pattern}' found in {root_dir}."

        header = f"Search in {root_dir} for '{pattern}' ({len(matches)} match(es)):"
        return header + "\n" + "\n".join(matches)
    except Exception as e:
        return f"Error searching in '{directory}': {e}"


def read_file_content(path: str, max_chars: int = 15000) -> str:
    """Read the text content of a file on the user's laptop.

    Args:
        path: Path to the file (supports shortcuts like 'Documents/notes.txt' or full path).
        max_chars: Maximum number of characters to return (default 15000).
    """
    try:
        target = resolve_user_path(path)
        if not target.exists():
            return f"Error: File does not exist: {target}"
        if target.is_dir():
            return f"Error: '{target}' is a directory. Use list_directory instead."

        size = target.stat().st_size
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                with open(target, "r", encoding=enc) as f:
                    content = f.read(max_chars + 1)
                truncated = len(content) > max_chars
                if truncated:
                    content = content[:max_chars] + "\n...[truncated]..."
                return f"File: {target} ({_format_size(size)})\n\n{content}"
            except UnicodeDecodeError:
                continue

        return f"Error: '{target}' appears to be a binary file ({_format_size(size)}) and cannot be read as plain text."
    except Exception as e:
        return f"Error reading file '{path}': {e}"


def write_file_content(path: str, content: str, append: bool = False) -> str:
    """Create a new file, overwrite an existing file, or append text to a file on the user's laptop.

    Args:
        path: Destination file path (supports shortcuts like 'Desktop/todo.txt', 'Documents/...', or full path).
        content: Text content to write.
        append: If True, appends to the end of the file; if False (default), overwrites/creates the file.
    """
    try:
        target = resolve_user_path(path)
        if target.exists() and target.is_dir():
            return f"Error: Target path is an existing directory: {target}"

        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(target, mode, encoding="utf-8") as f:
            f.write(content)

        action = "Appended to" if append else "Wrote"
        return f"Success: {action} file '{target}' ({_format_size(target.stat().st_size)})."
    except Exception as e:
        return f"Error writing file '{path}': {e}"


def create_directory(path: str) -> str:
    """Create a new folder (and any parent folders) on the user's laptop.

    Args:
        path: Folder path to create (e.g. 'Desktop/NewProject' or full path).
    """
    try:
        target = resolve_user_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return f"Success: Directory ready at '{target}'."
    except Exception as e:
        return f"Error creating directory '{path}': {e}"


def move_or_rename_path(src_path: str, dest_path: str) -> str:
    """Move or rename a file or folder on the user's laptop.

    Args:
        src_path: Existing file or folder path.
        dest_path: New destination path or new name.
    """
    try:
        src = resolve_user_path(src_path)
        dest = resolve_user_path(dest_path)
        if not src.exists():
            return f"Error: Source path does not exist: {src}"

        dest.parent.mkdir(parents=True, exist_ok=True)
        final_dest = shutil.move(str(src), str(dest))
        return f"Success: Moved/renamed '{src}' -> '{final_dest}'."
    except Exception as e:
        return f"Error moving '{src_path}' to '{dest_path}': {e}"


def copy_path(src_path: str, dest_path: str) -> str:
    """Copy a file or folder to another location on the user's laptop.

    Args:
        src_path: Source file or folder path.
        dest_path: Destination file or folder path.
    """
    try:
        src = resolve_user_path(src_path)
        dest = resolve_user_path(dest_path)
        if not src.exists():
            return f"Error: Source path does not exist: {src}"

        if src.is_dir():
            if dest.exists() and dest.is_dir():
                dest = dest / src.name
            shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
            return f"Success: Copied directory '{src}' -> '{dest}'."
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            final_dest = shutil.copy2(str(src), str(dest))
            return f"Success: Copied file '{src}' -> '{final_dest}'."
    except Exception as e:
        return f"Error copying '{src_path}' to '{dest_path}': {e}"


def delete_path(path: str) -> str:
    """Delete a file or folder on the user's laptop.

    Args:
        path: Path to the file or folder to delete.
    """
    try:
        target = resolve_user_path(path)
        if not target.exists():
            return f"Error: Path does not exist: {target}"

        # Guard against deleting top-level OS/user root directories
        protected_roots = {
            Path(target.anchor).resolve(),
            Path(r"C:\Windows").resolve(),
            Path(r"C:\Program Files").resolve(),
            Path(r"C:\Program Files (x86)").resolve(),
            Path(r"C:\Users").resolve(),
            HOME_DIR.resolve(),
            ONEDRIVE_DIR.resolve(),
        }
        for k_path in _get_known_folders().values():
            protected_roots.add(k_path.resolve())

        if target.resolve() in protected_roots:
            return f"Error: Refusing to delete protected root directory '{target}'."

        if target.is_dir():
            shutil.rmtree(str(target))
            return f"Success: Deleted directory '{target}'."
        else:
            target.unlink()
            return f"Success: Deleted file '{target}'."
    except Exception as e:
        return f"Error deleting '{path}': {e}"


def open_in_explorer(path: str = ".") -> str:
    """Open a folder or file on the user's laptop using Windows File Explorer / default app.

    Args:
        path: Folder or file path to open (e.g. 'Downloads', 'Documents', '.', or full path).
    """
    try:
        target = resolve_user_path(path)
        if not target.exists():
            return f"Error: Path does not exist: {target}"
        os.startfile(str(target))
        return f"Opened in Windows Explorer: {target}"
    except Exception as e:
        return f"Error opening '{path}' in Explorer: {e}"


ALL_TOOLS = [
    web_search,
    fetch_webpage,
    open_in_browser,
    get_system_paths,
    list_directory,
    search_files,
    read_file_content,
    write_file_content,
    create_directory,
    move_or_rename_path,
    copy_path,
    delete_path,
    open_in_explorer,
]

# Fast <1ms regex patterns to detect queries asking about mutable, current, or time-sensitive facts
_HISTORICAL_PAST_RE = re.compile(
    r"\b(who\s+was|what\s+was|first\s+cm|first\s+pm|first\s+president|in\s+1[0-9]{3}|in\s+20[01][0-9]|history\s+of)\b",
    re.IGNORECASE,
)

_LOCAL_OS_OR_CODE_RE = re.compile(
    r"\b(write\s+a\s+python|write\s+code|debug\s+code|draft\s+a\s+message|draft\s+an?\s+email|"
    r"my\s+folder|my\s+files|downloads\s+folder|documents\s+folder|desktop\s+folder|onedrive|"
    r"c:\\|create\s+a\s+folder|delete\s+the\s+file|rename\s+the\s+file)\b",
    re.IGNORECASE,
)

_EXPLICIT_WEB_RE = re.compile(
    r"\b(search\s+online|check\s+online|look\s+up\s+online|search\s+the\s+web|google\s+it|browse\s+the\s+web)\b",
    re.IGNORECASE,
)

_OFFICEHOLDER_OR_ROLE_RE = re.compile(
    r"(\b(who\s+is|who's|whats\s+the\s+name\s+of|name\s+the|current|present|incumbent|new)\b.{0,40}"
    r"\b(cm|chief\s+minister|deputy\s+cm|pm|prime\s+minister|president|vice\s+president|governor|"
    r"minister|mayor|ceo|cfo|cto|coo|chairman|chairperson|md|managing\s+director|founder|owner|"
    r"chief\s+justice|speaker|commissioner|secretary|ambassador|captain|coach|champion|richest|"
    r"leader|head|mp|mla)\b)"
    r"|(\b(cm|chief\s+minister|pm|prime\s+minister|president|governor|ceo|mayor|captain|champion)\s+of\b)",
    re.IGNORECASE,
)

_TIME_SENSITIVE_FACT_RE = re.compile(
    r"\b(today|tonight|yesterday|this\s+week|this\s+month|this\s+year|right\s+now|currently|"
    r"latest|current|recent|recently|breaking|live\s+score|live\s+update|upcoming|"
    r"2024|2025|2026|2027|"
    r"weather|temperature|forecast|rain\s+today|aqi|"
    r"stock\s+price|share\s+price|price\s+of|gold\s+rate|silver\s+rate|exchange\s+rate|usd\s+to|inr|"
    r"bitcoin|btc|eth|crypto|market\s+cap|net\s+worth|"
    r"election|polls|ruling\s+party|cabinet|"
    r"winner\s+of|who\s+won|score|standings|points\s+table|ipl|world\s+cup|olympics|tournament|"
    r"box\s+office|release\s+date|new\s+movie|"
    r"news\s+about|news\s+on|headlines|what\s+happened\s+in|update\s+on)\b",
    re.IGNORECASE,
)


def should_prefer_web_search(prompt: str) -> bool:
    """Fast (<1ms) classifier deciding whether a prompt asks about a mutable/time-sensitive fact
    that requires live online verification via `web_search` rather than static pre-trained memory."""
    if not prompt or not prompt.strip():
        return False
    text = prompt.strip()

    # Always honor explicit requests to check/search online
    if _EXPLICIT_WEB_RE.search(text):
        return True

    # Do not force web search on local OS file commands or pure coding/drafting prompts
    if _LOCAL_OS_OR_CODE_RE.search(text):
        return False

    # Present-tense officeholders/roles (unless explicitly asking about past history like 'who was')
    if _OFFICEHOLDER_OR_ROLE_RE.search(text) and not _HISTORICAL_PAST_RE.search(text):
        return True

    # Time-sensitive keywords, years (2024+), prices, weather, elections, sports, news
    if _TIME_SENSITIVE_FACT_RE.search(text):
        return True

    return False


def get_system_instruction(force_web_search: bool = False) -> str:
    """Build the system instruction with today's live date and the autonomous Local-vs-Online decision policy."""
    today_str = datetime.date.today().strftime("%B %d, %Y")
    base = (
        f"You are SanviBot, a helpful AI assistant running on the user's Windows laptop.\n"
        f"CURRENT DATE: {today_str}.\n"
        f"CRITICAL KNOWLEDGE CUTOFF WARNING: Your internal pre-trained weights have a past knowledge cutoff and are "
        f"OUTDATED for any real-world role, officeholder, political leader, price, event, or fact that can change over time.\n\n"
        f"=== AUTONOMOUS DECISION POLICY: LOCAL KNOWLEDGE vs. ONLINE WEB SEARCH ===\n"
        f"Before answering ANY user query, you must autonomously decide whether to answer from your local knowledge "
        f"or call `web_search` FIRST (without waiting for the user to ask you to search online):\n\n"
        f"1. WHEN TO AUTOMATICALLY CALL `web_search` FIRST (MANDATORY):\n"
        f"   - Current Officeholders & Positions: Any question asking who is the CM (Chief Minister), PM (Prime Minister), "
        f"President, Vice President, Governor, Minister, CEO, Chairman, Mayor, Chief Justice, Captain, Coach, or Champion "
        f"(e.g., 'Who is the CM of Tamil Nadu?', 'Who is the CEO of Intel?'), EVEN IF the user does not write the word 'current' or 'online'.\n"
        f"   - Mutable & Time-Sensitive Facts: Elections, political changes, government policies, sports scores/winners/rosters, "
        f"stock/crypto/gold prices, exchange rates, weather, company news, product/software latest versions, box office, "
        f"schedules, or anything from 2024–{datetime.date.today().year}.\n"
        f"   - Uncertain or Recent Entities: Whenever there is ANY possibility that the fact changed after your training cutoff.\n\n"
        f"2. WHEN TO USE LOCAL KNOWLEDGE DIRECTLY (DO NOT CALL `web_search`):\n"
        f"   - Coding, debugging, software engineering, math, logic, algorithms, and technical explanations (e.g., 'Explain sync vs async').\n"
        f"   - Creative writing, drafting emails/messages, grammar, translation, brainstorming, or summarizing text provided in the chat.\n"
        f"   - Timeless scientific principles or fixed historical facts in the past (e.g., 'Who was the first CM of Tamil Nadu?').\n\n"
        f"3. LAPTOP OS FOLDER & FILE TOOLS (`get_system_paths`, `list_directory`, `search_files`, `read_file_content`, "
        f"`write_file_content`, `create_directory`, `move_or_rename_path`, `copy_path`, `delete_path`, `open_in_explorer`):\n"
        f"   - Use these tools whenever the user asks to inspect, browse, search, read, create, edit, organize, move, copy, delete, "
        f"or open folders and files on their laptop (Desktop, Documents, Downloads, OneDrive, or any path)."
    )
    if force_web_search:
        base += (
            f"\n\n[MANDATORY ROUTING FOR THIS TURN: The user's question asks about a current officeholder, mutable status, "
            f"or time-sensitive fact as of {today_str}. Do NOT rely on pre-trained memory. You MUST call `web_search` "
            f"before answering, and use the live search results as the authoritative truth.]"
        )
    return base


SYSTEM_INSTRUCTION = get_system_instruction(force_web_search=False)
