"""
run.py - Startup script for ARIA AI Assistant Web Server
Run: python run.py
Then open: http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.align import Align
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

import threading
import webbrowser
import uvicorn

console = Console() if HAS_RICH else None

def open_browser_delayed(url: str):
    time.sleep(1.8)
    try:
        webbrowser.open(url)
    except Exception:
        pass

BANNER = r"""
     _    ____  ___    _    
    / \  |  _ \|_ _|  / \   
   / _ \ | |_) || |  / _ \  
  / ___ \|  _ < | | / ___ \ 
 /_/   \_\_| \_\___/_/   \_\
"""

def print_startup(host: str, port: int):
    if not HAS_RICH:
        print("=" * 60)
        print("ARIA - Offline AI Assistant")
        print(f"Server: http://{host}:{port}")
        print("=" * 60)
        return

    console.clear()
    console.print(Panel(
        Align.center(
            Text(BANNER, style="bold cyan") +
            Text("\nArtificially Responsive Intelligent Assistant\n", style="bold white") +
            Text("AI Project - Offline AI using Ollama\n", style="dim") +
            Text("v2.0.0", style="bold green")
        ),
        border_style="bright_cyan",
        padding=(1, 4),
    ))

    table = Table(show_header=False, box=box.SIMPLE_HEAVY, padding=(0, 2))
    table.add_column("", style="dim cyan", width=18)
    table.add_column("", style="bold white")
    table.add_row("Web UI",    f"[link]http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}[/link]")
    table.add_row("Terminal",  "python chat.py")
    table.add_row("API Docs",  f"http://127.0.0.1:{port}/docs")
    table.add_row("Health",    f"http://127.0.0.1:{port}/health")
    table.add_row("Model",     "llama3.2:latest")
    table.add_row("Backend",   "FastAPI + Ollama + LangChain")

    console.print(Panel(table, title="[bold cyan]Server Starting[/]", border_style="cyan"))
    console.print()
    console.print("[bold green]Press Ctrl+C to stop the server[/]")
    console.print()


def parse_args():
    parser = argparse.ArgumentParser(description="ARIA AI Assistant Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true", default=False)
    return parser.parse_args()


def free_port_if_busy(port: int):
    """Ensure port is not blocked before binding."""
    try:
        import psutil
        cur_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.pid == cur_pid:
                continue
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr and conn.laddr.port == port:
                        proc.kill()
                        time.sleep(0.5)
            except Exception:
                pass
    except Exception:
        pass


def main():
    args = parse_args()
    free_port_if_busy(args.port)
    print_startup(args.host, args.port)
    url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}"
    threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
