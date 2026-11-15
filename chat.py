"""
chat.py - Terminal mode for ARIA AI Assistant
Run: python chat.py
"""

import asyncio
import os
import sys
import time
from datetime import datetime

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
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich import box
    from rich.align import Align
    from rich.padding import Padding
except ImportError:
    print("ERROR: 'rich' library not found. Run: pip install rich")
    sys.exit(1)

import httpx

console = Console(highlight=False)
BASE_URL  = "http://127.0.0.1:8080"
SESSION   = "term_" + datetime.now().strftime("%H%M%S")

BANNER = r"""
     _    ____  ___    _
    / \  |  _ \|_ _|  / \
   / _ \ | |_) || |  / _ \
  / ___ \|  _ < | | / ___ \
 /_/   \_\_| \_\___/_/   \_\
"""

HELP_TEXT = """
[bold cyan]Commands[/]

  [yellow]/help[/]      Show this menu
  [yellow]/clear[/]     Clear the screen
  [yellow]/model[/]     Switch AI model
  [yellow]/info[/]      Show system status
  [yellow]/bench[/]     Open benchmark in browser
  [yellow]/exit[/]      Quit ARIA

[dim]Press Enter to send  •  Shift+Enter for newline[/]
"""

MODELS = [
    "llama3.2:latest",
    "llama3.1:8b",
    "phi3:mini",
    "gemma2:2b",
]

current_model = "llama3.2:latest"
msg_count = 0


# ── Splash ────────────────────────────────────────────────────────────────────
def splash():
    console.clear()
    banner_text = Text(BANNER, style="bold purple")
    subtitle = Text("\n  Artificially Responsive Intelligent Assistant", style="bold white")
    footer   = Text("  Offline AI  •  Powered by Ollama\n", style="dim")
    console.print(Panel(
        Align.center(banner_text + subtitle + footer),
        border_style="bright_magenta",
        padding=(0, 2),
    ))


def show_connecting():
    with console.status("[bold purple]Connecting to ARIA server...[/]", spinner="aesthetic"):
        time.sleep(1.2)


# ── Health check ──────────────────────────────────────────────────────────────
async def check_server():
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{BASE_URL}/health")
            d = r.json()
            return d.get("status") == "healthy", d
    except Exception:
        return False, {}


def build_status_table(health: dict) -> Table:
    hw = health.get("hardware", {})
    t  = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    t.add_column("k", style="dim", width=16)
    t.add_column("v", style="bold white")
    t.add_row("Status",  "[bold green]ONLINE[/]")
    t.add_row("Model",   f"[purple]{current_model}[/]")
    t.add_row("Ollama",  "[green]Connected[/]" if health.get("ollama_online") else "[red]Offline[/]")
    t.add_row("RAM",     f"{hw.get('ram_available_mb', 0):.0f} MB free")
    t.add_row("CPU",     hw.get("cpu", "—")[:40])
    t.add_row("Session", SESSION)
    return t


# ── Streaming chat ─────────────────────────────────────────────────────────────
async def stream_chat(message: str):
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream(
            "POST", f"{BASE_URL}/chat",
            json={"message": message, "session_id": SESSION, "model_name": current_model},
        ) as resp:
            async for chunk in resp.aiter_text():
                yield chunk


# ── Commands ──────────────────────────────────────────────────────────────────
def cmd_help():
    console.print(Panel(HELP_TEXT, title="[bold purple]ARIA Help[/]", border_style="purple"))


def cmd_clear():
    splash()


async def cmd_info():
    ok, health = await check_server()
    if not ok:
        console.print("[red]Cannot reach server.[/]")
        return
    console.print(Panel(
        build_status_table(health),
        title="[bold purple]System Status[/]",
        border_style="purple",
    ))


def cmd_model():
    global current_model
    console.print("[bold]Available models:[/]")
    for i, m in enumerate(MODELS, 1):
        mark = "[purple]●[/]" if m == current_model else " "
        console.print(f"  {mark} [cyan]{i}.[/] {m}")
    choice = Prompt.ask("[dim]Enter number to switch (or Enter to keep current)[/]", default="")
    if choice.isdigit() and 1 <= int(choice) <= len(MODELS):
        current_model = MODELS[int(choice) - 1]
        console.print(f"[green]Switched to[/] [purple]{current_model}[/]")


def cmd_bench():
    import webbrowser
    webbrowser.open(f"{BASE_URL}")
    console.print("[dim]Opened browser — go to the Benchmark tab.[/]")


# ── Main loop ─────────────────────────────────────────────────────────────────
async def main():
    global msg_count

    splash()
    show_connecting()

    ok, health = await check_server()
    if not ok:
        console.print(Panel(
            "[bold red]Cannot connect to ARIA server.[/]\n\n"
            "Start it first:\n"
            "  [yellow]python run.py[/]\n\n"
            "Then run chat.py again.",
            title="[red]Connection Error[/]",
            border_style="red",
        ))
        input("\nPress Enter to exit...")
        return

    console.print(Panel(
        build_status_table(health),
        title="[bold purple]ARIA is Online[/]",
        border_style="bright_magenta",
    ))
    console.print()
    console.print("[dim]Type a message and press Enter. Use /help for commands.[/]\n")

    while True:
        try:
            user_in = Prompt.ask("[bold green]You[/]", console=console).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_in:
            continue

        # ── Commands
        cmd = user_in.lower()
        if cmd == "/exit":
            console.print(Panel("[bold purple]Goodbye! ARIA signing off.[/]", border_style="purple"))
            break
        elif cmd == "/help":
            cmd_help(); continue
        elif cmd == "/clear":
            cmd_clear(); continue
        elif cmd == "/info":
            await cmd_info(); continue
        elif cmd == "/model":
            cmd_model(); continue
        elif cmd == "/bench":
            cmd_bench(); continue

        # ── Chat
        msg_count += 1
        console.print()
        console.rule(f"[dim]Message {msg_count}[/]", style="dim purple")

        full    = ""
        t_start = time.perf_counter()

        with Live(console=console, refresh_per_second=20) as live:
            live.update(Panel(
                Spinner("aesthetic", text=" Thinking…"),
                title="[bold purple]ARIA[/]",
                border_style="purple",
            ))
            first = True
            try:
                async for chunk in stream_chat(user_in):
                    if first:
                        first = False
                    full += chunk
                    display = Markdown(full) if len(full) > 30 else Text(full, style="white")
                    live.update(Panel(
                        display,
                        title=f"[bold purple]ARIA[/] [dim]({current_model})[/]",
                        border_style="bright_magenta",
                    ))
            except Exception as e:
                live.update(Panel(f"[red]Error: {e}[/]", border_style="red"))

        elapsed = time.perf_counter() - t_start
        console.print(f"  [dim]⏱  {elapsed:.2f}s  •  {current_model}[/]\n")


if __name__ == "__main__":
    asyncio.run(main())
