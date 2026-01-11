#!/usr/bin/env python3
"""
v1_basic_agent.py - Mini Claude Code: Model as Agent (~200 lines)

Core Philosophy: "The Model IS the Agent"
=========================================
The secret of Claude Code, Cursor Agent, Codex CLI? There is no secret.

Strip away the CLI polish, progress bars, permission systems. What remains
is surprisingly simple: a LOOP that lets the model call tools until done.

Traditional Assistant:
    User -> Model -> Text Response

Agent System:
    User -> Model -> [Tool -> Result]* -> Response
                          ^________|

The asterisk (*) matters! The model calls tools REPEATEDLY until it decides
the task is complete. This transforms a chatbot into an autonomous agent.

KEY INSIGHT: The model is the decision-maker. Code just provides tools and
runs the loop. The model decides:
  - Which tools to call
  - In what order
  - When to stop

The Four Essential Tools:
------------------------
Claude Code has ~20 tools. But these 4 cover 90% of use cases:

    | Tool       | Purpose              | Example                    |
    |------------|----------------------|----------------------------|
    | bash       | Run any command      | npm install, git status    |
    | read_file  | Read file contents   | View src/index.ts          |
    | write_file | Create/overwrite     | Create README.md           |
    | edit_file  | Surgical changes     | Replace a function         |

With just these 4 tools, the model can:
  - Explore codebases (bash: find, grep, ls)
  - Understand code (read_file)
  - Make changes (write_file, edit_file)
  - Run anything (bash: python, npm, make)

Usage:
    python v1_basic_agent.py
"""

import asyncio
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai import Agent
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
WORKDIR = Path.cwd()

model = OpenRouterModel(
    'mistralai/devstral-2512:free',
    provider=OpenRouterProvider(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        app_url='https://github.com/likui628/learn-agent',
        app_title='Learn Agent',
    )
)


# =============================================================================
# System Prompt - The only "configuration" the model needs
# =============================================================================

SYSTEM_PROMPT = f"""You are a coding agent at {WORKDIR}.

Loop: think briefly -> use tools -> report results.

Rules:
- Prefer tools over prose. Act, don't just explain.
- Never invent file paths. Use bash ls/find first if unsure.
- Make minimal changes. Don't over-engineer.
- After finishing, summarize what changed."""


agent = Agent(
    model=model,
    output_type=str,
    system_prompt=SYSTEM_PROMPT,
)


def safe_path(p: str) -> Path:
    """
    Ensure path stays within workspace (security measure).

    Prevents the model from accessing files outside the project directory.
    Resolves relative paths and checks they don't escape via '../'.
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


# =============================================================================
# Tool Definitions & Implementations - 4 tools cover 90% of coding tasks
# =============================================================================

@agent.tool_plain(name="bash")
def run_bash(command: str) -> str:
    """
    Run a shell command. Use for: ls, find, grep, git, npm, python, etc.

    Args:
        command: The shell command to execute

    Notes:
        - Execute shell command with safety checks.
        - Security: Blocks obviously dangerous commands.
        - Timeout: 60 seconds to prevent hanging.
        - Output: Truncated to 50KB to prevent context overflow.
    """
    # Basic safety - block dangerous patterns
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s)"
    except Exception as e:
        return f"Error: {e}"


@agent.tool_plain(name="read_file")
def run_read(path: str, limit: int = None) -> str:
    """
    Read file contents. Returns UTF-8 text.
    Args:
        path: Relative path to the file
        limit: Optional maximum number of lines to read

    Notes:
        Read file contents with optional line limit.

        For large files, use limit to read just the first N lines.
        Output truncated to 50KB to prevent context overflow.
    """
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()

        if limit and limit < len(lines):
            lines = lines[:limit]
            lines.append(f"... ({len(text.splitlines()) - limit} more lines)")

        return "\n".join(lines)[:50000]

    except Exception as e:
        return f"Error: {e}"


@agent.tool_plain(name="write_file")
def run_write(path: str, content: str) -> str:
    """
    Write content to a file. Creates parent directories if needed.
    Args:
        path: Relative path to the file
        content: Text content to write

    Notes:
        Write content to file, creating parent directories if needed.

        This is for complete file creation/overwrite.
        For partial edits, use edit_file instead.
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    except Exception as e:
        return f"Error: {e}"


@agent.tool_plain(name="edit_file")
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    Replace exact text in a file. Use for surgical edits.
    Args:
        path: Relative path to the file
        old_text: Exact text to find (must match precisely)
        new_text: Replacement text

    Notes:
        Replace exact text in a file (surgical edit).

        Uses exact string matching - the old_text must appear verbatim.
        Only replaces the first occurrence to prevent accidental mass changes.
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()

        if old_text not in content:
            return f"Error: Text not found in {path}"

        # Replace only first occurrence for safety
        new_content = content.replace(old_text, new_text, 1)
        fp.write_text(new_content)
        return f"Edited {path}"

    except Exception as e:
        return f"Error: {e}"

# =============================================================================
# Main REPL
# =============================================================================


async def chat(prompt: str) -> str:
    """
    Execute a single agent task.

    Args:
        prompt: User's request

    Returns:
        Final text response from the model
    """
    result = await agent.run(prompt)
    return result.output


async def interactive_mode():
    """
    Simple Read-Eval-Print Loop for interactive use.

    The history list maintains conversation context across turns,
    allowing multi-turn conversations with memory.
    """
    print(f"Mini Claude Code v1 - {WORKDIR}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        try:
            response = await chat(user_input)
            print(response)
            print()  # Blank line between turns
        except Exception as e:
            print(f"\033[31mError: {e}\033[0m")

if __name__ == "__main__":
    asyncio.run(interactive_mode())
