#!/usr/bin/env python
"""
pydantic_bash_agent.py - Mini Claude Code with Pydantic AI

Core Philosophy: "Bash is All You Need" + Pydantic AI's type safety
"""

import os
import sys
import subprocess
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

# Initialize OpenRouter model
model = OpenRouterModel(
    'mistralai/devstral-2512:free',
    provider=OpenRouterProvider(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        app_url='https://github.com/likui628/learn-agent',
        app_title='Learn Agent',
    )
)

# System prompt teaches the model HOW to use bash effectively
SYSTEM_PROMPT = f"""You are a CLI agent at {os.getcwd()}. Solve problems using bash commands.

Rules:
- Prefer tools over prose. Act first, explain briefly after.
- Read files: cat, grep, find, rg, ls, head, tail
- Write files: echo '...' > file, sed -i, or cat << 'EOF' > file
- Subagent: For complex subtasks, spawn a subagent to keep context clean:
  python {__file__} "explore src/ and summarize the architecture"

When to use subagent:
- Task requires reading many files (isolate the exploration)
- Task is independent and self-contained
- You want to avoid polluting current conversation with intermediate details

The subagent runs in isolation and returns only its final summary."""

# Create agent with bash tool
agent = Agent[None, str](
    model=model,
    output_type=str,
    system_prompt=SYSTEM_PROMPT,
)


@agent.tool
def bash(ctx: RunContext[None], command: str) -> str:
    """Execute shell command. Common patterns:
    - Read: cat/head/tail, grep/find/rg/ls, wc -l
    - Write: echo 'content' > file, sed -i 's/old/new/g' file
    - Subagent: python {__file__} 'task description' (spawns isolated agent, returns summary)

    Args:
        command: The shell command to execute

    Returns:
        Combined stdout and stderr output
    """
    print(f"\033[33m$ {command}\033[0m")  # Yellow color for commands

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.getcwd()
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = "(timeout after 300s)"
    except Exception as e:
        output = f"Error: {str(e)}"

    print(output or "(empty)")
    return output[:50000]  # Truncate very long outputs


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
    """Interactive REPL mode with conversation history."""
    print("Pydantic AI Bash Agent - Interactive Mode")
    print("Type 'q', 'exit', or Ctrl+C to quit\n")

    while True:
        try:
            query = input("\033[36m>> \033[0m")  # Cyan prompt
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if query.strip() in ("q", "exit", ""):
            break

        try:
            response = await chat(query)
            print(response)
        except Exception as e:
            print(f"\033[31mError: {e}\033[0m")


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) > 1:
        # Subagent mode: execute task and print result
        task = " ".join(sys.argv[1:])
        result = asyncio.run(chat(task))
        print(result)
    else:
        # Interactive REPL mode
        asyncio.run(interactive_mode())
