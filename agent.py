"""
TODO: build agent loop

1. take starting url and goal/instruction
2. call get_observation() to get initial observation
3. build system prompt/user message with observation, goal, and history
4. call llm to get action (need to impl openai calls in llm.py)
5. parse and validate action
6. execute action
7. log everything and repeat until stop or max_steps
"""

from __future__ import annotations

from actions import ACTION_VOCABULARY


_SYSTEM_PROMPT = f"""You are an autonomous browser agent. You control a real web browser.

Each turn you will receive:
  - GOAL: the task you must complete
  - URL: the current page URL
  - PAGE: a numbered list of interactive elements on the current page
  - HISTORY: the actions you have already taken (empty on the first turn)

Your job is to decide the single next action that best makes progress toward the GOAL.

{ACTION_VOCABULARY}

Rules:
1. Output EXACTLY ONE action per reply, on a single line. Nothing else — no explanation, no preamble.
2. Use only the actions listed above. Any other output will be treated as a parse error.
3. When the GOAL is achieved, output: stop
4. If the page does not help and you cannot make progress, output: stop
""".strip()


def build_system_prompt() -> str:
    """Returns the fixed system message for every episode."""
    return _SYSTEM_PROMPT


# per-turn user message
def build_user_message(
    goal: str,
    obs_text: str,
    action_history: list[str],
) -> str:
    """
    Builds the user message for one step.

    Args:
        goal:           The task instruction (set once per episode, repeated every turn)
        obs_text:       Full text from BrowserEnv.get_observation() for this step
        action_history: All action lines taken so far this episode (empty on first step)
    """
    history_block = (
        "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(action_history))
        if action_history
        else "  (none)"
    )

    return (
        f"GOAL: {goal}\n\n"
        f"HISTORY:\n{history_block}\n\n"
        f"PAGE:\n{obs_text}"
    )


if __name__ == "__main__":
    # test- print what the model will see on the first turn
    sys_msg = build_system_prompt()
    user_msg = build_user_message(
        goal="Find the Search bar and search for 'playwright'",
        obs_text="URL: https://google.com\nTitle: Google\nInteractive elements: 1\n\n[0] input | Search | Search bar | Search input",
        action_history=[],
    )
    print("SYSTEM PROMPT:")
    print(sys_msg)
    print("\nUSER MESSAGE (turn 1):")
    print(user_msg)
