"""Prompt and action schema shared by official eval and MoT training."""

import json

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of actions to complete the task. You need to choose actions from the the following list:
action_type: Click, action_target: Element description, value: None, point_2d: [x, y]
    ## Explanation: Tap or click a specific UI element and provide its coordinates

action_type: Write, action_target: Element description or None, value: Text to enter, point_2d: [x, y]
    ## Explanation: Enter text into a specific input field or at the current focus if coordinate is None

action_type: LongPress, action_target: Element description, value: None, point_2d: [x, y]
    ## Explanation: Press and hold on a specific UI element (mobile only) and provide its coordinates

action_type: Scroll, action_target: None, value: "up" | "down" | "left" | "right", point_2d: None
    ## Explanation: Scroll a view or container in the specified direction

action_type: Wait, action_target: None, value: Number of seconds, point_2d: None
    ## Explanation: Pause execution to allow the UI to load or update

action_type: NavigateBack, action_target: None, value: None, point_2d: None
    ## Explanation: Press the system "Back" button

action_type: OpenApp, action_target: None, value: App name, point_2d: None
    ## Explanation: Launch an app by its name (mobile only)

action_type: Terminate, action_target: None, value: End-task message, point_2d: None
    ## Explanation: Signal the end of the current task with a final message
"""

RESPONSE_TEMPLATE = """The response should be structured in the following format:
<thinking>Your step-by-step thought process here...</thinking>
<answer>
{
  "action_type": "the type of action to perform, e.g., Click, Write, Scroll, Answer, etc. Please follow the system prompt for available actions.",
  "action_target": "the description of the target of the action, such as the color, text, or position on the screen of the UI element to interact with",
  "value": "the input text or direction ('up', 'down', 'left', 'right') for the 'scroll' action, if applicable; otherwise, use 'None'",
  "point_2d": [x, y]
}
</answer>"""


def official_query(row, image_size):
    history = "".join(
        f"\nStep {index + 1}\n Action: {official_history_action(action)}\n"
        for index, action in enumerate(row.get("previous_actions", []) or [])
    )
    size = f"(original image size {image_size[0]}x{image_size[1]})"
    question = (
        f"Please generate the next move according to the UI screenshot {size}, "
        "instruction and previous actions.\n\n"
        f"Instruction: {row['step_instruction']}\n\nInteraction History: {history}\n"
    )
    return question + "\n" + RESPONSE_TEMPLATE


def official_history_action(action):
    """Render a manifest action as an official-style history entry."""
    if isinstance(action, str):
        return action
    kind = action["type"]
    if kind in {"click", "long_press"}:
        x, y = round(action["x"] * 999), round(action["y"] * 999)
        return f"{'Click' if kind == 'click' else 'Long press'} at [{x}, {y}]"
    if kind == "scroll":
        return f"Scroll {action['direction']}"
    if kind == "input_text":
        return f"Write {action['text']}"
    if kind == "open_app":
        return f"Open app {action['app_name']}"
    return {
        "navigate_home": "Go to the home screen",
        "navigate_back": "Go back to the previous screen",
        "wait": "Wait",
    }[kind]


def official_action_text(action, instruction):
    """Supervise the fields parsed by the official evaluator."""
    kind = action["type"]
    action_type = {
        "click": "Click",
        "long_press": "LongPress",
        "scroll": "Scroll",
        "input_text": "Write",
        "open_app": "OpenApp",
        "navigate_back": "NavigateBack",
        "wait": "Wait",
    }[kind]
    point = None
    target = "None"
    value = "None"
    if kind in {"click", "long_press"}:
        point = [round(action["x"] * 999), round(action["y"] * 999)]
        target = instruction
    elif kind == "scroll":
        value = action["direction"]
    elif kind == "input_text":
        target = instruction
        value = action["text"]
    elif kind == "open_app":
        value = action["app_name"]
    payload = {"action_type": action_type, "action_target": target, "value": value, "point_2d": point}
    # The manifest has no teacher reasoning; keep the required wrapper empty.
    return "<thinking></thinking>\n<answer>\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n</answer>"
