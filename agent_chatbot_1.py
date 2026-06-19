import json
import os
import ast
import operator
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
# Set your key before running, e.g. on macOS/Linux:
#   export GEMINI_API_KEY="your-key-here"
# On Windows (PowerShell):
#   $env:GEMINI_API_KEY="your-key-here"

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable not set. "
        "Set it before running this script, e.g.\n"
        "  export GEMINI_API_KEY='your-key-here'   (macOS/Linux)\n"
        "  $env:GEMINI_API_KEY='your-key-here'      (Windows PowerShell)"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def get_weather(city: str):
    """Look up current weather for a city using wttr.in."""
    url = f"https://wttr.in/{city}?format=%C+%t"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text}."
        return "Something went wrong while fetching the weather."
    except requests.RequestException as e:
        return f"Weather lookup failed: {e}"


def run_command(cmd: str):
    """
    Execute a shell command — but only after the user explicitly confirms it.
    This prevents the LLM from silently running destructive commands.
    """
    print(f"\n⚠️  The assistant wants to run this shell command:\n    {cmd}")
    confirm = input("   Allow this? [y/N]: ").strip().lower()

    if confirm != "y":
        return "Command was not executed — user denied permission."

    exit_code = os.system(cmd)
    return f"Command executed with exit code {exit_code}."


# --- Safe calculator (no eval/exec) ----------------------------------------
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} not allowed.")
        return _ALLOWED_OPERATORS[op_type](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} not allowed.")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.operand))
    raise ValueError("Unsupported expression.")


def calculator(expression: str):
    """
    Safely evaluate a basic arithmetic expression (+, -, *, /, //, %, **).
    No access to names, attributes, function calls, or imports — unlike eval().
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def search_wikipedia(topic: str):
    """Search Wikipedia and return a summary of the top matching article."""
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "MyAI-Agent/1.0"}

    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": 1,
        "format": "json",
    }

    try:
        search_response = requests.get(url, params=search_params, headers=headers, timeout=10)
        search_response.raise_for_status()
    except requests.RequestException as e:
        return f"Wikipedia search failed: {e}"

    search_data = search_response.json()
    results = search_data.get("query", {}).get("search", [])

    if not results:
        return "No Wikipedia article found."

    title = results[0]["title"]

    content_params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
    }

    try:
        content_response = requests.get(url, params=content_params, headers=headers, timeout=10)
        content_response.raise_for_status()
    except requests.RequestException as e:
        return f"Wikipedia content fetch failed: {e}"

    content_data = content_response.json()
    page = next(iter(content_data["query"]["pages"].values()))
    article = page.get("extract", "No article content found")

    return f"Wikipedia Article: {title}\n\n{article[:3000]}"


available_tools = {
    "get_weather": get_weather,
    "run_command": run_command,
    "search_wikipedia": search_wikipedia,
    "calculator": calculator,
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a helpful AI Assistant who is specialized in resolving user queries.
You work in start, plan, action, observe mode.

For the given user query and available tools, plan the step by step execution,
based on the planning, select the relevant tool from the available tools, and
based on the tool selection you perform an action to call the tool.

Wait for the observation, and based on the observation from the tool call,
resolve the user query.

Rules:
- Follow the Output JSON Format strictly.
- Always perform one step at a time and wait for the next input.
- Carefully analyse the user query.
- Don't do all steps at once — go one by one, step by step.
- run_command may be denied by the user; if it is, explain that to the user
  in your final output instead of trying again.

Output JSON Format:
{
    "step": "string",
    "content": "string",
    "function": "The name of the function if the step is action",
    "input": "The input parameter for the function"
}

Available Tools:
- "get_weather": Takes a city name as input and returns the current weather for the city.
- "run_command": Takes a linux command as a string, asks the user for confirmation,
   and if approved executes it and returns the output.
- "search_wikipedia": Takes a topic name as input and returns information from Wikipedia.
- "calculator": Takes a basic arithmetic expression as a string (+, -, *, /, //, %, **)
   and returns the result. No variables or function calls allowed.

Example:
User Query: What is the weather of new york?
Output: { "step": "plan", "content": "The user is interested in weather data of new york" }
Output: { "step": "plan", "content": "From the available tools I should call get_weather" }
Output: { "step": "action", "function": "get_weather", "input": "new york" }
Output: { "step": "observe", "content": "12 Degree Cel" }
Output: { "step": "output", "content": "The weather for new york seems to be 12 degrees." }
"""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("AI Agent ready. Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("> ")
        if query.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": query})

        while True:
            response = client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"},
                messages=messages,
            )

            raw_content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": raw_content})

            try:
                parsed_response = json.loads(raw_content)
            except json.JSONDecodeError:
                print("⚠️  Model returned invalid JSON, stopping this turn.")
                break

            step = parsed_response.get("step")

            if step == "plan":
                print(f"🧠: {parsed_response.get('content')}")
                continue

            if step == "action":
                tool_name = parsed_response.get("function")
                tool_input = parsed_response.get("input")

                print(f"🛠️:  Calling Tool: {tool_name} with input {tool_input}")

                tool_fn = available_tools.get(tool_name)
                if tool_fn is None:
                    output = f"Unknown tool: {tool_name}"
                else:
                    output = tool_fn(tool_input)

                messages.append({
                    "role": "user",
                    "content": json.dumps({"step": "observe", "output": output}),
                })
                continue

            if step == "output":
                print(f"🤖: {parsed_response.get('content')}")
                break

            # Unrecognized step — avoid infinite loop
            print(f"⚠️  Unrecognized step '{step}', stopping this turn.")
            break


if __name__ == "__main__":
    main()
