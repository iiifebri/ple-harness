# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A minimal harness: one model, three tools, one loop.

The model gets the task and three tools that act in the task container. Each
tool call runs there and its result goes back to the model, until the model
answers without calling a tool. That answer is the final message.
"""

import asyncio
import json
import os
import re
import time

import openai
import taskbridge

MODEL = os.environ["MODEL_NAME"]
# The model's highest reasoning level: max for Claude, GLM, GPT-6 and
# DeepSeek-V4, high for the others (Gemini refuses max). REASONING_EFFORT
# overrides it.
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)
WRAP_UP_SEC = 240  # ask for the summary this long before the time limit
WRAP_UP = {
    "role": "user",
    "content": "Time is almost up. Stop and reply with your summary.",
}
DRAFT_FRACTION = 0.5  # ask for a first draft once this share of time is used
DRAFT_STEP = 12  # ask for a first draft once this many agent steps pass
# Seconds a streamed turn may go without a token before it is aborted.
FIRST_TOKEN_TIMEOUT = 600
DRAFT = {
    "role": "user",
    "content": (
        "You are past the halfway point of your time limit and no"
        " deliverable file exists yet. Stop reading and write a complete"
        " draft of every deliverable now (including submission.zip), then"
        " keep refining it. A draft on disk is worth far more than a"
        " perfect plan with no output."
    ),
}
RESULT_CHARS = 20_000  # a longer tool result loses its middle
# Seconds between retries once the SDK's own quick retries are exhausted.
RETRY_WAITS = (5, 15, 30, 60, 120)
# How endpoints word a 400 saying the conversation no longer fits the context
# window: Claude, OpenAI and vLLM, Gemini, xAI.
CONTEXT_FULL = re.compile(
    r"context.?(length|window)|prompt is too long|maximum prompt length"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
CUT_KEEP = 1000  # characters a tool result keeps once the context is full
# times in a row an empty or cut-off reply gets a "continue" before the run
# stops
MAX_NUDGES = 3
# lower it for a small context window
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
# Claude caches a prompt prefix only when marked with cache_control. Each call
# marks the system prompt and the latest message (using 2 of Claude's 4 cache
# breakpoints) so each turn reads the prior conversation from cache. Other
# providers cache automatically without markers.
CACHE_MARK = "claude" in MODEL.lower()

# The evaluator sets OPENAI_BASE_URL and OPENAI_API_KEY for every model.
# The SDK handles quick retries; recover() handles longer backoffs. Calls
# stream, so bytes arrive continuously and the egress proxy's 30-minute idle
# limit is never hit; a turn with no bytes for FIRST_TOKEN_TIMEOUT seconds is
# aborted and retried. The 1800 s timeout is the outer bound.
client = openai.AsyncOpenAI(max_retries=5, timeout=1800)

SYSTEM = """You are an expert engineer working on a task in a Linux container with no internet access. Your tools \
act inside it, and bash starts in {workdir}. Read the task and the files it points to before you start. Compute with \
code, not by hand. Batch all reading into one bash command per turn: never spend a turn on a single file or a single \
sed range; combine every file, grep, sed range, and conversion you need into one command and read the combined \
output. Work to a time-boxed plan: write a first complete draft of every deliverable (including submission.zip) \
within the first 60 minutes; this is a hard requirement, not a guideline; then refine it; never spend more than ~40% \
of the remaining time only reading. Write each deliverable exactly where and in the format the task asks, then open \
it again to check it. When you are done, reply without calling a tool: a short summary of each deliverable, where it \
is, and the key results."""


def tool(name, description, **params):
  """Builds an OpenAI function-tool definition dictionary.

  Args:
    name: Tool function name.
    description: Human-readable tool description.
    **params: Parameter names mapped to their JSON Schema type strings.

  Returns:
    An OpenAI function-tool specification dictionary.
  """
  return {
      "type": "function",
      "function": {
          "name": name,
          "description": description,
          "parameters": {
              "type": "object",
              "properties": {k: {"type": v} for k, v in params.items()},
              "required": list(params),
          },
      },
  }


TOOLS = [
    tool(
        "bash",
        "Run a bash command (120 s limit) and return its exit code and output.",
        command="string",
    ),
    tool("read_file", "Read a text file.", path="string"),
    tool(
        "write_file",
        "Create or overwrite a file; parent folders are created.",
        path="string",
        content="string",
    ),
]


def clip(text):
  """Truncates `text` in the middle if it exceeds `RESULT_CHARS` characters.

  Args:
    text: Tool output string to clip.

  Returns:
    The original string if within `RESULT_CHARS`, or a middle-truncated string.
  """
  if len(text) <= RESULT_CHARS:
    return text
  half = RESULT_CHARS // 2
  return (
      f"{text[:half]}\n[... {len(text) - RESULT_CHARS} characters cut"
      f" ...]\n{text[-half:]}"
  )


def marked(messages):
  """Returns the messages a model call sends.

  For Claude, this is a copy whose system prompt and newest message carry
  `cache_control`. The stored conversation stays as it is, so each turn marks
  its own newest message.

  Args:
    messages: The conversation so far, in OpenAI chat format.

  Returns:
    `messages` itself for other models, or the marked copy for Claude.
  """
  if not CACHE_MARK:
    return messages
  out = list(messages)
  for i in {0, len(out) - 1}:
    m, content = out[i], out[i].get("content")
    if isinstance(content, str) and content:
      out[i] = {
          **m,
          "content": [{
              "type": "text",
              "text": content,
              "cache_control": {"type": "ephemeral"},
          }],
      }
    else:
      out[i] = {**m, "cache_control": {"type": "ephemeral"}}
  return out


async def run_tool(session, fn):
  """Executes a single tool call in the task container and returns its output.

  Rewrites `fn["arguments"]` as canonical JSON (or `"{}"` if invalid) so the
  conversation history remains valid for strict endpoints. Any tool failure is
  returned as an error string for the model.

  Args:
    session: Active `taskbridge.Session` connected to the task container.
    fn: Function-call dictionary containing `name` and `arguments`.

  Returns:
    The clipped tool output or error string to send back to the model.
  """
  name = fn["name"]
  try:
    args = json.loads(fn["arguments"] or "{}")
    if not isinstance(args, dict):
      raise TypeError("not a JSON object")
    fn["arguments"] = json.dumps(args, allow_nan=False)
  except (TypeError, ValueError) as exc:
    fn["arguments"] = "{}"
    return (
        f"error: the arguments are not a JSON object ({exc}); send the call"
        " again"
    )
  try:
    if name == "bash":
      result = await session.exec(args["command"])
      return clip(f"exit code {result['exit_code']}\n{result['output']}")
    if name == "read_file":
      return clip(
          (await session.read_file(args["path"])).decode("utf-8", "replace")
      )
    if name == "write_file":
      return (
          f"wrote {await session.write_file(args['path'], args['content'])}"
          f" bytes to {args['path']}"
      )
    return f"error: there is no tool named {name}"
  # ValueError: content with an unpaired surrogate
  except (taskbridge.TaskError, KeyError, ValueError) as exc:
    return f"error: {type(exc).__name__}: {exc}"


async def recover(session, messages, exc, waits, until):
  """Handles a failed model call and returns whether to retry.

  On a 400 context-overflow error, shortens the oldest tool results. On a rate
  limit, server error, or connection drop, sleeps for the next interval in
  `RETRY_WAITS` (capped at the wrap-up deadline). Errors are logged rather than
  added to the conversation.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, shortened in place when the context is full.
    exc: The `openai.APIError` the call raised.
    waits: Iterator over the remaining intervals in `RETRY_WAITS`.
    until: Seconds before the time limit at which the wrap-up starts.

  Returns:
    `True` to retry the call, `False` to re-raise.
  """
  status = getattr(exc, "status_code", None)  # None: no HTTP answer
  if status == 400 and CONTEXT_FULL.search(str(exc)):
    return await shorten(session, messages)
  wait = next(waits, None)
  if wait is None or not (
      status is None or status in (408, 409, 429) or status >= 500
  ):
    return False
  left = session.time_left()
  if left is not None and left > until:
    wait = min(wait, left - until)
  await session.log(
      f"model call failed ({str(exc)[:200]}); again in {wait:.0f} s"
  )
  await asyncio.sleep(wait)
  return True


async def shorten(session, messages):
  """Truncates the older half of long tool results to `CUT_KEEP` characters.

  Repeated calls truncate progressively newer results. Claude's thinking blocks
  are kept intact because each block's signature is self-contained and Claude
  rejects tool loops if the latest assistant turn loses its thinking block.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, modified in place.

  Returns:
    Whether any tool result was truncated.
  """
  long = [
      n
      for n, m in enumerate(messages)
      if m["role"] == "tool" and len(m["content"]) > 2 * CUT_KEEP
  ]
  if not long:
    return False
  cut = long[: max(1, len(long) // 2)]
  for n in cut:
    messages[n]["content"] = (
        messages[n]["content"][:CUT_KEEP]
        + "\n[... cut to save context; run the tool again if needed]"
    )
  await session.log(f"context full: cut {len(cut)} old tool results")
  return True


async def has_deliverable(session, marker):
  """Returns whether the agent has written any file since the run started.

  Checks the container paths that can hold a deliverable (the task world's
  `/filesystem` and `/app` trees) for files newer than `marker`, a file
  touched in the container at the start of the run. When `marker` is None
  (the touch failed), falls back to looking for any submission.zip under
  those trees. A failed check reports False so the draft prompt still fires.

  Args:
    session: Active `taskbridge.Session` for the task.
    marker: Path of the start-of-run marker file in the container, or None.

  Returns:
    True when a file written after the run started was found, else False.
  """
  if marker is not None:
    command = (
        "find /filesystem /app -maxdepth 6 -type f -newer "
        f"{marker} 2>/dev/null | head -1"
    )
  else:
    command = (
        "find /filesystem /app -maxdepth 6 -name submission.zip -type f"
        " 2>/dev/null | head -1"
    )
  try:
    result = await session.exec(command)
  except taskbridge.Cancelled:
    raise
  except Exception:  # pylint: disable=broad-exception-caught
    return False
  return bool((result.get("output") or "").strip())


class StalledTurn(Exception):
  """A streamed model turn produced no token within FIRST_TOKEN_TIMEOUT."""


class StreamOptionsRejected(Exception):
  """The endpoint rejected stream_options; retry the turn without it."""


def _step_line(step, wall, started, usage):
  """Formats one step-timing log line for the agent log.

  Args:
    step: The agent step number (one loop iteration each).
    wall: Wall-clock start time of the step (seconds since the epoch).
    started: Monotonic start time of the step.
    usage: The step's model-turn token usage, or None when unreported.

  Returns:
    A one-line summary: step number, start time, duration, tokens.
  """
  took = time.monotonic() - started
  stamp = time.strftime("%H:%M:%S", time.localtime(wall))
  tokens = (
      f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens"
      if usage is not None
      else "usage not reported"
  )
  return f"step {step}: started {stamp}, took {took:.1f} s, {tokens}"


async def call_model(session, messages, wrap_up, use_usage):
  """Streams one model turn and returns (turn, cut, usage).

  `turn` is the assistant message as a dict in the shape the non-streaming
  call returned (role, content, reasoning_content when present, tool_calls),
  so the caller appends it to the conversation unchanged. `cut` is True when
  the reply stopped at the output limit. `usage` is the final chunk's token
  usage when the endpoint reports it, else None.

  Chunks keep the egress connection alive: a turn with no chunk for
  FIRST_TOKEN_TIMEOUT seconds is aborted as StalledTurn so the caller retries
  it through recover(). Reasoning deltas are accumulated separately; they are
  not output text.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation to send, already cache-marked.
    wrap_up: Whether the wrap-up deadline applies (forces tool_choice none).
    use_usage: Whether to ask for usage in the final chunk.

  Returns:
    (turn, cut, usage) as described above.

  Raises:
    StalledTurn: No chunk arrived within FIRST_TOKEN_TIMEOUT.
    StreamOptionsRejected: The endpoint rejected stream_options.
    openai.APIError: Any other endpoint error, for the caller to recover().
  """
  started = time.monotonic()
  wall_start = time.time()
  kwargs = {
      "model": MODEL,
      "messages": messages,
      "tools": TOOLS,
      "tool_choice": "none" if wrap_up else "auto",
      "reasoning_effort": EFFORT,
      "max_completion_tokens": MAX_OUTPUT_TOKENS,
      "stream": True,
  }
  if use_usage:
    kwargs["stream_options"] = {"include_usage": True}
  try:
    stream = await client.chat.completions.create(**kwargs)
  except openai.APIError as exc:
    if (
        use_usage
        and getattr(exc, "status_code", None) == 400
        and "stream_options" in str(exc).lower()
    ):
      raise StreamOptionsRejected(str(exc))
    raise
  content, reasoning = "", ""
  tool_calls = {}
  finish_reason = None
  usage = None
  first_token = None
  chunks = 0
  try:
    while True:
      try:
        chunk = await asyncio.wait_for(
            stream.__anext__(), FIRST_TOKEN_TIMEOUT
        )
      except StopAsyncIteration:
        break
      except asyncio.TimeoutError:
        raise StalledTurn(
            f"no model tokens for {FIRST_TOKEN_TIMEOUT} s; aborting the turn"
        )
      chunks += 1
      if first_token is None:
        first_token = time.monotonic() - started
      if chunk.usage is not None:
        usage = chunk.usage
      if not chunk.choices:
        continue
      if chunk.choices[0].finish_reason is not None:
        finish_reason = chunk.choices[0].finish_reason
      delta = chunk.choices[0].delta
      if delta is None:
        continue
      if delta.content:
        content += delta.content
      reasoning_delta = getattr(delta, "reasoning_content", None)
      if reasoning_delta:
        reasoning += reasoning_delta
      for call in delta.tool_calls or []:
        part = tool_calls.setdefault(
            call.index,
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        if call.id:
          part["id"] = call.id
        if call.function:
          if call.function.name:
            part["function"]["name"] = call.function.name
          if call.function.arguments:
            part["function"]["arguments"] += call.function.arguments
  finally:
    await stream.close()
  total = time.monotonic() - started
  await session.log(
      "model turn: started "
      + time.strftime("%H:%M:%S", time.localtime(wall_start))
      + (
          f", first token {first_token:.1f} s"
          if first_token is not None
          else ", no tokens"
      )
      + f", total {total:.1f} s, {chunks} chunks, {len(content)} chars"
      + (
          f", {usage.prompt_tokens} in / {usage.completion_tokens} out tokens"
          if usage is not None
          else ", usage not reported"
      )
  )
  turn = {"role": "assistant"}
  if content:
    turn["content"] = content
  if reasoning:
    turn["reasoning_content"] = reasoning
  if tool_calls:
    turn["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
  return turn, finish_reason == "length", usage


async def solve(session):
  """Runs the minimal tool-calling loop on `session` until completion.

  Args:
    session: Active `taskbridge.Session` for the task.

  Returns:
    The final summary message from the model.
  """
  messages = [
      {"role": "system", "content": SYSTEM.format(workdir=session.workdir)},
      {"role": "user", "content": session.instruction},
  ]
  # head: the start of an answer cut off at the output limit
  nudges, head, waits = 0, "", iter(RETRY_WAITS)
  use_usage = True  # ask for usage in the final stream chunk; cleared if rejected
  step = 0  # agent steps so far (one loop iteration each)
  step_started = step_wall = step_usage = None
  total = session.time_left()  # the full budget; None when there is no limit
  marker = "/tmp/.plebench-start-marker"
  try:
    await session.exec(f"touch {marker}")
  except taskbridge.Cancelled:
    raise
  except Exception:  # pylint: disable=broad-exception-caught
    marker = None
  draft_sent = False
  while True:
    if step_started is not None:  # log the step that just finished
      await session.log(
          _step_line(step, step_wall, step_started, step_usage)
      )
    step += 1
    step_started = time.monotonic()
    step_wall = time.time()
    step_usage = None
    left = session.time_left()  # None when the task has no time limit
    wrap_up = left is not None and left < WRAP_UP_SEC
    if wrap_up and messages[-1:] != [WRAP_UP]:  # asked once, also on a retry
      messages.append(dict(WRAP_UP))
    if not draft_sent and (
        (
            total is not None
            and left is not None
            and left < total * (1 - DRAFT_FRACTION)
        )
        or step >= DRAFT_STEP
    ):
      draft_sent = True
      if not await has_deliverable(session, marker):
        messages.append(dict(DRAFT))
    try:  # a call still running when the wrap-up starts is stopped
      turn, cut, usage = await asyncio.wait_for(
          call_model(session, marked(messages), wrap_up, use_usage),
          None if left is None or wrap_up else left - WRAP_UP_SEC,
      )
    except StalledTurn as exc:
      if await recover(session, messages, exc, waits, WRAP_UP_SEC):
        continue
      raise
    except StreamOptionsRejected as exc:
      await session.log(
          f"stream_options rejected ({str(exc)[:200]}); retrying without usage"
      )
      use_usage = False
      continue
    except asyncio.TimeoutError:
      continue
    except openai.APIError as exc:
      if await recover(session, messages, exc, waits, WRAP_UP_SEC):
        continue
      raise
    waits = iter(RETRY_WAITS)
    step_usage = usage
    if usage is not None:
      details = getattr(usage, "prompt_tokens_details", None)
      await session.usage(
          MODEL,
          usage.prompt_tokens,
          usage.completion_tokens,
          getattr(details, "cached_tokens", 0) or 0,
      )
    text = head + (turn.get("content") or "")
    if text:
      # what taskbridge sends if the time runs out first
      session.final_message = text
    # Send the message back as the endpoint returned it: Gemini refuses a tool
    # call that comes back without its thought signature, which is one of the
    # endpoint's own fields.
    if wrap_up or not turn.get("tool_calls"):
      if wrap_up or (turn.get("content") and not cut) or nudges == MAX_NUDGES:
        await session.log(
            _step_line(step, step_wall, step_started, step_usage)
        )
        return session.final_message
      nudges, head = nudges + 1, text
      if turn.get("content"):  # cut off at the output limit
        messages.append(turn)
      messages.append({
          "role": "user",
          "content": (
              "Your reply was cut off; continue where it stopped."
              if cut
              else "Continue the task, and finish with your summary."
          ),
      })
      continue
    nudges, head = 0, ""
    messages.append(turn)
    for call in turn["tool_calls"]:
      await session.log(
          f"{call['function']['name']} {call['function']['arguments'][:200]}"
      )
      content = await run_tool(session, call["function"])
      messages.append(
          {"role": "tool", "tool_call_id": call["id"], "content": content}
      )
    if left is not None:
      messages[-1][
          "content"
      ] += f"\n[{int(session.time_left() // 60)} minutes left]"


if __name__ == "__main__":
  taskbridge.run(solve)
