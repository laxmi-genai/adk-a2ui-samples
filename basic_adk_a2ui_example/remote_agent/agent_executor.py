
import json
import logging
import re
from a2a import types as a2a_types
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import UnsupportedOperationError, Task
from a2a.utils import new_agent_parts_message, new_task
from google.adk.runners import Runner
from google.adk.utils.context_utils import Aclosing
from google.genai import types

from .a2ui_extension import create_a2ui_part, try_activate_a2ui_extension

logger = logging.getLogger(__name__)

def process_llm_output_to_a2ui_parts(llm_output: str) -> list[a2a_types.Part]:
    """
    Parses the LLM output, expecting a JSON string containing A2UI messages.
    Converts the JSON data into a list of A2A DataParts with the A2UI MIME type.
    """
    if not llm_output:
        logger.warning("LLM output is empty.")
        return []

    llm_output = llm_output.strip()
    logger.info(f"LLM output: {llm_output}")
    # Find the start of JSON content to ignore markdown or conversational filler
    match = re.search(r'[\{\[]', llm_output)
    if not match:
        logger.warning(f"No JSON start character found in LLM output: {llm_output}")
        return [a2a_types.Part(root=a2a_types.TextPart(text=llm_output))]

    json_str = llm_output[match.start():]
    logger.info(f"JSON string: {json_str}")
    data = None
    try:
        # Use raw_decode to stop safely at the end of the JSON object/array
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"raw_decode failed: {e}. Attempting fallback parsing.")
        # Fallback: find the last matching bracket and try strict loads
        end_match = max(json_str.rfind('}'), json_str.rfind(']'))
        if end_match != -1:
            try:
                data = json.loads(json_str[:end_match+1])
            except json.JSONDecodeError as e2:
                logger.error(f"Fallback parsing failed: {e2}")
    
    if data is None:
        logger.warning(f"Could not extract valid JSON from LLM output: {llm_output}")
        return [a2a_types.Part(root=a2a_types.TextPart(text=llm_output))]

    a2ui_messages = []
    if isinstance(data, list):
        a2ui_messages = data
    elif isinstance(data, dict):
        if "messages" in data and isinstance(data["messages"], list):
            a2ui_messages = data["messages"]
        else:
            # Assume the dict itself is a single A2UI message
            a2ui_messages = [data]
    else:
        logger.warning(f"Parsed JSON is neither list nor dict: {type(data)}")
        return [a2a_types.Part(root=a2a_types.TextPart(text=str(data)))]

    final_parts = []
    for message in a2ui_messages:
        final_parts.append(create_a2ui_part(message))
    
    if not final_parts:
         logger.warning("No A2UI parts generated from LLM output.")

    logger.info(f"Generated A2UI parts: {final_parts}")  
    return final_parts

class A2UIAgentExecutor(AgentExecutor):
    """Generic executor for A2UI agents that expect JSON output."""

    def __init__(self, runner: Runner):
        super().__init__()
        self._runner = runner

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input_text = context.get_user_input() or ""
        
        # Check if there is an A2UI C2S userAction payload in the message parts
        user_action_data = None
        if context.message and context.message.parts:
            for part in context.message.parts:
                if isinstance(part.root, a2a_types.DataPart) and "userAction" in part.root.data:
                    action_payload = part.root.data["userAction"]
                    action_name = action_payload.get("name")
                    action_context = action_payload.get("context", [])
                    
                    # Convert A2UI v0.8 context array to a simple dictionary for the LLM
                    context_dict = {}
                    for item in action_context:
                        key = item.get("key")
                        val = item.get("value", {})
                        if "literalString" in val:
                            context_dict[key] = val["literalString"]
                        elif "literalNumber" in val:
                            context_dict[key] = val["literalNumber"]
                        elif "literalBoolean" in val:
                            context_dict[key] = val["literalBoolean"]
                        elif "path" in val:
                            context_dict[key] = f"{{path: {val['path']}}}"

                    user_action_data = f"User initiated action: '{action_name}'. Action context data: {json.dumps(context_dict)}. Please use the appropriate tool to handle this."
                    logger.info(f"Extracted action: {user_action_data}")
                    break

        if not user_input_text and not user_action_data:
            return

        # Activate the A2UI extension for this request context
        try_activate_a2ui_extension(context)

        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        await updater.update_status(a2a_types.TaskState.working)

        # Create input content for the agent
        msg_text = user_input_text
        if user_action_data:
            msg_text = f"{msg_text}\n{user_action_data}".strip()
            logger.info(f"Injecting user action into LLM text prompt: {user_action_data}")

        new_message = types.Content(role='user', parts=[types.Part(text=msg_text)])
        
        accumulated_text = ""
        
        try:
            # Run the agent using the ADK Runner
            async with Aclosing(self._runner.run_async(
                user_id="user", # Default user ID
                session_id=context.context_id,
                new_message=new_message
            )) as agen:
                async for event in agen:
                    # Capture text from the model response
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                accumulated_text += part.text
        except Exception as e:
            logger.error(f"Error executing agent: {e}", exc_info=True)
            # Fail gracefully
            await updater.update_status(
                 a2a_types.TaskState.failed,
                 new_agent_parts_message([a2a_types.Part(root=a2a_types.TextPart(text=f"Error: {str(e)}"))], task.context_id, task.id),
                 final=True
            )
            return

        # Process the LLM's final accumulated output string
        final_a2a_parts = process_llm_output_to_a2ui_parts(accumulated_text)

        await updater.update_status(
            a2a_types.TaskState.completed,
            new_agent_parts_message(final_a2a_parts, task.context_id, task.id),
            final=True,
        )

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
