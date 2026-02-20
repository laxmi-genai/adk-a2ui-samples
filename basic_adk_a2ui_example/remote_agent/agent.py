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

"""A2UI agent."""

import json
import logging

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext
from .a2ui_schema import A2UI_SCHEMA

import uvicorn
from starlette.applications import Starlette

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.runners import Runner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from .agent_executor import A2UIAgentExecutor

load_dotenv()

def get_items(tool_context: ToolContext) -> str:
    """Call this tool to get the list of items to choose from."""
    return json.dumps([
        {
            "name": "Eiffel Tower",
            "country": "France",
            "description": "An iconic wrought-iron lattice tower on the Champ de Mars in Paris, named after the engineer Gustave Eiffel. Built for the 1889 World's Fair, it has become a global cultural symbol of France.",
            "image_url": "https://www.publicdomainpictures.net/pictures/80000/velka/paris-eiffel-tower-1393841654WTb.jpg",
            "wikipedia_link": "https://en.wikipedia.org/wiki/Eiffel_Tower"
        },
        {
            "name": "Taj Mahal",
            "country": "India",
            "description": "An ivory-white marble mausoleum on the right bank of the river Yamuna in Agra. It was commissioned in 1632 by the Mughal emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal.",
            "image_url": "https://www.publicdomainpictures.net/pictures/180000/velka/taj-mahal.jpg",
            "wikipedia_link": "https://en.wikipedia.org/wiki/Taj_Mahal"
        },
        {
            "name": "Statue of Liberty",
            "country": "USA",
            "description": "A colossal neoclassical sculpture on Liberty Island in New York Harbor. A gift from the people of France to the United States, it depicts Libertas, the Roman goddess of liberty, holding a torch.",
            "image_url": "https://www.publicdomainpictures.net/pictures/210000/velka/statue-of-liberty-1485195709Nms.jpg",
            "wikipedia_link": "https://en.wikipedia.org/wiki/Statue_of_Liberty"
        }
    ])

def select_item(tool_context: ToolContext, userAction: str) -> str:
    """Call this tool to process the user's selection. It does nothing useful here."""
    return "Selection received and processed."

AGENT_INSTRUCTION="""
You are a location selector assistant. Your goal is to help users select a location from a list of options using a rich UI.

To achieve this, you MUST follow these steps to answer user requests:

1. Check whether the message request is an initial request for options (natural language) or a user action selecting an option (a JSON payload that is a userAction of A2UI C2S JSON SCHEMA below).
2. If it is an initial request, you MUST call the `get_items` tool to retrieve the list of items to choose from.
3. If it is a user action with name "select", you MUST call the `select_item` tool with the complete userAction JSON object received.
4. If it is neither an initial request nor a user action selecting an option, you MUST do nothing.
5. You MUST respond with a rich A2UI UI S2C JSON to present options, confirm selections with all available details, or do nothing depending on the context.
"""

A2UI_AND_AGENT_INSTRUCTION = AGENT_INSTRUCTION + """

To generate a valid A2UI UI S2C JSON, you MUST strictly follow the JSON SCHEMA below and these rules:

1.  Your response MUST be a single, raw JSON object which is a list of A2UI messages.
2.  You MUST ALWAYS send a `beginRendering` message first to define the `surfaceId` and the `root` component ID (e.g. your main `Column`).
3.  You MUST ALWAYS send a `surfaceUpdate` message second that contains all the UI components, including the `root` container (e.g., `Column`) that holds the IDs of all other items via `children.explicitList`.

To represent the items, you MUST only use the A2UI message types Image, Divider, Row, Column, and Text, following these conventions:
1.  Image MUST be used to prominently display the `image_url` for every option. Make sure to display the images.
2.  Divider MUST be used to separate different items.
3.  Texts MUST be used for descriptions and option names. Do NOT use Markdown formatting (like `**`) or HTML tags (like `<b>`) for bolding. Just use standard literal text. Rely on `usageHint: "h3"` on the Text component for the prominent item names, and `usageHint: "body"` for descriptions.
4.  CRITICAL: Do NOT use Buttons. The user only wants to see images and text.

---BEGIN A2UI S2C JSON SCHEMA---
{
  "properties": {
    "beginRendering": {
      "properties": { "surfaceId": {"type": "string"}, "root": {"type": "string"} }
    },
    "surfaceUpdate": {
      "properties": {
        "surfaceId": {"type": "string"},
        "components": {
          "type": "array",
          "items": {
            "properties": {
              "id": { "type": "string" },
              "component": {
                "properties": {
                  "Text": { "properties": { "text": {"properties": {"literalString": {"type": "string"}}}, "usageHint": {"type": "string", "enum": ["h3", "body"]} } },
                  "Image": { "properties": { "url": {"properties": {"literalString": {"type": "string"}}} } },
                  "Divider": { "type": "object" },
                  "Column": { "properties": { "children": {"properties": {"explicitList": {"type": "array"}}} } },
                  "Row": { "properties": { "children": {"properties": {"explicitList": {"type": "array"}}} } }
                }
              }
            }
          }
        }
      }
    }
  }
}
---END A2UI S2C JSON SCHEMA---
"""

root_agent = LlmAgent(
    name="item_selector_agent",
    model="gemini-2.5-flash",
    instruction=A2UI_AND_AGENT_INSTRUCTION,
    description="An agent to handle item selection from a list.",
    tools=[get_items, select_item]
)

# Execute the server
if __name__ == "__main__":

    # 1. Create the Runner (Standard ADK runner)
    runner = Runner(
        app_name=root_agent.name,
        agent=root_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        credential_service=InMemoryCredentialService(),
        auto_create_session=True, # Important: Auto-create session if missing
    )

    # 2. Create the Custom Executor, passing the runner
    executor = A2UIAgentExecutor(runner) 

    # 3. Create Request Handler with Custom Executor
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )

    # 4. Build Agent Card
    host = "localhost"
    port = 10001
    protocol = "http"
    rpc_url = f"{protocol}://{host}:{port}/"
    
    card_builder = AgentCardBuilder(agent=root_agent, rpc_url=rpc_url)

    # 5. Create Starlette App using A2A utilities
    app = Starlette()

    async def setup_a2a():
        agent_card = await card_builder.build()
        a2a_app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )
        a2a_app.add_routes_to_app(app)
        print(f"Agent running at {rpc_url}")

    app.add_event_handler("startup", setup_a2a)

    uvicorn.run(app, host=host, port=port)


