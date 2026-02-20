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


import httpx
from google.adk.agents import remote_a2a_agent

# Hard code the extension header to use A2UI.
custom_headers = {'X-A2A-Extensions': 'https://a2ui.org/a2a-extension/a2ui/v0.8'}
client = httpx.AsyncClient(headers=custom_headers, timeout=300.0)

# A2A Agent URL (Assuming it is running on localhost:10001)
a2a_agent_url = 'http://localhost:10001' 

# Create the proxying agent.
root_agent = remote_a2a_agent.RemoteA2aAgent(
    name="a2a_proxy_agent",
    agent_card=f"{a2a_agent_url}/.well-known/agent.json",
    httpx_client=client,
)
