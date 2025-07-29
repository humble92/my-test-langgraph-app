"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from typing_extensions import TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph


class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    messages: List[BaseMessage]


model = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7
)

async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output using GPT-4.1-nano model.

    Can use runtime configuration to alter behavior.
    """
    configuration = config["configurable"]
    config_param = configuration.get('my_configurable_param', 'not specified')
    
    if state.messages:
        last_message = state.messages[-1]
        if hasattr(last_message, 'content'):
            # BaseMessage object
            user_input = last_message.content
        elif isinstance(last_message, dict) and 'content' in last_message:
            # Dictionary format from humble.py
            user_input = last_message['content']
        else:
            user_input = str(last_message)
    else:
        user_input = "No input provided"
    
    # Create system message and user message for the model
    # human->user: ai-human (Langchain) format -> user-assistant (OpenAI API) format
    messages = [
        {"role": "system", "content": f"You are a helpful AI assistant. Configuration context: {config_param}"},
        {"role": "user", "content": user_input}
    ]
    
    response = await model.ainvoke(messages)
    
    # Return the new AI message
    return {
        "messages": [AIMessage(content=response.content)]
    }


# Define the graph
graph = (
    StateGraph(State, config_schema=Configuration)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .add_edge("call_model", "__end__")  # End of the graph, explicitly
    .compile(name="New Graph")
)
