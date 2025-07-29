from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")

async def main():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
        "messages": [{
            "role": "human",
            "content": "What is LangGraph Platform?",
            }],
        },
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

asyncio.run(main())


# Synchronous client

# from langgraph_sdk import get_sync_client

# client = get_sync_client(url="http://localhost:2024")

# for chunk in client.runs.stream(
#     None,  # Threadless run
#     "agent", # Name of assistant. Defined in langgraph.json.
#     input={
#         "messages": [{
#             "role": "human",
#             "content": "What is LangGraph?",
#         }],
#     },
#     stream_mode="messages-tuple",
# ):
#     print(f"Receiving new event of type: {chunk.event}...")
#     print(chunk.data)
#     print("\n\n")