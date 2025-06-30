"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import json
import os
import pprint
from typing import TypedDict, Annotated, Optional

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from src.agent.models import InputSalesData, OutputData
from src.agent.prompt import SystemPrompts, UserPrompts

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=1.0,
    max_retries=2,
    google_api_key=api_key,
)


class State(TypedDict):
    response: Annotated[list, add_messages]
    input_data: dict
    validated_data: Optional[InputSalesData] = None
    processed_data: Optional[OutputData] = None
    recommendation: Optional[str] = None


state_graph = StateGraph(State)


def input_handler(state: State):
    try:
        return {"validated_data": InputSalesData(**state.get("input_data"))}
    except ValidationError as e:
        print(f"Validation error: {e}")
        raise


def calculate_node(state: State):
    structured_llm = gemini.with_structured_output(OutputData)
    result = structured_llm.invoke(
        [
            SystemMessage(SystemPrompts.CALCULATE_DATA),
            HumanMessage(
                content=UserPrompts.INPUT_DATA.format(data=state.get("validated_data"))
            ),
        ]
    )
    return {"processed_data": result}


def recommendation_chat_bot(state: State):
    result = gemini.invoke(
        [
            SystemMessage(SystemPrompts.RECOMMENDATION),
            HumanMessage(
                content=UserPrompts.INPUT_DATA.format(data=state.get("processed_data"))
            ),
        ]
    )
    return {"recommendation": result}


state_graph.add_node("input_handler", input_handler)
state_graph.add_node("calculate_node", calculate_node)
state_graph.add_node("recommendation_chat_bot", recommendation_chat_bot)
state_graph.add_edge("__start__", "input_handler")
state_graph.add_edge("input_handler", "calculate_node")
state_graph.add_edge("calculate_node", "recommendation_chat_bot")
state_graph.add_edge("calculate_node", "__end__")

graph = state_graph.compile()

print("Please paste the input in this JSON format:\n")
print(
    """{
    "date": "2025-06-30",
    "sales": 150000,
    "costs": 10000,
    "customers_acquired": 60,
    "previous_day": {
        "sales": 12000,
        "costs": 8000,
        "customers_acquired": 35
    }
}"""
)

user_input_raw = input("\nPaste JSON input here:\n")
try:
    user_input = json.loads(user_input_raw)
except json.JSONDecodeError as e:
    print(f"\nInvalid JSON input: {e}")

result = graph.invoke({"input_data": user_input})
processed_data: OutputData = result["processed_data"]
print(pprint.pprint(processed_data.model_dump()))
print(result["recommendation"].content)
