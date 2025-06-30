"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import os
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.func import entrypoint
from langgraph.graph import StateGraph
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


@entrypoint()
def wrapper_agent(input_data: InputSalesData) -> dict:
    result = graph.invoke({"input_data": input_data})
    return {
        "processed_data": result["processed_data"],
        "recommendation": result["recommendation"].content,
    }
