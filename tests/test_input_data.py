import pytest
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from src.agent.graph import (
    input_handler,
    calculate_node,
    recommendation_chat_bot,
    wrapper_agent,
)
from src.agent.models import PreviousDayData, InputSalesData, OutputData


@pytest.fixture(autouse=True)
def dummy_env(monkeypatch):
    # Provide a dummy GEMINI_API_KEY for any imports/use
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")


@pytest.fixture
def sample_input_data():
    return {
        "date": "2025-06-30",
        "sales": 1500,
        "costs": 500,
        "customers_acquired": 75,
        "previous_day": {
            "sales": 1200,
            "costs": 400,
            "customers_acquired": 60,
        },
    }


@pytest.fixture
def sample_output_data():
    # Mirror the updated OutputData model
    return OutputData(
        daily_profit=1000.0,
        revenue_change_percentage=25.0,
        cost_change_percentage=25.0,
        today_CAC=6.67,
        CAC_change_percentage=11.11,
        CAC_alert=False,
    )


def test_input_handler_valid(sample_input_data):
    state = {"input_data": sample_input_data}
    out = input_handler(state)
    assert "validated_data" in out
    validated = out["validated_data"]
    assert isinstance(validated, InputSalesData)
    assert validated.sales == 1500
    # Check nested PreviousDayData
    assert isinstance(validated.previous_day, PreviousDayData)
    assert validated.previous_day.customers_acquired == 60


def test_input_handler_invalid_missing_sales():
    # Missing required field 'sales'
    state = {"input_data": {"date": "2025-06-30"}}
    with pytest.raises(ValidationError):
        input_handler(state)


def test_calculate_node_invokes_llm(monkeypatch, sample_input_data, sample_output_data):
    validated = InputSalesData(**sample_input_data)
    state = {"validated_data": validated}

    class DummyStructuredLLM:
        def invoke(self, messages):
            # confirm prompts
            assert any(isinstance(m, SystemMessage) for m in messages)
            assert any(isinstance(m, HumanMessage) for m in messages)
            return sample_output_data

    # Monkeypatch on the class to bypass pydantic instance restrictions
    monkeypatch.setattr(
        ChatGoogleGenerativeAI,
        "with_structured_output",
        lambda self, model: DummyStructuredLLM(),
        raising=False,
    )

    out = calculate_node(state)
    assert "processed_data" in out
    assert out["processed_data"] == sample_output_data


def test_recommendation_chat_bot_invokes_llm(monkeypatch, sample_output_data):
    state = {"processed_data": sample_output_data}

    class DummyChatResult:
        content = "Optimize CAC by focusing on high-LTV channels"

    # Patch invoke on the class
    monkeypatch.setattr(
        ChatGoogleGenerativeAI,
        "invoke",
        lambda self, messages: DummyChatResult(),
        raising=False,
    )

    out = recommendation_chat_bot(state)
    assert "recommendation" in out
    assert out["recommendation"].content.startswith("Optimize CAC")


def test_wrapper_agent_full_flow(monkeypatch, sample_input_data, sample_output_data):
    # Patch structured output on the class
    class DummyStructured:
        def invoke(self, msgs):
            return sample_output_data

    monkeypatch.setattr(
        ChatGoogleGenerativeAI,
        "with_structured_output",
        lambda self, model: DummyStructured(),
        raising=False,
    )

    # Patch recommendation invoke on the class
    class DummyRec:
        content = "All metrics look healthy"

    monkeypatch.setattr(
        ChatGoogleGenerativeAI,
        "invoke",
        lambda self, msgs: DummyRec(),
        raising=False,
    )

    # Prepare input
    input_model = InputSalesData(**sample_input_data)
    # Call the Pregel entrypoint with a named kwarg
    result = wrapper_agent.invoke(input_model.model_dump())

    assert result["processed_data"] == sample_output_data
    assert result["recommendation"] == "All metrics look healthy"
