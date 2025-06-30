from typing import Optional

from pydantic import BaseModel


class PreviousDayData(BaseModel):
    sales: int
    costs: int
    customers_acquired: int


class InputSalesData(BaseModel):
    date: Optional[str] = None
    sales: int
    costs: int
    customers_acquired: int
    previous_day: Optional[PreviousDayData] = None


class OutputData(BaseModel):
    daily_profit: float
    revenue_change_percentage: float
    cost_change_percentage: float
    today_CAC: float
    CAC_change_percentage: float
    CAC_alert: bool
