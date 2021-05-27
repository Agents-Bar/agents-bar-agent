from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

ActionType = Union[int, float, List[int], List[float]]


class AgentAction(BaseModel):
    action: ActionType


class AgentStep(BaseModel):
    state: List[float]
    action: List[float]
    next_state: List[float]
    reward: float
    done: bool


class AgentStateJSON(BaseModel):
    model: str
    state_space: int
    action_space: int
    encoded_config: str
    encoded_network: str
    encoded_buffer: str


class AgentInfo(BaseModel):
    model: str
    hyperparameters: Dict[str, Any]
    last_active: datetime


AgentLoss = Dict[str, Optional[float]]
