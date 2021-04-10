from typing import Dict, List, Optional

from pydantic import BaseModel


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


