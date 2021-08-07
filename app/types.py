from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

ActionType = Union[int, float, List[int], List[float]]
AgentLoss = Dict[str, Optional[float]]


class AgentAction(BaseModel):
    action: ActionType


class AgentStep(BaseModel):
    obs: List[float]
    action: List[float]
    next_obs: List[float]
    reward: float
    done: bool


class AgentStateJSON(BaseModel):
    model: str
    state_space: Dict[str, Any]
    action_space: Dict[str, Any]
    encoded_config: str
    encoded_network: str
    encoded_buffer: str


class AgentInfo(BaseModel):
    model: str
    hyperparameters: Dict[str, Any]
    last_active: datetime
    discret: bool


class AgentCreate(BaseModel):
    model_type: str
    obs_space: Dict[str, Any]
    action_space: Dict[str, Any]
    model_config: Optional[Dict[str, Any]]
    network_state: Optional[bytes] = None
    buffer_state: Optional[bytes] = None
