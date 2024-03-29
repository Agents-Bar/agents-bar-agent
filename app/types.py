from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ai_traineree.types import DataSpace, ObsType
from pydantic import BaseModel

ActionType = Union[int, float, List[int], List[float]]
AgentLoss = Dict[str, Optional[float]]


class AgentAction(BaseModel):
    action: ActionType


class AgentStep(BaseModel):
    obs: ObsType
    action: List[float]
    next_obs: ObsType
    reward: float
    done: bool


class AgentStateJSON(BaseModel):
    model: str
    obs_space: Dict[str, Any]
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
    obs_space: DataSpace
    action_space: DataSpace
    model_config: Optional[Dict[str, Any]]
    network_state: Optional[bytes] = None
    buffer_state: Optional[bytes] = None

    class Config:
        arbitrary_types_allowed = True
