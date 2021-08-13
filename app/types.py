from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from ai_traineree.types.primitive import ObservationType
from pydantic import BaseModel, validator

ActionType = Union[int, float, List[int], List[float]]
AgentLoss = Dict[str, Optional[float]]


class DataSpace(BaseModel):
    # TODO: This is a copy/paste from ai_traineree.types.dataspace.Dataspace
    #       The reason why it isn't used directly is that the other type ^
    #       uses Dataclass which requires a bit more formatting for OpenAPI
    #       than BaseModel. Keeping them in sync will be a problem so... yeah.
    dtype: str
    shape: Sequence[int]
    low: Optional[Union[float, List[float]]] = None
    high: Optional[Union[float, List[float]]] = None

    @validator('dtype')
    def validate_dtype(cls, v):
        assert v.startswith('int') or v.startswith('float'), "Only int* or float* formats supported"
        return v
    
    def to_feature(self):
        if self.dtype.startswith('int') and len(self.shape) == 1:
            return (int(self.high - self.low + 1), )
        return self.shape


class AgentAction(BaseModel):
    action: ActionType


class AgentStep(BaseModel):
    obs: ObservationType
    action: List[float]
    next_obs: ObservationType
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
    obs_space: DataSpace
    action_space: DataSpace
    model_config: Optional[Dict[str, Any]]
    network_state: Optional[bytes] = None
    buffer_state: Optional[bytes] = None

