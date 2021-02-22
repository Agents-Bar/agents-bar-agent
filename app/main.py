import json

from ai_traineree.types import AgentType
from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.agents.ddpg import DDPGAgent
from ai_traineree.agents.ppo import PPOAgent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

app = FastAPI()

agent: Optional[AgentType] = None


class AgentStep(BaseModel):
    state: List[float]
    action: List[float]
    next_state: List[float]
    reward: float
    done: bool


SUPPORTED_AGENTS = ['dqn', 'ppo', 'ddpg']


@app.post("/agent/create")
def create_agent(model_type: str, state_size: int, action_size: int, model_config: Optional[Dict[str, str]]):
    global agent
    if agent is not None:
        raise HTTPException(status_code=400, detail="Agent already exists. If you want to create a new one, please first remove old one.")

    if model_type.lower() not in SUPPORTED_AGENTS:
        raise HTTPException(status_code=400, detail=f"Only {SUPPORTED_AGENTS} agent types are supported")

    if model_type.lower() == "dqn":
        agent = DQNAgent(input_shape=state_size, output_shape=action_size, **model_config)
    elif model_type.lower() == "ppo":
        agent = PPOAgent(state_size=state_size, action_size=action_size, **model_config)
    elif model_type.lower() == 'ddpg':
        agent = DDPGAgent(state_size=state_size, action_size=action_size, **model_config)
    
    print(f"Agent: {agent}")
    return {"status": 201, "response": "Successfully created a new agent"}


@app.delete("/agent/delete")
def delete_agent(agent_name: str):
    """Assumption is that the service contains only one agent. Otherwise... something else."""
    global agent
    if agent is None:
        return {"status": 204, "response": "Agent doesn't exist."}

    if agent.name == agent_name:
        agent = None
        return {"status": 204, "response": "Deleted successfully."}

    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    

@app.get("/agent/info")
def get_agent_info():
    print(agent.hparams)
    return {"status": 200, "response": json.dumps(agent.hparams)}

@app.get("/agent/loss")
def get_agent_loss():
    print(agent.loss)
    return {"status": 200, "response": json.dumps(agent.loss)}

@app.post("/agent/step")
def agent_step(agent_step: AgentStep):
    if isinstance(agent, DQNAgent):
        action = agent_step.action[0]
    else:
        action = agent_step.action

    agent.step(
        agent_step.state,
        action,
        agent_step.reward,
        agent_step.next_state,
        agent_step.done
    )
    return {"status": 200, "response": "Stepping"}

@app.post("/agent/act")
def agent_act(state: List[float]):
    try:
        action = agent.act(state)
        return {"status": 200, "response": {"action": action}}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sorry :(\n{e}")
