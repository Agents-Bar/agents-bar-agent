import os

from ai_traineree.types import AgentType
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

def define_agent(model_type: str, state_size, action_size, model_config):
    if model_type.lower() == "dqn":
        from ai_traineree.agents.dqn import DQNAgent
        agent = DQNAgent(input_shape=state_size, output_shape=action_size, **model_config)
    elif model_type.lower() == "rainbow":
        from ai_traineree.agents.rainbow import RainbowAgent
        agent = RainbowAgent(input_shape=state_size, output_shape=action_size, **model_config)
    elif model_type.lower() == "ppo":
        from ai_traineree.agents.ppo import PPOAgent
        agent = PPOAgent(state_size=state_size, action_size=action_size, **model_config)
    elif model_type.lower() == 'ddpg':
        from ai_traineree.agents.ddpg import DDPGAgent
        agent = DDPGAgent(state_size=state_size, action_size=action_size, **model_config)
    elif model_type.lower() == 'd3pg':
        from ai_traineree.agents.d3pg import D3PGAgent
        agent = D3PGAgent(state_size=state_size, action_size=action_size, **model_config)
    elif model_type.lower() == 'd4pg':
        from ai_traineree.agents.d4pg import D4PGAgent
        agent = D4PGAgent(state_size=state_size, action_size=action_size, **model_config)
    elif model_type.lower() == 'sac':
        from ai_traineree.agents.sac import SACAgent
        agent = SACAgent(state_size=state_size, action_size=action_size, **model_config)
    else:
        agent = None
    return agent


app = FastAPI()

agent_type: Optional[str] = os.environ.get('AGENT_TYPE', 'DQN')
state_size = os.environ.get("state_size", 27)
action_size = os.environ.get("action_size", 256)
model_config = {}
agent: Optional[AgentType] = define_agent(agent_type, state_size, action_size, model_config=model_config)


class AgentStep(BaseModel):
    state: List[float]
    action: List[float]
    next_state: List[float]
    reward: float
    done: bool


SUPPORTED_AGENTS = ['dqn', 'ppo', 'ddpg', 'sac', 'd3pg', 'd4pg', 'rainbow', 'td3']

@app.get("/ping")
def ping():
    return {"msg": "All good"}

@app.post("/agent", status_code=201)
def create_agent(model_type: str, state_size: int, action_size: int, model_config: Optional[Dict[str, str]]):
    global agent
    # if agent is not None:
    #     raise HTTPException(status_code=400, detail="Agent already exists. If you want to create a new one, please first remove old one.")

    if model_type.lower() not in SUPPORTED_AGENTS:
        raise HTTPException(status_code=400, detail=f"Only {SUPPORTED_AGENTS} agent types are supported")

    agent = define_agent(model_type, state_size, action_size, model_config)
    if agent is None:
        raise HTTPException(400, detail="It's not clear how you got here. Well done. But that's incorrect. Please select supported agent.")
    
    print(f"Agent: {agent}")
    return {"response": "Successfully created a new agent"}


@app.delete("/agent/{agent_name}", status_code=204)
def delete_agent(agent_name: str):
    """Assumption is that the service contains only one agent. Otherwise... something else."""
    global agent
    if agent is None:
        return {"response": "Agent doesn't exist."}

    if agent.name == agent_name:
        agent = None
        return {"response": "Deleted successfully."}

    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    

@app.get("/agent/")
def get_agent_info():
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    print(agent)
    agent_info = {"agent_type": str(agent), "hyperparameters": agent.hparams}
    return agent_info


@app.get("/agent/hparams")
def get_agent_info():
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    print(agent.hparams)
    return {"hyperparameters": agent.hparams}


@app.get("/agent/loss")
def get_agent_loss():
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    print(agent.loss)
    return agent.loss


@app.post("/agent/step", status_code=200)
def agent_step(agent_step: AgentStep):
    global agent
    # TODO: Agent should have a property whether it's discrete
    if agent.name in ('DQN', 'Rainbow'):
        action = int(agent_step.action[0])
    else:
        action = agent_step.action

    agent.step(
        agent_step.state,
        action,
        agent_step.reward,
        agent_step.next_state,
        agent_step.done
    )
    return {"response": "Stepping"}


@app.post("/agent/act")
def agent_act(state: List[float], noise: float=0.):
    global agent
    try:
        action = agent.act(state, noise)
        return {"action": action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sorry :(\n{e}")
