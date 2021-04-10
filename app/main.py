import sys
from base64 import decode
import os
from typing import Dict, List, Optional

import logging
import requests
from ai_traineree.agents.agent_factory import AgentFactory
from ai_traineree.types import AgentState
from ai_traineree.utils import serialize
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .utils import decode_pickle, encode_pickle


# Initiate module with setting up a server
app = FastAPI()

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

standalone = os.environ.get('STANDALONE')
if standalone is not None:
    standalone = not (standalone.lower() == 'false' or standalone == '0')
else:
    standalone = True


# NOTE: Sometimes there's something wrong with these. Pay attention.
print(f"OS env: {standalone}")
print(f"OS env: {os.environ}")
print("Updated")

if not standalone:
    agent_id = int(os.environ.get('AGENT_ID'))
    token = os.environ.get('TOKEN')
    url_base = os.environ.get("URL", "http://backend/api/v1/snap")
    url = f'{url_base}/{agent_id}'
    print(f"GET {url}")
    response = requests.get(url, headers={"token": token})
    print(response.text)
    data = response.json()

    agent_type = data['model'].upper()
    state_size = int(data.pop('state_size'))
    action_size = int(data.pop('action_size'))
    agent_config = decode_pickle(data['encoded_config'])
    network_state = decode_pickle(data['encoded_network'])
    buffer_state = decode_pickle(data['encoded_buffer'])

    agent_state = AgentState(
        model=agent_type,
        state_space=state_size,
        action_space=action_size,
        config=agent_config,
        network=network_state,
        buffer=buffer_state,
    )

    # Leave things to get magic done in factory
    agent = AgentFactory.from_state(state=agent_state)


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


SUPPORTED_AGENTS = ['dqn', 'ppo', 'ddpg', 'sac', 'd3pg', 'd4pg', 'rainbow', 'td3']


@app.get("/ping")
def ping():
    return {"msg": "All good"}


@app.post("/agent", status_code=201)
def create_agent(
    model_type: str,
    state_size: int,
    action_size: int,
    model_config: Optional[Dict[str, str]],
    network_state: Optional[Dict[str, str]],
    buffer_state: Optional[Dict[str, str]]
):
    global agent
    # if agent is not None:
    #     raise HTTPException(status_code=400, detail="Agent already exists. If you want to create a new one, please first remove old one.")

    if model_type.lower() not in SUPPORTED_AGENTS:
        raise HTTPException(status_code=400, detail=f"Only {SUPPORTED_AGENTS} agent types are supported")

    network_state = None
    buffer_state = None
    agent_state = AgentState(model=model_type.upper(), state_space=state_size, action_space=action_size, config=model_config, network=network_state, buffer=buffer_state)
    agent = AgentFactory.from_state(agent_state)
    if agent is None:
        raise HTTPException(
            status_code=400,
            detail="It's not clear how you got here. Well done. But that's incorrect. Please select supported agent.")
    
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
    

@app.get("/agent/state", response_model=AgentStateJSON)
def get_agent_state():
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    agent_state = agent.get_state()
    agent_config = agent_state.config
    print(agent_config)
    logging.info(str(agent_config))
    for (k, v) in agent_config.items():
        if k.lower() == 'device':
            agent_config[k] = str(v)
        logging.info(f"{k=}  |  {v=}  |  {type(v)}")

    out = AgentStateJSON(
        model=agent_state.model, state_space=agent_state.state_space, action_space=agent_state.action_space,
        encoded_config=encode_pickle(agent_config),
        encoded_network=encode_pickle(agent_state.network),
        encoded_buffer=encode_pickle(agent_state.buffer)
    )
    return out


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
