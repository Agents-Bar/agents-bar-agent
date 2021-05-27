import logging
import math
import os
import sys
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from ai_traineree.agents.agent_factory import AgentFactory
from ai_traineree.types import AgentState
from fastapi import FastAPI, HTTPException

from .types import AgentInfo, AgentLoss, AgentStateJSON, AgentStep
from .utils import decode_pickle, encode_pickle

# Initiate module with setting up a server
app = FastAPI()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

SUPPORTED_AGENTS = ['DQN', 'PPO', 'DDPG', 'SAC', 'D3PG', 'D4PG', 'RAINBOW', 'TD3']

last_active = datetime.utcnow()
previous_active = datetime.utcnow()
metrics_buffer = deque(maxlen=20)


def sync_agent_state(agent_id: int, token: str) -> AgentState:
    url_base = os.environ.get("URL", "http://backend/api/v1/snapshots/auto")
    url = f'{url_base}/{agent_id}'
    print(f"GET {url}")
    response = requests.get(url, headers={"token": token})
    data = response.json()
    agent_type = data['model'].upper()
    state_size = int(data.pop('state_size'))
    action_size = int(data.pop('action_size'))
    agent_config = decode_pickle(data['encoded_config'])
    network_state = decode_pickle(data['encoded_network'])
    buffer_state = decode_pickle(data['encoded_buffer'])

    return AgentState(
        model=agent_type,
        state_space=state_size,
        action_space=action_size,
        config=agent_config,
        network=network_state,
        buffer=buffer_state,
    )


@app.get("/ping")
def ping():
    return {"msg": "All good"}


@app.post("/agent", status_code=201)
def create_agent(
    model_type: str,
    state_size: int,
    action_size: int,
    model_config: Optional[Dict[str, str]],
    network_state: Optional[bytes] = None,
    buffer_state: Optional[bytes] = None,
):
    global agent
    # if agent is not None:
    #     raise HTTPException(status_code=400, detail="Agent already exists. If you want to create a new one, please first remove old one.")

    if model_type.upper() not in SUPPORTED_AGENTS:
        raise HTTPException(status_code=400, detail=f"Only {SUPPORTED_AGENTS} agent types are supported")

    agent_state = AgentState(
        model=model_type, state_space=state_size, action_space=action_size,
        config=model_config, network=network_state, buffer=buffer_state
    )
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


@app.get("/agent/", response_model=AgentInfo)
def get_agent_info():
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    return AgentInfo(model=agent.name, hyperparameters=agent.hparams, last_active=last_active)


@app.get("/agent/hparams")
def get_agent_info():
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    print(agent.hparams)
    return {"hyperparameters": agent.hparams}


@app.get("/agent/loss", response_model=List[AgentLoss])
def get_agent_loss(last_samples: int = 1):
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    if len(metrics_buffer) == 0:
        raise HTTPException(status_code=400, detail=f"Not enough collected samples. Current count is {len(metrics_buffer)}.")
    beg_idx = max(0, len(metrics_buffer) - last_samples)
    return [metrics_buffer[i] for i in range(beg_idx, len(metrics_buffer))]


@app.post("/agent/step", status_code=200)
def agent_step(step: AgentStep):
    # TODO: Agent should have a property whether it's discrete
    global last_active
    last_active = datetime.utcnow()
    if agent.name in ('DQN', 'Rainbow'):
        action = int(step.action[0])
    else:
        action = step.action

    agent.step(
        step.state,
        action,
        step.reward,
        step.next_state,
        step.done
    )

    collect_metrics()

    return {"response": "Stepping"}


@app.post("/agent/act")
def agent_act(state: List[float], noise: float=0.):
    global last_active
    last_active = datetime.utcnow()
    try:
        action = agent.act(state, noise)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sorry :(\n{e}")

    collect_metrics()
    return {"action": action}
    

def collect_metrics(wait_seconds=20):
    global previous_active
    now_time = datetime.utcnow()
    if now_time < previous_active + timedelta(seconds=wait_seconds):
        return

    loss = {k: v if not math.isinf(v) else None for (k, v) in agent.loss.items()}
    loss['time'] = now_time.timestamp()
    metrics_buffer.append(loss)
    previous_active = now_time


##############################
# MAIN

# By default should be Standalone, except when explicitly mentioned by ENV var
standalone = os.environ.get('STANDALONE', '1')
standalone = not (standalone.lower() == 'false' or standalone == '0')

# NOTE: Sometimes there's something wrong with these. Pay attention.
print(f"OS env: {standalone}")
print(f"OS env: {os.environ}")

if not standalone:
    agent_id = int(os.environ.get('AGENT_ID'))
    token = os.environ.get('TOKEN')
    state = sync_agent_state(agent_id, token)
    # Leave things to get magic done in factory
    agent = AgentFactory.from_state(state=state)
