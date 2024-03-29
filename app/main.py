import logging
import math
import sys
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import List, Optional

import requests
from ai_traineree.agents.agent_factory import AgentFactory
from ai_traineree.types import AgentState, Experience
from ai_traineree.types.primitive import ObservationType
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseSettings, root_validator

from .types import AgentAction, AgentCreate, AgentInfo, AgentLoss, AgentStateJSON, AgentStep, DataSpace
from .utils import dataspace_fix, decode_pickle, encode_pickle

# Initiate module with setting up a server
app = FastAPI(
    title="Agents Bar - Agent",
    description="Agents Bar compatible Agent entity",
    docs_url="/docs",
    version="0.1.2",
)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger()

SUPPORTED_AGENTS = ("DQN", "PPO", "DDPG", "SAC", "D3PG", "D4PG", "RAINBOW", "TD3")


class AgentSettings(BaseSettings):
    AGENT_ID: Optional[int]
    TOKEN: Optional[str]
    URL: Optional[str] = "http://backend/api/v1/snapshots/auto"
    STANDALONE: bool = True

    @root_validator
    def check_not_standalone(cls, values):
        standalone = values.get("STANDALONE")
        if not standalone:
            assert all([v is not None for v in values.values()]), f"No value should be None. {values.items()}"
        return values


agent = None
last_step = {}
last_active = datetime.utcnow()
last_metrics_time = datetime.utcnow()
metrics_buffer = deque(maxlen=20)


def global_agent():
    "Global agent handling. It's mainly a helper function to handle exception."
    global agent
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent found")
    return agent


@app.get("/ping")
def ping():
    return {"msg": "All good"}


@app.post("/agent", status_code=201)
def api_post_agent(agent_create: AgentCreate):
    """Create agent.

    The agent is reused by other methods. There is no "update" method to agent's interal state
    so in case it needs changes it should be deleted and recreated.

    """
    global agent

    if agent_create.model_type.upper() not in SUPPORTED_AGENTS:
        raise HTTPException(status_code=400, detail=f"Only {SUPPORTED_AGENTS} agent types are supported")

    config = agent_create.model_config or {}
    agent_create.obs_space.shape = tuple(agent_create.obs_space.shape)
    agent_create.action_space.shape = tuple(agent_create.action_space.shape)
    config["obs_space"] = agent_create.obs_space
    config["action_space"] = agent_create.action_space

    network_state = agent_create.network_state
    buffer_state = agent_create.buffer_state

    agent_state = AgentState(
        model=agent_create.model_type,
        obs_space=agent_create.obs_space,
        action_space=agent_create.action_space,
        config=config,
        network=network_state,
        buffer=buffer_state,
    )
    try:
        agent = AgentFactory.from_state(agent_state)
    except:
        raise HTTPException(
            status_code=400,
            detail="It's not clear how you got here. Well done. But that's incorrect. Please select supported agent.",
        )

    return {"response": "Successfully created a new agent"}


@app.delete("/agent/{agent_name}", status_code=204)
def api_delete_agent(agent_name: str, agent=Depends(global_agent)):
    """Deletes agent and all related attribute. A hard reset.

    Agent name (reference) is necessary to prevent accidental deletion.
    """
    if agent.name == agent_name:
        agent = None
        return {"response": "Deleted successfully."}

    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")


@app.get("/agent", response_model=AgentInfo)
def api_get_agent_info(agent=Depends(global_agent)):
    """Describes agent.

    Provides summary information of the agent.
    The method should be relatively light.
    """
    discret = agent.model.upper() in ("DQN", "RAINBOW")
    return AgentInfo(
        model=agent.model,
        hyperparameters=agent.hparams,
        last_active=last_active,
        discret=discret,
    )


@app.get("/agent/state", response_model=AgentStateJSON)
def api_get_agent_state(agent=Depends(global_agent)):
    """Retruns agent's state.

    The state should be sufficient to fully describe and reconstruct the agent.
    It might be that in some situations the reconstructed agent doesn't produce
    the exact same output, e.g. due to internal randomness, but statistically
    they need to be the same.

    """
    agent_state = agent.get_state()
    agent_config = agent_state.config
    agent_config["device"] = str(agent_config.get("device", "cpu"))

    if "obs_space" in agent_config:
        del agent_config["obs_space"]
    if "action_space" in agent_config:
        del agent_config["action_space"]

    return AgentStateJSON(
        model=agent_state.model,
        obs_space=asdict(agent_state.obs_space),
        action_space=asdict(agent_state.action_space),
        encoded_config=encode_pickle(agent_config),
        encoded_network=encode_pickle(agent_state.network),
        encoded_buffer=encode_pickle(agent_state.buffer),
    )


@app.get("/agent/last_active", response_model=datetime)
def api_get_agent_last_active(agent=Depends(global_agent)):
    """Returns timestamp of agent's latest usage."""
    return last_active


@app.get("/agent/hparams")
def api_get_agent_hparasm(agent=Depends(global_agent)):
    """Returns hashmap of agent's hyperparameters."""
    print(agent.hparams)
    return {"hyperparameters": agent.hparams}


@app.get("/agent/loss", response_model=List[AgentLoss])
def api_get_agent_loss(last_samples: int = 1, agent=Depends(global_agent)):
    """Returns agent's loss values.

    By default it only returns the most recent metrics, i.e. single timestamp.
    Max timestamp values is based on the agent's intialization.
    """
    if len(metrics_buffer) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough collected samples. Current count is {len(metrics_buffer)}.",
        )
    beg_idx = max(0, len(metrics_buffer) - last_samples)
    return [metrics_buffer[i] for i in range(beg_idx, len(metrics_buffer))]


@app.post("/agent/step", status_code=200)
def api_post_agent_step(step: AgentStep, commit: bool = True, agent=Depends(global_agent)):
    """Feed agent with step information.

    The minimum required is the current state and reward.
    Some agents, for convinience, might also require passing last action,
    auxilary infrmation whether state is terminal (done) and the next state.

    By default, the Step is committed to the agent in the request.
    In case it's needed to delay committing, e.g. gathering all information first,
    one can use `/agent/commit` method.

    """
    # TODO: Agent should have a property whether it's discrete
    global last_active, last_step
    last_active = datetime.utcnow()
    if agent.model in ("DQN", "Rainbow"):
        action = int(step.action[0])
    else:
        action = step.action

    last_step = dict(
        last_step_obs=step.obs,
        last_action=action,
        last_step_reward=step.reward,
        last_step_next_obs=step.next_obs,
        last_step_done=step.done,
    )
    if commit:
        agent_commit(agent)
        return {"response": "Stepping"}

    return {"response": "Submitted"}


@app.post("/agent/commit", status_code=200)
def api_post_agent_commit(agent=Depends(global_agent)):
    """Commits submitted step into Agent.

    Before using this method the data needs to be submitted using `/agent/step`.
    """
    global last_active, last_step
    last_active = datetime.utcnow()
    agent_commit(agent)


@app.post("/agent/reset", status_code=200)
def api_post_agent_reset(agent=Depends(global_agent)):
    """Resets Agent.

    Clears Agents states.
    """
    agent.reset()


@app.post("/agent/act", response_model=AgentAction)
def api_post_agent_act(state: ObservationType, noise: float = 0.0, agent=Depends(global_agent)):
    """Infers action based on provided observation."""
    global last_active
    last_active = datetime.utcnow()
    try:
        exp = Experience(obs=state)
        exp = agent.act(exp, noise)
        action = exp.action
    except Exception as e:
        logger.exception("Failed to exceute `agent.act` with state=%s and noise=%s", str(state), str(noise))
        raise HTTPException(status_code=500, detail=f"Sorry :(\n{e}")

    collect_metrics()
    return AgentAction(action=action)


def agent_commit(agent):
    global last_step
    assert agent is not None, "Agent needs to be initialized"

    agent.step(
        Experience(
            obs=last_step["last_step_obs"],
            action=last_step["last_action"],
            reward=last_step["last_step_reward"],
            next_obs=last_step["last_step_next_obs"],
            done=last_step["last_step_done"],
        )
    )
    last_step = {}  # Empty once used

    collect_metrics()


def collect_metrics(wait_seconds: int = 20):
    global last_metrics_time
    if agent is None:
        raise ValueError("Agent needs to be initiated before it can be used")

    now_time = datetime.utcnow()
    if now_time < last_metrics_time + timedelta(seconds=wait_seconds):
        return

    loss = {k: v if not (math.isinf(v) or math.isnan(v)) else None for (k, v) in agent.loss.items()}
    loss["time"] = now_time.timestamp()
    metrics_buffer.append(loss)
    last_metrics_time = now_time


def sync_agent_state(agent_id: int, token: str) -> AgentState:
    logging.info("Synchronizing agent with the backend")
    url = f"{AgentSettings().URL}/{agent_id}"
    response = requests.get(url, headers={"token": token})
    data = response.json()
    agent_type = data["model"].upper()

    obs_space = dataspace_fix(data.pop("obs_space"))
    action_space = dataspace_fix(data.pop("action_space"))

    agent_config = decode_pickle(data["encoded_config"])
    network_state = decode_pickle(data["encoded_network"])
    buffer_state = decode_pickle(data["encoded_buffer"])

    if "obs_space" in agent_config:
        agent_config.pop("obs_space")
    if "action_space" in agent_config:
        agent_config.pop("action_space")

    return AgentState(
        model=agent_type,
        obs_space=DataSpace(**obs_space),
        action_space=DataSpace(**action_space),
        config=agent_config,
        network=network_state,
        buffer=buffer_state,
    )


##############################
# MAIN

config = AgentSettings()

if not config.STANDALONE:
    if config.AGENT_ID is None or config.TOKEN is None:
        raise ValueError("")
    state = sync_agent_state(config.AGENT_ID, config.TOKEN)
    agent = AgentFactory.from_state(state=state)
    print("Initiated agent: " + str(agent))
