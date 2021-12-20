# Agent - Agents Bar

The core of Agent's interation.
Implements [Reinforcement Learning APIs](https://github.com/Agents-Bar/rl-api-definitions) to enable interaction with [AI Traineree](https://github.com/laszukdawid/ai-traineree).

The Agent is provided as a Docker image.
Please see GitHub's [Agents Bar - Agent container repository](https://github.com/Agents-Bar/agents-bar-agent/pkgs/container/agents-bar-agent) to access the image.

## Development and Production images

### Development

In order to build the development image, including the full Pytoch dependency, run
```
docker build --target dev . -t agent_service:dev_version
```

### Production

In order to build the production image, including the Pytoch+cpu dependency (no support for CUDA), run
```
docker build --target prod . -t agent_service:latest
```
