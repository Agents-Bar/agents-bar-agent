# ai-traineree-service

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
