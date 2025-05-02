#!/bin/bash

# Run the Docker container with GPU support
# Usage: ./docker-run.sh [command]
# If no command is provided, it will open a bash shell

if [ "$#" -eq 0 ]; then
    # Run container with bash if no command provided
    docker run --gpus all -it --rm \
        -v $(pwd):/app \
        --env="NVIDIA_VISIBLE_DEVICES=all" \
        --env="NVIDIA_DRIVER_CAPABILITIES=compute,utility" \
        iseebetter:latest
else
    # Run the specified command
    docker run --gpus all -it --rm \
        -v $(pwd):/app \
        --env="NVIDIA_VISIBLE_DEVICES=all" \
        --env="NVIDIA_DRIVER_CAPABILITIES=compute,utility" \
        iseebetter:latest -c "$*"
fi 