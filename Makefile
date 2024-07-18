# Define user-specific variables
USER_ID ?= $(shell id -u)
GROUP_ID ?= $(shell id -g)
USER_NAME ?= $(shell whoami)

# Define project-specific variables
PROJECT ?= panosupervised
WORKSPACE ?= /home/${USER_NAME}/workspace/${PROJECT}
DOCKER_IMAGE ?= ${PROJECT}:${USER_NAME}_latest

# Define Docker options
SHMSIZE ?= 444G
WANDB_MODE ?= run
DOCKER_OPTS := \
    --rm -it \
    --gpus all \
    --shm-size=${SHMSIZE} \
    -e AWS_DEFAULT_REGION \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e WANDB_API_KEY \
    -e WANDB_ENTITY \
    -e WANDB_MODE \
    -e HOST_HOSTNAME= \
    -e OMP_NUM_THREADS=1 \
    -e KMP_AFFINITY="granularity=fine,compact,1,0" \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 \
    -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    -e NCCL_DEBUG=VERSION \
    -e DISPLAY=${DISPLAY} \
    -e XAUTHORITY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v ~/.aws:/home/${USER_NAME}/.aws \
    -v ~/.ssh:/home/${USER_NAME}/.ssh \
    -v ~/.cache:/home/${USER_NAME}/.cache \
    -v /data:/data \
    -v /dev/null:/dev/raw1394 \
    -v /mnt/fsx/tmp:/tmp \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /run/dbus/system_bus_socket:/run/dbus/system_bus_socket:ro \
    -v ${WORKSPACE}/../dgp:${WORKSPACE}/../dgp \
    -v ${WORKSPACE}:${WORKSPACE} \
    -w ${WORKSPACE} \
    --privileged \
    --ipc=host \
    --network=host

NGPUS=$(shell nvidia-smi -L | wc -l)

all: clean

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -exec rm -rf {} +

docker-build:
	docker build \
		--build-arg TZ=Asia/Seoul \
		--build-arg USER_ID=2000 \
		--build-arg GROUP_ID=2000 \
		--build-arg USER_NAME=${USER_NAME} \
		-f docker/Dockerfile_latest \
		-t ${DOCKER_IMAGE} .

docker-interactive: docker-build
	docker run ${DOCKER_OPTS} --name ${PROJECT}_interactive2 ${DOCKER_IMAGE} /bin/bash

docker-jupyter: docker-build
	docker run ${DOCKER_OPTS} --name ${PROJECT}_jupyter ${DOCKER_IMAGE} \
		bash -c "jupyter lab --port=8989 --ip=0.0.0.0 --allow-root --no-browser --notebook-dir=${WORKSPACE}"

docker-run: docker-build
	docker run ${DOCKER_OPTS} --name ${PROJECT} ${DOCKER_IMAGE} bash -c "${COMMAND}"
