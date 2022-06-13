USER_ID ?= ${shell id -u}
USER_NAME ?= ${shell whoami}

PROJECT ?= panodepth-vidar
WORKSPACE ?= /home/${USER_NAME}/workspace/${PROJECT}
DOCKER_IMAGE ?= ${PROJECT}:${USER_NAME}
CUSTOM_HDD_PATH ?= /media/soonminh/HDD8TB/

SHMSIZE ?= 444G
WANDB_MODE ?= run
DOCKER_OPTS := \
			--rm -it \
			--shm-size=${SHMSIZE} \
			-e AWS_DEFAULT_REGION \
			-e AWS_ACCESS_KEY_ID \
			-e AWS_SECRET_ACCESS_KEY \
			-e WANDB_API_KEY \
			-e WANDB_ENTITY \
			-e WANDB_MODE \
			-e HOST_HOSTNAME= \
			-e OMP_NUM_THREADS=1 -e KMP_AFFINITY="granularity=fine,compact,1,0" \
			-e OMPI_ALLOW_RUN_AS_ROOT=1 \
			-e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
			-e NCCL_DEBUG=VERSION \
            -e DISPLAY=${DISPLAY} \
            -e XAUTHORITY \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
			-v ~/.aws:/root/.aws \
			-v /root/.ssh:/root/.ssh \
			-v ~/.cache:/root/.cache \
			-v /data:/data \
			-v /dev/null:/dev/raw1394 \
			-v /mnt/fsx/tmp:/tmp \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v /run/dbus/system_bus_socket:/run/dbus/system_bus_socket:ro \
			-v ${CUSTOM_HDD_PATH}:${CUSTOM_HDD_PATH} \
			-v ${WORKSPACE}:${WORKSPACE} \
			-w ${WORKSPACE} \
			--privileged \
			--ipc=host \
			--network=host

NGPUS=$(shell nvidia-smi -L | wc -l)

all: clean

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

docker-build:
	docker build \
		--build-arg TZ=America/New_York \
		--build-arg USER_ID=${USER_ID} \
		--build-arg USER_NAME=${USER_NAME} \
		-f docker/Dockerfile \
		-t ${DOCKER_IMAGE} .

docker-interactive: docker-build
	nvidia-docker run ${DOCKER_OPTS} --name ${PROJECT}_interactive ${DOCKER_IMAGE} /bin/bash

docker-jupyter:
	nvidia-docker run ${DOCKER_OPTS} --name ${PROJECT}_jupyter ${DOCKER_IMAGE} \
		bash -c "jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --no-browser --notebook-dir=${WORKSPACE}"

docker-run: docker-build
	nvidia-docker run ${DOCKER_OPTS} --name ${PROJECT} ${DOCKER_IMAGE} bash -c "${COMMAND}"
