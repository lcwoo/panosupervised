FROM nvidia/cuda:11.3.1-devel-ubuntu18.04

ENV PROJECT=panodepth-vidar
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHON_VERSION=3.8
ENV PYTORCH_VERSION=1.10.0+cu113
ENV TORCHVISION_VERSION=0.11.1+cu113
ENV CUDNN_VERSION=8.2.1.32-1+cuda11.3
ENV NCCL_VERSION=2.9.9-1+cuda11.3

# Update GPG keys for the CUDA repository
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

# Install basic libraries
RUN apt-get update && apt-get install -y \
    build-essential cmake g++-4.8 git curl docker.io vim wget ca-certificates

# Install python and pip
RUN apt-get install -y python${PYTHON_VERSION} python3-pip
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Install pytorch and torchvision
RUN pip install \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install CUDNN and NCCL
RUN apt-get install -y \
    libcudnn8=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION}

# Install extra packages (apt-get)
RUN apt-get install -y \
    ffmpeg \
    tmux

# Install extra packages (pip)
RUN pip install \
    tqdm==4.61.0 \
    boto3==1.17.83 \
    termcolor==1.1.0 \
    pyyaml==5.4.1 \
    wandb==0.10.31 \
    opencv-python==4.5.2.52 \
    flow_vis==0.1 \
    matplotlib==3.3.4 \
    fire==0.4.0 \
    pyquaternion==0.9.9 \
    pandas==1.1.5 \
    xarray==0.16.2 \
    diskcache==5.2.1 \
    tenacity==7.0.0 \
    pycocotools==2.0.2 \
    awscli==1.19.101 \
    timm==0.4.9 \
    ref==0.0.2.2 \
    positional-encodings==4.0.0 \
    einops==0.3.2 \
    wget \
    ftfy \
    regex \
    tqdm

# Install CamViz dependencies
RUN pip install \
    pygame==2.0.1 \
    PyOpenGL==3.1.5 \
    PyOpenGL-accelerate==3.1.5
RUN apt-get install -y \
    mesa-utils \
    freeglut3-dev \
    libsdl2-2.0-0 \
    python-pygame

# Install PyTorch3D
RUN pip install pytorch3d

# Install CuPY
RUN pip install cupy

# Install extras (should be moved to top when stable)
RUN pip install lpips wget scikit-image pyhocon dotmap path sacremoses filelock huggingface_hub
RUN pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# (Optional) Install packages
RUN pip install pythreejs ipywebrtc

# Set timezone
ARG TZ
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# (Optional) Add user
ARG USER_ID
ARG USER_NAME
RUN apt-get install -y sudo
RUN groupadd -r ${USER_NAME} -g ${USER_ID} && \
	useradd -u ${USER_ID} -g ${USER_NAME} -m -s /bin/bash ${USER_NAME} -p "$(openssl passwd -1 ${USER_NAME})" && \
    usermod -aG sudo ${USER_NAME}

WORKDIR /home/${USER_NAME}/workspace
RUN chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/workspace
USER ${USER_NAME}

# Install DGP (dataset utils)
WORKDIR /home/${USER_NAME}/workspace
RUN git clone https://github.com/VitorGuizilini-TRI/dgp.git
ENV PYTHONPATH="/home/${USER_NAME}/workspace/dgp:$PYTHONPATH"

################################################################################################
# Do not copy workspace, mounting workspace provides us more flexibility in the interactive mode
#   e.g. modify code through remote ssh
################################################################################################
# Create workspace folder
# RUN mkdir -p /home/${USER_NAME}/workspace/experiments
# RUN mkdir -p /home/${USER_NAME}/workspace/${PROJECT}
# WORKDIR /home/${USER_NAME}/workspace/${PROJECT}
# # Copy project to workspace folder
# COPY --chown=${USER_NAME}:${USER_NAME} . /home/${USER_NAME}/workspace/${PROJECT}

# Set environment variables
ENV PYTHONPATH="/home/${USER_NAME}/workspace/${PROJECT}:$PYTHONPATH"
ENV PYTHONPATH="/home/${USER_NAME}/workspace/${PROJECT}/externals/camviz:$PYTHONPATH"

# (Optional) Apply custom BASH Prompt to make it different from local prompt
RUN echo 'export PS1="(docker) [\e[0;32m(\D{%T})\e[0m \e[0;33m\u@\h:\w\e[0m]\$ "' >> /home/$USER_NAME/.bashrc

# Expose Port for jupyter (8888)
EXPOSE 8888
RUN pip install jupyter
