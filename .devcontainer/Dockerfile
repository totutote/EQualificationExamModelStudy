FROM python:3.11.13
LABEL purpose="devcontainer"

RUN pip install --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 pytorch-lightning \
  grad-cam==1.5.4 matplotlib==3.9.3 numpy==2.0.2 opencv-python==4.10.0.84 \
  torchmetrics==0.6.0 torchinfo==1.8.0 opencv-python==4.10.0.84 pyautogui \
  gymnasium[other]==1.0.0 gymnasium[classic-control]==1.0.0

RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        sudo \
        git \
        curl \
        gnupg2 \
        lsb-release \
        build-essential \
        python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -SL https://github.com/docker/compose/releases/download/v2.4.1/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose

RUN chmod +x /usr/local/bin/docker-compose

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && usermod -aG sudo $USERNAME \
    && echo $USERNAME ALL=\(NOPASSWD\) ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

WORKDIR /workspace