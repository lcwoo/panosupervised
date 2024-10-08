FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV TZ=Asia/Seoul
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update && apt-get install -y python3-opencv

RUN pip install \
    opencv-python==4.9.0.80
