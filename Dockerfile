FROM tensorflow/tensorflow:2.3.2-gpu

# dont write pyc files
# dont buffer to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TZ=Europe/Kiev

COPY requirements.txt /

# system preparation
RUN DEBIAN_FRONTEND=noninteractive; \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-opencv \
        libcairo2-dev \
    && pip install --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# pip installs
RUN pip install -r /requirements.txt \
    && pip install mavis_core \
    && rm -rf /root/.cache/pip /tmp/* /var/tmp/*


CMD mavis