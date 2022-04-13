FROM tensorflow/tensorflow:2.4.1-gpu

# dont write pyc files
# dont buffer to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TZ=Europe/Kiev

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
RUN pip install pycairo \
    && pip install mavis-core \
    && rm -rf /root/.cache/pip /tmp/* /var/tmp/*


CMD mavis