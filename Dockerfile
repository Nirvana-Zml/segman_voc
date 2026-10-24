FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Default to a China-reachable Ubuntu mirror; override with:
#   docker build --build-arg APT_MIRROR=archive.ubuntu.com ...
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
RUN sed -i \
      -e "s|http://archive.ubuntu.com/ubuntu|http://${APT_MIRROR}/ubuntu|g" \
      -e "s|http://security.ubuntu.com/ubuntu|http://${APT_MIRROR}/ubuntu|g" \
      /etc/apt/sources.list \
    && printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\n' \
      > /etc/apt/apt.conf.d/80-retries \
    && apt-get update \
    && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/voc_segmentation

COPY voc_segmentation/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import torch, torchvision; assert torch.__version__.startswith('2.1.2'); assert torchvision.__version__.startswith('0.16.2')"

COPY voc_segmentation/ /workspace/voc_segmentation/

RUN python -m unittest discover -s tests -v \
    && python tools/check_environment.py

CMD ["bash"]
