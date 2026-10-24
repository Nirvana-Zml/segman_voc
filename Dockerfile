# 基础镜像：CUDA 12.1 + cuDNN 8 + Ubuntu 20.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH="/root/anaconda3/bin:${PATH}"

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ====================== 这里换成了 Anaconda3 ======================
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh -O anaconda.sh \
    && bash anaconda.sh -b -p /root/anaconda3 \
    && rm anaconda.sh \
    && conda init bash \
    && conda clean -afy
# ==================================================================

# 创建并激活segman环境
RUN conda create -n segman python=3.10.14 -y \
    && echo "conda activate segman" >> ~/.bashrc

# 激活环境并安装PyTorch/TorchVision（适配CUDA 12.1）
RUN conda run -n segman pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
RUN conda run -n segman python -c "import torch; print('Torch 版本:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

RUN conda run -n segman pip cache purge
RUN conda run -n segman pip install "numpy<2" --force-reinstall -i http://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn

# 安装MMSegmentation v0.30.0依赖
RUN conda run -n segman pip install packaging && \
    conda run -n segman pip install -U openmim --trusted-host pypi.org --trusted-host files.pythonhosted.org && \
    conda run -n segman mim install mmcv-full==1.7.2


# 复制SegMAN源码到容器
COPY SegMAN-main/ /workspace/segman/
WORKDIR /workspace/segman

# 安装MMSegmentation
WORKDIR /workspace/segman/segmentation
RUN conda run -n segman pip install -v -e .


# 修复 mmcv 兼容 torch>=2.1.0（保留原缩进，避免 IndentationError）
COPY patch_mmcv_torch21.py /tmp/patch_mmcv_torch21.py
RUN conda run -n segman pip install "numpy<2" packaging && \
    conda run -n segman python /tmp/patch_mmcv_torch21.py


# 最后安装 natten
RUN conda run -n segman pip install natten==0.17.3+torch210cu121 \
    -f https://shi-labs.com/natten/wheels/cu121/ \
    --trusted-host shi-labs.com \
    --only-binary=natten



# 安装Selective Scan 2D
WORKDIR /workspace/segman/kernels/selective_scan
RUN conda run -n segman python setup.py bdist_wheel && \
    conda run -n segman pip install dist/*.whl


# 安装剩余依赖
WORKDIR /workspace/segman
RUN conda run -n segman pip install -r requirements.txt

# 启动默认进入 segman 环境
CMD ["/bin/bash", "-c", "source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman && exec bash"]
