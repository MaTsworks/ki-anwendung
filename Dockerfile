FROM rocm/pytorch:latest

# Install YOLOv4 dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    wget

# Install Python dependencies
RUN pip3 install --no-cache-dir numpy opencv-python==4.5.5.64 torch torchvision tqdm

# Clone YOLOv4 repo
RUN git clone https://github.com/AlexeyAB/darknet.git /workspace/darknet && \
    cd /workspace/darknet && \
    make -j$(nproc)

WORKDIR /workspace/darknet

RUN apt install -y tar


# Download and extract datasets \
RUN mkdir -p /workspace/darknet/data/ && \
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar && \
    tar -xvf VOCtrainval_11-May-2012.tar -C /workspace/darknet/data/ && \
    rm VOCtrainval_11-May-2012.tar

WORKDIR /workspace/darknet

COPY src/format.py .

CMD ["python", "format.py", "--voc_dir", "/workspace/darknet/data/VOCdevkit/VOC2012/Annotations", "--output_dir", "/workspace/darknet/data/labels", "--labels_file", "/workspace/darknet/data/voc.names"]
