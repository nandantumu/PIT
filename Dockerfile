FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
RUN apt update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y python3 python3-pip git wget curl libgl1-mesa-glx apt-utils libglib2.0-0 cmake

RUN apt -y install tzdata vim tmux 
#cuda-11-6 
#libnccl2=2.12.12-1+cuda11.6 libnccl-dev=2.12.12-1+cuda11.6\

RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - 
RUN apt-get install -y nodejs

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lcal/cuda/

RUN pip3 install jupyterlab jupyterlab_widgets
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install jupyterlab_nvdashboard scikit-image scikit-learn ipdb tqdm gpustat
RUN git config --global --add safe.directory /PIMP/PIT
RUN git config --global --add safe.directory /PIMP/LaneGCN
RUN git config --global --add safe.directory /PIMP/Trajectron-plus-plus
RUN git config --global --add safe.directory /PIMP/VehicleMotionPrediction

RUN bash -c "$(wget https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh -O -)"
#COPY ../.bashrc ~/.bashrc

CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"

