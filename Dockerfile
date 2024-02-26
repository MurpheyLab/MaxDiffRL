FROM thewtex/opengl:ubuntu2004@sha256:dbd220bf2cbe3ffeb8ec7f803ef92d0495a5bb2c79a1423d75925810fb04723d

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gedit \
    gir1.2-keybinder-3.0 \
    gnome-terminal \
    htop \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    python3-dev \
    python3-pip \
    python3-tk \
    unzip \
    vim \
    wget \
    && apt-get clean 

USER user

ENV HOME /home/user

WORKDIR $HOME

RUN mkdir -p $HOME/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d $HOME/.mujoco \
    && mv $HOME/.mujoco/mujoco200_linux $HOME/.mujoco/mujoco200 \
    && rm mujoco.zip \
    && wget https://www.roboti.us/file/mjkey.txt \
    && mv mjkey.txt $HOME/.mujoco/mjkey.txt

ENV LD_LIBRARY_PATH $HOME/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

WORKDIR $HOME/work

ADD ./requirements.txt $HOME/work/

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir jupyterlab

ADD . $HOME/work

RUN echo "LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}" >> $HOME/.bashrc
RUN echo "PATH=$HOME/.local/bin:${PATH}" >> $HOME/.bashrc
RUN echo "alias python=python3" >> $HOME/.bashrc
RUN echo "gedit $HOME/work/README.md &" >> $HOME/.bashrc

USER root

RUN mkdir $HOME/work/results
RUN chown -R user $HOME/work
VOLUME $HOME/work/results
