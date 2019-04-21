FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV NVIDIA_VISIBLE_DEVICES "all"
ENV NVIDIA_DRIVER_CAPABILITIES "video,compute,utility"

ARG USERID=1000
ARG GROUPID=1000
ARG USERNAME=ccv

ENV APT_CLEAN_CMD "sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*"

RUN apt-get update && apt-get -y upgrade && \
    apt-get -y install sudo zsh wget git && \
    eval $APT_CLEAN_CMD

RUN groupadd -g $GROUPID $USERNAME && \
    useradd -m -u $USERID -g $GROUPID -o $USERNAME && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    usermod -s /bin/zsh $USERNAME && \
    usermod -aG sudo $USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

# Install ZSH theme
RUN wget --no-check-certificate http://install.ohmyz.sh -O - | zsh || true && \
    echo "local ret_status=\"%(?:%{\$fg_bold[green]%}➜ :%{\$fg_bold[red]%}➜ )\"\nPROMPT='\${ret_status} %{\$fg[green]%}%n%{\$fg[yellow]%}@%{\$fg[magenta]%}%m%{\$fg[cyan]%}/%c%{\$reset_color%} \$(git_prompt_info)'\n\nZSH_THEME_GIT_PROMPT_PREFIX=\"%{\$fg_bold[blue]%}git:(%{\$fg[red]%}\"\nZSH_THEME_GIT_PROMPT_SUFFIX=\"%{\$reset_color%} \"\nZSH_THEME_GIT_PROMPT_DIRTY=\"%{\$fg[blue]%}) %{\$fg[yellow]%}✗\"\nZSH_THEME_GIT_PROMPT_CLEAN=\"%{\$fg[blue]%})\""  > /home/$USERNAME/.oh-my-zsh/themes/$USERNAME.zsh-theme && \
    sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="$USERNAME"/g' /home/$USERNAME/.zshrc
CMD source ~/.zshrc

RUN sudo apt-get update && sudo apt-get -y upgrade && \
    sudo apt-get -y install \
        python3-dev python3-pip \
        # enables add-apt-repository
        software-properties-common && \
    eval $APT_CLEAN_CMD

# PyCharm dependencies
RUN sudo add-apt-repository ppa:openjdk-r/ppa && sudo apt-get update && \
    sudo apt-get install -y libcanberra-gtk-module openjdk-8-jdk -y && \
    eval $APT_CLEAN_CMD

# OpenCV, Matplotlib, Numpy, Pytorch, TorchVision..
RUN pip3 install opencv-python opencv-contrib-python matplotlib
RUN pip3 install numpy torchvision_nightly && \
    pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html

ENV QT_X11_NO_MITSHM "1"
