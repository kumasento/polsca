# This Dockerfile configures a Docker environment that 
# contains all the required packages for the tool
FROM ubuntu:20.04
ARG UID
ARG GID
ARG VHLS_PATH
RUN echo "Group ID: $GID"
RUN echo "User ID: $UID"

USER root
RUN apt-get update -y && apt-get install apt-utils -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Install basic packages 
RUN apt-get upgrade -y 
RUN apt-get update -y \
    && apt-get install -y clang-9 graphviz-dev libclang-9-dev \
                          pkg-config g++ libxtst6 xdg-utils \
                          libboost-all-dev llvm gcc ninja-build \
                          python3 python3-pip build-essential \
                          libssl-dev git vim wget htop sudo \
                          lld parallel ninja-build
RUN apt-get update -y \
    && apt-get install -y build-essential libtool autoconf \
                          pkg-config flex bison libgmp-dev \
                          clang-9 libclang-9-dev texinfo \
                          vim ninja-build git libncurses5 gcc-multilib \
                          libidn11

RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 100 \
    && update-alternatives --install /usr/bin/FileCheck FileCheck /usr/bin/FileCheck-9 100 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-9 100

# Append any packages you need here
# RUN apt-get ...

# Build package from source
RUN mkdir -p /dkrPkgs
# cmake 3.20
RUN cd /dkrPkgs \
    && wget https://github.com/Kitware/CMake/releases/download/v3.20.3/cmake-3.20.3.tar.gz \
    && tar zxvf cmake-3.20.3.tar.gz \
    && cd cmake-3.20.3 \
    && ./bootstrap --prefix=/usr/local \
    && make -j $(grep -c ^processor /proc/cpuinfo) \
    && make install

CMD ["bash"]

# Add dev-user
RUN groupadd -o -g $GID dev-user
RUN useradd -r -g $GID -u $UID -m -d /home/dev-user -s /sbin/nologin -c "User" dev-user
RUN echo "dev-user     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER dev-user

# Install PyTorch and Torch-MLIR
ENV PATH="${PATH}:/home/dev-user/.local/bin"
RUN pip3 install --user --upgrade pip \
    && pip3 install pandas dataclasses colorlog pyyaml


# Add environment variables
ENV vhls $VHLS_PATH
RUN printf "\
\nexport LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LIBRARY_PATH \
\n# Vitis HLS setup \
\nsource ${vhls}/Vitis_HLS/2020.2/settings64.sh \
\nexport PATH=/dkrPkgs/cmake-3.20.3/bin:$PATH:/workspace/bin:/workspace/polygeist/llvm/build/bin:/workspace/polygeist/build/bin:/home/dev-user/.local/bin \
\n# Thread setup \
\nexport nproc=\$(grep -c ^processor /proc/cpuinfo) \
\n# Terminal color... \
\nexport PS1=\"[\\\\\\[\$(tput setaf 3)\\\\\\]\\\t\\\\\\[\$(tput setaf 2)\\\\\\] \\\u\\\\\\[\$(tput sgr0)\\\\\\]@\\\\\\[\$(tput setaf 2)\\\\\\]\\\h \\\\\\[\$(tput setaf 7)\\\\\\]\\\w \\\\\\[\$(tput sgr0)\\\\\\]] \\\\\\[\$(tput setaf 6)\\\\\\]$ \\\\\\[\$(tput sgr0)\\\\\\]\" \
\nexport LS_COLORS='rs=0:di=01;96:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01' \
\nalias ls='ls --color' \
\nalias grep='grep --color'\n" >> /home/dev-user/.bashrc
#Add vim environment
RUN printf "\
\nset autoread \
\nautocmd BufWritePost *.cpp silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.c   silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.h   silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.hpp silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.cc  silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.py  silent! !python3 -m black <afile> \
\nautocmd BufWritePost *.sv  silent! !verible-verilog-format --inplace <afile> \
\nautocmd BufWritePost *.v  silent! !verible-verilog-format --inplace <afile> \
\nautocmd BufWritePost * redraw! \
\n" >> /home/dev-user/.vimrc

# Entrypoint set up
WORKDIR workspace

