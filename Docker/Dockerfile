FROM centos:centos8  
ARG VHLS_PATH
ARG UID
ARG GID

USER root

# Basic packages
RUN dnf update -y
RUN dnf install -y gcc-c++ sudo vim openssh-clients \
                   wget curl-devel expat-devel gettext-devel zlib-devel \
                   perl-CPAN perl-devel git

# Vitis HLS dependences
RUN dnf install -y libXext libXrender libXtst gettext libtool \
                   rpm-sign autoconf automake binutils bison \
                   flex make patch pkgconfig redhat-rpm-config \
                   rpm-build ncurses-compat-libs
RUN dnf install -y https://extras.getpagespeed.com/release-el8-latest.rpm
RUN dnf install -y gperftools gperftools-devel 

# Phism dependences
RUN dnf install -y llvm clang python3 gmp-devel pcre-devel \
                   openssl openssl-devel
RUN dnf --enablerepo=powertools install texinfo -y
# build the following packages from source:
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
RUN getent group $GID || groupadd -g $GID dev-user
RUN useradd -r -g $GID -u $UID -m -d /home/dev-user -s /sbin/nologin -c "User" dev-user
RUN echo "dev-user     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER dev-user

# Add PATH
RUN echo 'export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH' >> /home/dev-user/.bashrc
RUN echo 'export LD_LIBRARY_PATH=""' >> /home/dev-user/.bashrc
ENV vhls $VHLS_PATH
RUN echo "source ${vhls}/Vitis_HLS/2020.2/settings64.sh" >> /home/dev-user/.bashrc
RUN echo 'export PATH=$PATH:/workspace/llvm/build/bin:/workspace/build/bin' >> /home/dev-user/.bashrc
RUN echo 'source /workspace/scripts/setup-vitis-hls-llvm.sh' >> /home/dev-user/.bashrc
RUN echo 'export nproc=$(grep -c ^processor /proc/cpuinfo)' >> /home/dev-user/.bashrc

WORKDIR workspace

