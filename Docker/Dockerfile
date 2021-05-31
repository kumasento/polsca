FROM centos:centos7
ARG VHLS_PATH
ARG UID
ARG GID

USER root

# Basic packages
RUN yum install -y sudo gcc-c++ vim git openssh-clients
RUN yum install -y wget 

# Vitis HLS dependences
RUN yum install -y libXext libXrender libXtst gettext libtool
RUN yum install -y boost-devel gperftools gperftools-devel rpm-sign
RUN yum install -y autoconf automake binutils bison flex 
RUN yum install -y make patch pkgconfig redhat-rpm-config rpm-build

# Phism dependences
RUN yum install -y llvm clang llvm-devel
# cmake 3.12
RUN (cd /tmp; wget https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz; tar zxvf cmake-3.*; cd cmake-3.*; ./bootstrap --prefix=/usr/local; make -j4; make install)

# RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 100
# RUN update-alternatives --install /usr/bin/FileCheck FileCheck /usr/bin/FileCheck-9 100
# RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 100
# RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-9 100

CMD ["bash"]

# Add dev-user
RUN getent group $GID || groupadd $GID
RUN useradd -r -g $GID -u $UID -m -d /home/dev-user -s /sbin/nologin -c "User" dev-user
RUN echo "dev-user     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER dev-user

RUN echo 'export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH' >> /home/dev-user/.bashrc
RUN echo 'export LD_LIBRARY_PATH=""' >> /home/dev-user/.bashrc
ENV vhls $VHLS_PATH
RUN echo "source ${vhls}/Vitis_HLS/2020.2/settings64.sh" >> /home/dev-user/.bashrc
RUN echo 'export PATH=$PATH:/workspace/llvm/build/bin:/workspace/build/bin' >> /home/dev-user/.bashrc

WORKDIR workspace
