FROM centos:centos7
ARG VHLS_PATH
ARG UID
ARG GID

USER root

# Basic packages
RUN yum update -y
RUN yum install -y sudo gcc-c++ vim git openssh-clients \
                   wget glibc-devel glibc-static

# Vitis HLS dependences
RUN yum install -y libXext libXrender libXtst gettext libtool \
                   boost-devel gperftools gperftools-devel rpm-sign \
                   autoconf binutils bison flex make \
                   patch pkgconfig redhat-rpm-config rpm-build

# Phism dependences
RUN yum install -y llvm clang llvm-devel python3 gmp-devel clang-devel \
                   perl pcre-devel zlib-devel textinfo
# openssl - required by cmake
RUN cd /tmp \
    && wget https://ftp.openssl.org/source/old/1.1.1/openssl-1.1.1.tar.gz \
    && tar xvf openssl-1.1.1.tar.gz \
    && cd openssl-1.1.1/ \
    && ./config --prefix=/usr --openssldir=/etc/ssl --libdir=lib no-shared zlib-dynamic \
    && make -j $(grep -c ^processor /proc/cpuinfo) \
    && make install
# cmake 3.20
RUN cd /tmp \
    && wget https://github.com/Kitware/CMake/releases/download/v3.20.3/cmake-3.20.3.tar.gz \
    && tar zxvf cmake-3.* \
    && cd cmake-3.* \
    && ./bootstrap --prefix=/usr/local \
    && make -j $(grep -c ^processor /proc/cpuinfo) \
    && make install
# automake 1.14
RUN cd /tmp \
    && wget http://ftp.gnu.org/gnu/automake/automake-1.14.tar.gz \
    && tar xvzf automake-1.14.tar.gz \
    && cd automake-1.14 \
    && ./configure \
    && make -j $(grep -c ^processor /proc/cpuinfo) \
    && make install
RUN yum install -y centos-release-scl-rh
RUN yum install -y llvm-toolset-7-llvm
RUN scl enable llvm-toolset-7 bash

# Centos issue with libtool:
# https://superuser.com/questions/565988/autoconf-libtool-and-an-undefined-ac-prog-libtool
RUN for file in argz libtool ltdl ltoptions ltsugar ltversion lt~obsolete \
    do \
      ln -s /usr/share/aclocal/$file.m4 /usr/local/share/aclocal/$file.m4 \
    done

CMD ["bash"]

# Add dev-user
RUN getent group $GID || groupadd -g $GID dev-user
RUN useradd -r -g $GID -u $UID -m -d /home/dev-user -s /sbin/nologin -c "User" dev-user
RUN echo "dev-user     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER dev-user

RUN echo 'export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH' >> /home/dev-user/.bashrc
RUN echo 'export LD_LIBRARY_PATH=""' >> /home/dev-user/.bashrc
ENV vhls $VHLS_PATH
RUN echo "source ${vhls}/Vitis_HLS/2020.2/settings64.sh" >> /home/dev-user/.bashrc
RUN echo 'export PATH=/workspace/llvm/build/bin:/workspace/build/bin:/tmp/cmake-3.20.3/bin:$PATH' >> /home/dev-user/.bashrc
RUN echo 'source /workspace/scripts/setup-vitis-hls-llvm.sh' >> /home/dev-user/.bashrc
RUN echo 'export nproc=$(grep -c ^processor /proc/cpuinfo)' >> /home/dev-user/.bashrc

WORKDIR workspace
