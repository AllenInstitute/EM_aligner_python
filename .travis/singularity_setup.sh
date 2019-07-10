#!/bin/bash -ex

# install go
export VERSION=1.11 OS=linux ARCH=amd64
wget -q https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz
sudo tar -C /usr/local -xzf go$VERSION.$OS-$ARCH.tar.gz
rm go$VERSION.$OS-$ARCH.tar.gz
echo 'export GOPATH=${HOME}/go' >> ~/.bashrc
echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc
source ~/.bashrc
go get -u github.com/golang/dep/cmd/dep
sudo sed -i -e 's/^Defaults\tsecure_path.*$//' /etc/sudoers

# Check Python
echo "Python Version:"
python --version
pip install sregistry[all]
sregistry version
echo "sregistry Version:"

# Install Singularity
export VERSION=3.1.1 
mkdir -p "$GOPATH"/src/github.com/sylabs
cd "$GOPATH"/src/github.com/sylabs
wget -q https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz
tar -xzf singularity-${VERSION}.tar.gz
cd singularity

./mconfig
make -C ./builddir
sudo make -C ./builddir install

export PATH="${GOPATH}/bin:${PATH}"

mkdir -p "${GOPATH}/src/github.com/sylabs"
cd "${GOPATH}/src/github.com/sylabs"

git clone -b release-3.2 https://github.com/sylabs/singularity
cd singularity
./mconfig -v -p /usr/local
make -j $(nproc 2>/dev/null || echo 1) -C ./builddir all
sudo make -C ./builddir install
