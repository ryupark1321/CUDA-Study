echo "Checking nvcc version"
nvcc --version
echo "Checking connected GPU Device"
nvidia-smi
echo "Checking GNU C Compiler Version"
gcc --version

apt update
apt install cmake