# lightning-kit

## Overview
lightning-kit is an open-source reactive programming framework written in C++ designed to facilitate efficient data transfer between GPUs (Graphics Processing Units), NICs (Network Interface Cards), DPUs (Data Processing Units), and various storage devices.
This project aims to optimize data transfer and processing operations, making it highly suitable for applications in high-performance computing, data analytics, and machine learning environments.

## Features
- High-Performance Data Transfers: Utilize advanced algorithms to maximize throughput and minimize latency when moving data across different hardware interfaces.
- Cross-Platform Compatibility: Supports a wide range of GPUs, NICs, DPUs, and storage devices across various platforms and architectures.
- Easy Integration: Designed to be easily integrated with existing applications and workflows, providing a seamless data movement solution.
- Flexible API: Offers a comprehensive and flexible API, allowing developers to customize data transfer operations to meet their specific requirements.
- Security and Reliability: Implements robust error handling and data integrity checks to ensure secure and reliable data movement.

## Installation
### Prerequisites
- C++ Compiler
- CMake (>=3.12)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (>=12.1)
- [DOCA](https://developer.nvidia.com/networking/doca) (==2.2.0)

### Build
Clone the repository:
```bash
git clone https://github.com/fixstars/lightning-kit.git
cd lightning-kit
```

Build the project using CMake:
```bash
mkdir build && cd build
cmake ..
make
```

### Test

On server:
```
echo 2048 | sudo tee /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
echo 8 | sudo tee /sys/devices/system/node/node0/hugepages/hugepages-1048576kB/nr_hugepages
sudo modprobe nvidia-peermem
sudo ip link set dev <dev-name> mtu 8000
sudo ip addr add 10.0.0.2.2/24 dev <dev-name>

sudo ./build/test
```

On client:
```
sudo ip link set dev <dev-name> mtu 8000
sudo ip link addr 10.0.0.1/24 dev <dev-name>

netcat 10.0.0.2 1234 < <some-file>
```

## Contributing
We welcome contributions to lightning-kit! If you'd like to contribute, please follow these steps:

- Fork the repository and clone your fork.
- Create a new branch for your feature or bug fix.
- Develop and test your changes.
- Submit a Pull Request against the main branch with a clear description of the changes and any relevant issue numbers.

## License
lightning-kit is released under the MIT License with common clause. See the LICENSE file for more information.
