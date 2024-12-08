## FLIGHT
FLIGHT is a secure aggregation framework tailored for personalized federated learning, designed to simultaneously defend against poisoning attacks from compromised clients and inference attacks from compromised servers by adversaries. By employing a lightweight defense mechanism, it enables secure aggregation in common (personalized) federated learning scenarios, providing users with a robust and secure federated learning solution. This work is based on the [FederatedScope](https://github.com/alibaba/FederatedScope) and [Piranha](https://github.com/ucbrise/piranha).

## Code Structure

- `doc/` Automatic documentation
- `environment/` Environment dependencies
- `federatedscope/` Main file for running FL tasks
- `mpc/` Main file for running FL tasks with encryption
- `scripts/` Predefined run configuration
- `setup.py` Automated installation
- `network_setting.sh` Set network environment



## Preparation

#### Requirements

We recommend that users use conda to install the required packages and ensure the CUDA version is >= 11.8.

- [Anaconda3](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)
- [CUDA](https://developer.nvidia.com/cuda-11-8-0-download-archive)

#### Clone this repository

Users need to clone this project and create a new environment using conda.

```
git clone https://github.com/SimonB6/FLIGHT_Experiment
cd flight
conda create -n FLIGHT python=3.8
```

#### Install related dependencies

Install the required packages for the framework and verify the installation settings by running the following commands.

```
conda install --file enviroment/requirements-torch1.10.txt -c pytorch -c conda-forge -c nvidia
pip install -e .
```

#### Build the code

Install the MPC dependencies by running the following commands.

**Step 1. Check out**

```
git submodule update --init --recursive
```

**Step 2. Build CUTLASS**

```
cd flight/mpc
cd ext/cutlass
mkdir build
cd build
cmake .. -DCUTLASS_NVCC_ARCHS=<GPU_ARCH> -DCMAKE_CUDA_COMPILER_WORKS=1 -DCMAKE_CUDA_COMPILER=<NVCC PATH> //Fill in <GPU_ARCH> according to your current GPU architecture and <NVCC PATH> according to the path of your system's NVCC.
export CUDA_HOME=/usr/local/cuda
make -j
```

**Step 3. Install GTest**

```
sudo apt install libgtest-dev libssl-dev
cd /usr/src/gtest
sudo mkdir build
cd build
sudo cmake ..
sudo make
sudo make install
```

**Step 4. Complie code**

```
make -j8 PIRANHA_FLAGS="-DFLOAT_PRECISION=10 -DTWOPC"
```

Users can also choose to build code with docker image.

```
sudo docker build -t flight:base -f flight.Dockerfile .
```

#### Network setting

Set up a LAN/WAN network environment.

```
# LAN setting
./network_setting.sh lan
# WAN setting
./network_setting.sh wan
# Delete setting
./network_setting.sh del
```



## Model training

#### Select dataset

In the `federatedscope/` directory, users can set `cfg.data.type = dataset_name` to conduct FL training on different datasets. The framework will automatically download the corresponding public datasets, such as:

```
cfg.data.type = 'CIFAR10@torchvision'
```

#### Select neural network model

Users can set `cfg.model.type = MODEL_NAME` to conduct FL training with different network parameter scales, for example:

```
cfg.model.type = 'resnet18'
```

#### Run Federated Learning task

We can run the following to verify if our installation process is correct:

```
python federatedscope/main.py --cfg federatedscope/example_configs/backdoor_fedavg_on_cifar10_pdr0.1
```

In the `script/` directory, we define configuration files with different predefined parameters in `.yaml` format. These parameters are designed to accommodate experiments with backdoor attacks in various federated learning (personalized federated learning) frameworks, each with different poisoning rates:

```
# Perform continuous backdoor attacks in the FedAvg framework with PMR=0.1
python federatedscope/main.py --cfg federatedscope/scripts/backdoor_scripts/backdoor_fedavg_on_cifar10_pmr0.1.yaml
# Perform continuous backdoor attacks in the FedAvg framework with PMR=0.2
python federatedscope/main.py --cfg federatedscope/scripts/backdoor_scripts/backdoor_fedavg_on_cifar10_pmr0.2.yaml
...

# Perform continuous backdoor attacks in the Ditto framework with PMR=0.1
python federatedscope/main.py --cfg federatedscope/scripts/backdoor_scripts/pfl_backdoor_ditto_on_cifar10_pmr0.1.yaml
# Perform continuous backdoor attacks in the Ditto framework with PMR=0.2
python federatedscope/main.py --cfg federatedscope/scripts/backdoor_scripts/pfl_backdoor_ditto_on_cifar10_pmr0.2.yaml
...

# Perform continuous backdoor attacks in the Fine-Tuning framework with PMR=0.1
python federatedscope/main.py --cfg federatedscope/scripts/backdoor_scripts/pfl_backdoor_ft_on_cifar10_pmr0.1.yaml
# Perform continuous backdoor attacks in the Fine-Tuning framework with PMR=0.2
python federatedscope/main.py --cfg federatedscope/scripts/backdoor_scripts/pfl_backdoor_ft_on_cifar10_pmr0.2.yaml
...
```

## Model training with MPC

#### Run Federated Learning task with MPC

We can run the following code to check that code bulid is correct:

```
# Run
./flight -p 0 -c config.json
./flight -p 1 -c config.json

# Run in docker
docker run --rm --gpus all --network host flight:base -p 0 -c config.json
docker run --rm --gpus all --network host flight:base -p 1 -c config.json
```

In the `/files` directory, we provide config files corresponding to three different training tasks and different number of clients, which users can run to implement the secure aggregation process under MPC:

```
# Run secure aggregation under mnist task with 10 clients.
./flight -p 0 -c files/config_mnist_10.json
./flight -p 1 -c files/config_mnist_10.json

# Run secure aggregation under mnist task with 25 clients.
./flight -p 0 -c files/config_mnist_25.json
./flight -p 1 -c files/config_mnist_25.json

# Run secure aggregation under mnist task with 50 clients.
./flight -p 0 -c files/config_mnist_50.json
./flight -p 1 -c files/config_mnist_50.json

# Run secure aggregation under mnist task with 100 clients.
./flight -p 0 -c files/config_mnist_50.json
./flight -p 1 -c files/config_mnist_50.json

# Run secure aggregation under cifar-10s task with 10 clients.
./flight -p 0 -c files/config_cifar10s_10.json
./flight -p 1 -c files/config_cifar10s_10.json

# Run secure aggregation under cifar-10l task with 10 clients.
./flight -p 0 -c files/config_cifar10l_10.json
./flight -p 1 -c files/config_cifar10l_10.json
```
