# ICE-CREAM: multI-agent fully CooperativE deCentRalizEd frAMework for Energy Efficiency in RAN Slicing 

## Project Scope and Objectives

The **ICE-CREAM** project addresses the growing need for efficient and sustainable network slicing in modern telecommunication networks. The framework aims to strike a balance between **energy efficiency** and **quality of service (QoS)** in sliced mobile networks. By dynamically managing the activation and deactivation of network slices and optimizing user association, ICE-CREAM provides a scalable solution to the dual challenges faced by mobile operators today.

### Key Objectives:
- **Minimizing energy consumption** in network slicing architectures.
- **Maximizing QoS** for users in a multiservice environment.
- **Deployable on real-world datasets** from the telecom industry, ensuring practical applicability.

The project is evaluated using **real-world traffic and user data from the Orange operator** but can be adapted to work with similar datasets (e.g., **Netmob dataset**).

---

## Project Components

This project consists of two main agents, along with a custom environment, to solve the joint problem of slice activation/deactivation and user association.

### 1. Slice Agent (`slice_agent.py`)
- **Timescale**: Short
- **Purpose**: The slice agent operates on a short timescale to handle rapid changes in network conditions. It decides when to activate or deactivate specific slices based on current traffic demand and resource availability.
- **Key Functions**:
  - `decision_making()`: Implements the logic for dynamic slice activation or deactivation.
  - `communicate_with_basestation()`: Interfaces with the base station agent to share information about slice statuses and requests.

### 2. Base Station Agent (`basestation_agent.py`)
- **Timescale**: Large
- **Purpose**: The base station agent operates over a longer timescale, managing the more strategic aspects of the network, such as user association and resource allocation across multiple slices. It ensures that base station resources are allocated in a way that maximizes both energy efficiency and QoS.
- **Key Functions**:
  - `optimize_resource_allocation()`: Handles long-term resource distribution among users and slices.
  - `coordinate_with_slices()`: Communicates with slice agents to maintain overall network efficiency.

### 3. Custom Environment (`environment_icecream.py`)
- **Purpose**: This environment is designed to simulate the interaction between the slice and base station agents in a real-world setting. It processes the data inputs (from traffic datasets like Orange or Netmob) and facilitates the decision-making process for the agents.
- **Key Functions**:
  - `step()`: Advances the simulation by one step, updating the network state and invoking agent decisions.
  - `reset()`: Resets the environment to an initial state, ready for a new simulation run.
  - `get_reward()`: Calculates the reward based on energy consumption and QoS metrics for reinforcement learning training.

---

## Running the Project

To run the project on a server, a **bash script** is provided to streamline the setup and execution process.

### Bash Script (`run_on_server.sh`)
This script automates the process of running the ICE-CREAM simulation on a server environment. It ensures the necessary environment variables are set and that the required datasets are properly loaded.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/ice-cream.git
