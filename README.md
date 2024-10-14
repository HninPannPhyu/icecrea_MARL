# ICE-CREAM: multI-agent fully CooperativE deCentRalizEd frAMework for Energy Efficiency in RAN Slicing 

## Project Scope and Objectives

The **ICE-CREAM** project addresses the growing need for efficient and sustainable network slicing in modern telecommunication networks. The framework aims to strike a balance between **energy efficiency** and **quality of service (QoS)** in sliced mobile networks. By dynamically managing the activation and deactivation of network slices and optimizing user association, ICE-CREAM provides a scalable solution to the dual challenges faced by mobile operators today.

### Key Objectives:
- **Minimizing energy consumption** in network slicing architectures.
- **Maximizing QoS** for users in a multiservice environment.
- **Deployable on real-world datasets** from the telecom industry, ensuring practical applicability.

The project is evaluated using **real-world traffic and user data from the Orange operator in France** but can be adapted to work with similar datasets (e.g., **Netmob dataset**).

---

## Project Components

This project consists of two main agents, along with a custom environment, to solve the joint problem of slice activation/deactivation and user association.

### 1. Slice Agent (`Slice_agent.py`)
- **Timescale**: Short
- **Purpose**: The slice agent operates on a short timescale to handle rapid changes in network conditions. It decides the user association to specific slice instance (including EcoSlice) considering the impact on the neighboring basestation.


### 2. Base Station Agent (`Basestation_agent.py`)
- **Timescale**: Large
- **Purpose**: The base station agent operates over a longer timescale. It decides when to activate or deactivate specific slices based on current traffic demand.

### 3. Custom Environment (`Environment_icecream.py`)
- **Purpose**: This environment is designed to simulate the interaction between the slice and base station agents in a real-world setting. It processes the data inputs (from traffic datasets like Orange or Netmob) and facilitates the decision-making process for the agents.


## Running the Project

To run the project on a server (i.e. compute canada linux server), a **bash script** is provided to streamline the setup and execution process.


