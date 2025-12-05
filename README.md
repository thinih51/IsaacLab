# Unitree G1 Simulation in NVIDIA Isaac Lab

This repository contains workflows and scripts for simulating the **Unitree G1 humanoid robot** using **NVIDIA Isaac Sim** and **Isaac Lab**. It explores both hardcoded kinematic movements and Deep Reinforcement Learning (RL) approaches for locomotion.

## 📋 Project Overview

The goal of this project was to implement a simulation environment for the Unitree G1 robot to learn autonomous walking behaviors using Reinforcement Learning (PPO). This work highlights the complexity of setting up a Sim-to-Real pipeline, from URDF importation to training configuration.

**Key Features:**
* **URDF to USD Import:** Conversion and setup of the G1 robot model for Isaac Sim (fixing root link issues).
* **Kinematic Control:** Scripts for predefined movements to validate joint articulation (Waving, Dancing).
* **RL Environment:** Custom wrapper and configuration for `RSL_RL` training.
* **Domain Randomization:** Implementation of mass, friction, and velocity push randomizations for robust training.

## ⚙️ Installation & Setup

1.  **Prerequisites:**
    * NVIDIA Isaac Sim (Version 4.x or 5.x)
    * Isaac Lab Installation
    * `RSL_RL` library for reinforcement learning

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/thinih51/IsaacLab.git](https://github.com/thinih51/IsaacLab.git)
    cd IsaacLab
    ```

3.  **Configuration:**
    Open the python scripts and ensure the `USD_PATH` variable points to your local G1 USD file location:
    ```python
    USD_PATH = "C:/Users/tshed/Documents/WIP/IsaacLab/resources/g1/g1_29dof/g1_29dof.usd"
    ```

## 🚀 Usage

Run the scripts using the Isaac Lab python launcher (e.g., `./isaaclab.bat -p` on Windows or `./isaaclab.sh -p` on Linux).

### 1. Hardcoded Movements
Test the joint articulation without neural networks to verify the robot model.

* **Arm Waving (Basic Test):**
    ```bash
    ./isaaclab.bat -p source/standalone/workflows/run_g1_move.py
    ```
* **Dance Mode (Coordination Test):**
    ```bash
    ./isaaclab.bat -p source/standalone/workflows/run_g1_dance.py
    ```

### 2. Reinforcement Learning (Training)
Train the robot to walk using PPO.
* **Command:**
    ```bash
    ./isaaclab.bat -p source/standalone/workflows/run_g1_train.py --num_envs 4096 --headless
    ```
* **Configuration:** We utilized **4096 parallel environments** with Domain Randomization enabled (mass variation, ground friction, and external pushes) to prevent overfitting.

### 3. Inference (Play)
Watch the trained model in real-time.
* **Command:**
    ```bash
    ./isaaclab.bat -p source/standalone/workflows/run_g1_play.py --num_envs 16
    ```

## 📊 Results & Observations

We conducted multiple training sessions, ranging from **30 minutes to 6 hours**.

### Physics Validation
The simulation environment was successfully set up. The physics engine (PhysX) correctly handles gravity, collisions, and friction. The "falling" behavior confirms that the robot is not statically fixed (`fix_root_link=False`) and is fully dynamic, interacting with the ground plane.

### Training Challenges
Despite 10,000 iterations and extensive Domain Randomization, the robot did not achieve a stable walking gait within the 6-hour training window. This aligns with current robotics research, which suggests that training humanoid locomotion from scratch for complex models like the G1 requires **multi-day training sessions** on high-performance compute clusters to converge to a robust policy.

## 📂 File Structure

* `run_g1_move.py`: Basic kinematic test (Arm movement).
* `run_g1_dance.py`: Advanced kinematic test (Coordination of legs, arms, and waist).
* `run_g1_train.py`: PPO Training script with Domain Randomization and reward function configuration.
* `run_g1_play.py`: Inference script to load the trained `model.pt` and visualize results in the viewport.

## 📝 License

This project is for educational purposes.