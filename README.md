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

<<<<<<< HEAD
2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/thinih51/IsaacLab.git](https://github.com/thinih51/IsaacLab.git)
    cd IsaacLab
    ```
=======
A detailed description of Isaac Lab can be found in our [arXiv paper](https://arxiv.org/abs/2511.04831).
>>>>>>> 02b7064df9c6c4948ce49a425a6eca51d24bad9d

3.  **Configuration:**
    Open the python scripts and ensure the `USD_PATH` variable points to your local G1 USD file location:
    ```python
    USD_PATH = "C:/Users/tshed/Documents/WIP/IsaacLab/resources/g1/g1_29dof/g1_29dof.usd"
    ```

## 🚀 Usage

<<<<<<< HEAD
Run the scripts using the Isaac Lab python launcher (e.g., `./isaaclab.bat -p` on Windows or `./isaaclab.sh -p` on Linux).
=======
- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with more than 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.
>>>>>>> 02b7064df9c6c4948ce49a425a6eca51d24bad9d

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

<<<<<<< HEAD
This project is for educational purposes.
=======
We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell: Share Your Inspiration

We encourage you to utilize our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell)
area in the `Discussions` section of this repository. This space is designed for you to:

* Share the tutorials you've created
* Showcase your learning content
* Present exciting projects you've developed

By sharing your work, you'll inspire others and contribute to the collective knowledge
of our community. Your contributions can spark new ideas and collaborations, fostering
innovation in robotics and simulation.

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas,
  asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of
  work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features,
  or general updates.

## Connect with the NVIDIA Omniverse Community

Do you have a project or resource you'd like to share more widely? We'd love to hear from you!
Reach out to the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to explore opportunities
to spotlight your work.

You can also join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to
connect with other developers, share your projects, and help grow a vibrant, collaborative ecosystem
where creativity and technology intersect. Your contributions can make a meaningful impact on the Isaac Lab
community and beyond!

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its
corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its
dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

Note that Isaac Lab requires Isaac Sim, which includes components under proprietary licensing terms. Please see the [Isaac Sim license](docs/licenses/dependencies/isaacsim-license.txt) for information on Isaac Sim licensing.

Note that the `isaaclab_mimic` extension requires cuRobo, which has proprietary licensing terms that can be found in [`docs/licenses/dependencies/cuRobo-license.txt`](docs/licenses/dependencies/cuRobo-license.txt).


## Citation

If you use Isaac Lab in your research, please cite the technical report:

```
@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}
```

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework.
We gratefully acknowledge the authors of Orbit for their foundational contributions.
>>>>>>> 02b7064df9c6c4948ce49a425a6eca51d24bad9d
