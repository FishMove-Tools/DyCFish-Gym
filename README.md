<div>
  <h1>
    🐟 DyCFish-Gym&nbsp;&nbsp;&nbsp;
    <span style="float: right; font-size: 16px; font-weight: normal; margin-top: 10px;">
     <b>English</b> | <a href="README_zh.md"> 中文</a>
    </span>
  </h1>
</div>

<p align="center">
  <em>An Intelligent Control Platform Bridging Reduced-Order Dynamics and Computational Fluid Dynamics for Propulsion</em>
</p>

<p align="center">
  <img src="./assets/framework_figure/Framework.png" width="100%">
</p>


### 🧠 Tech Stack / Tags

![](https://img.shields.io/badge/DeepRL-%23369FF7FF)  ![](https://img.shields.io/badge/BioRobotics-%23669FF7FF)  ![](https://img.shields.io/badge/CFD-%23766BF7FF)  ![](https://img.shields.io/badge/PPO-%23766BF7FF)  ![](https://img.shields.io/badge/ThunniformPropulsion-%23669FF7FF)  ![](https://img.shields.io/badge/Sim--to--Real-%2366BB66FF)  ![](https://img.shields.io/badge/GymEnv-%2366BB66FF)  ![](https://img.shields.io/badge/PyFluent-%23F7B93EFF)

---

## 📋 Contents
- [🏠 About](#-about)
- [📁 Project Structure](#-project-structure)
- [📚 Getting Started](#-getting-started)
- [🚀 Usage](#-usage)
- [📦 Benchmark & Method](#-benchmark--method)
- [📝 TODO List](#-todo-list)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

---

## 🏠 About

**DyCFish-Gym** is a unified intelligent control platform for bio-inspired thunniform propulsion. It bridges **reduced-order dynamics (ROM)** and **high-fidelity Computational Fluid Dynamics (CFD)** via a two-stage deep reinforcement learning (DRL) framework, achieving efficient policy learning while maintaining physical consistency.

Training a DRL agent entirely within full-order CFD solvers requires millions of environmental interactions, making it computationally prohibitive. DyCFish-Gym addresses this fundamental **efficiency–fidelity trade-off** through a hierarchical architecture:

1. **Stage 1 — ROM Pre-training:** A reduced-order articulated dynamics model enables rapid policy pretraining. The ROM captures essential degrees of freedom (body + tail joint) with CPG-parameterized actuation, enabling >10³ simulation steps per second.
2. **Stage 2 — CFD Fine-tuning:** The pretrained policy is transferred to a high-fidelity CFD environment (ANSYS Fluent) via the PyFluent interface for refinement under fully resolved Navier–Stokes equations. This stage accounts for only 10–15% of total training time.

The platform is systematically validated across **three representative closed-loop tasks**:
- 🎯 **Trajectory tracking** — straight-line, semicircular, and W-shaped paths
- 🏃 **Rapid escape** — predator evasion via C-start mechanism reconstruction
- ⚓ **Station-keeping** — maintaining position against uniform inflow disturbances

Key features include:

* **🧬 Biologically Interpretable Behaviors:** The DRL agent autonomously learns bio-inspired mechanisms, such as exploiting reverse Kármán vortices to minimize cost of transport (COT).
* **📊 Outperforms Classical Controllers:** 40% lower trajectory error, 30% smaller steady-state deviation, and 20% lower energy cost relative to PID and MPC baselines.
* **🤖 Sim-to-Real Transfer:** Learned policies are successfully deployed on a physical dual-joint robotic fish for trajectory tracking experiments.

---
<!-- 
## 📁 Project Structure

```
DyCFish-Gym/
├── README.md                          # Project documentation
├── dynamic_stage/                     # Stage 1: Reduced-order model pre-training
│   ├── fish_env.py                    # Gymnasium-based ROM environment (FishEnv)
│   ├── train.py                       # PPO pre-training script
│   └── test.py                        # Evaluation & video recording
├── CFD_stage/                         # Stage 2: CFD fine-tuning
│   ├── EnvFluent.py                   # ANSYS Fluent Gymnasium environment (FluentEnv)
│   ├── training.py                    # Multi-worker CFD training with checkpoint resume
│   └── lstm_policy.py                 # Custom LSTM policy network
└── assets/
    ├── framework_figure/              # Architecture diagrams
    │   └── Framework.png
    └── demos/                         # Demo videos
        ├── video1.mp4
        ├── video2.mp4
        └── video3.mp4
``` -->

## 📚 Getting Started

### Prerequisites

* Operating System: Windows or Linux (Ubuntu 20.04+ recommended)
* NVIDIA GPU (Optional, but recommended for PyTorch training)
* **ANSYS Fluent** (Required for Stage 2, must be installed and configured for `ansys-fluent-core`)
* Conda
* Python 3.9

### Installation

#### 1. Python Environment Setup

```bash
# Create and activate conda environment
conda create -n dycfish python=3.9.13
conda activate dycfish

# Upgrade pip
pip install --upgrade pip

# Install core libraries (Deep Learning and RL)
pip install numpy==2.0.2
pip install torch==2.1.0
pip install stable-baselines3[extra]

# Install visualization and utility packages
pip install pygame opencv-python matplotlib

# Install Pyfluent package (required for CFD stage)
pip install ansys-fluent-core

# Adjust Pandas version (to prevent conflicts)
pip uninstall pandas -y
pip install pandas==2.2.2
```

#### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DyCFish-Gym.git
cd DyCFish-Gym
```

#### 3. Pyfluent Setup (for Stage 2)

The `ansys-fluent-core` package enables seamless control of ANSYS Fluent from Python.

* For detailed documentation and guides, refer to the official repository: [PyFluent](https://github.com/ansys/pyfluent).

**Controlling Fluent via Python:**

| Operation | Command | Description |
| :--- | :--- | :--- |
| **Import** | `import ansys.fluent.core as pyfluent` | Import the core library |
| **Launch (No GUI)** | `session = pyfluent.launch_fluent()` | Start Fluent without GUI |
| **Launch (With GUI)** | `session = pyfluent.launch_fluent(show_gui=True)` | Start Fluent with GUI (meshing mode only) |
| **Exit** | `session.exit()` | Close the Fluent session |

**Pyfluent Journaling (Code Generation):**

1. Launch Fluent: `session = pyfluent.launch_fluent()`
2. Start recording: `(api-start-python-journal "python_journal.py")`
3. Perform TUI commands → Python code is generated in `python_journal.py`
4. Stop recording: `(api-stop-python-journal)`

---

## 🚀 Usage

### Stage 1: ROM Pre-training

The first stage uses the reduced-order dynamics environment for rapid policy learning.

**Train the agent:**

```bash
cd dynamic_stage
python train.py
```

This will:
- Launch the `FishEnv` environment with real-time Pygame rendering
- Train a PPO agent for 1,000,000 timesteps
- Save model checkpoints every 10,000 steps to `./models/`
- Log episode rewards to `./logs/episode_rewards.txt`
- Support TensorBoard monitoring via `./tensorboard/`

**Monitor training:**

```bash
tensorboard --logdir ./tensorboard/
```

**Evaluate a trained model:**

```bash
cd dynamic_stage
python test.py
```

This loads a trained model, runs 5 evaluation episodes, and saves the results as a video to `./logs/test_escape.mp4`.

### Stage 2: CFD Fine-tuning

The second stage fine-tunes the pretrained policy in a high-fidelity ANSYS Fluent CFD environment.

> ⚠️ **Note:** ANSYS Fluent must be installed and properly configured before running this stage.

**Run CFD training:**

```bash
cd CFD_stage
python training.py
```

This will:
- Launch ANSYS Fluent (double-precision 2D solver, 6 processors)
- Train with multi-worker support and automatic checkpoint resume
- Save models with local/global best tracking to `./saved_models/`
- Log detailed performance metrics per episode

---

## 📦 Benchmark & Method

### Platform Architecture

DyCFish-Gym is organized into four tightly coupled functional modules:

| Module | Description |
| :--- | :--- |
| **CPG Parameterization** | Encodes biological kinematics into a low-dimensional actuation space (amplitude *A*, frequency *f*) |
| **Model Construction** | ROM for rapid pretraining + CFD for high-fidelity refinement |
| **DRL Control** | PPO-based policy learning with stable cross-model transfer |
| **Strategy Biomimicry** | Behavioral evaluation: decodes optimized swimming strategies for mechanistic insights |

### Environment Specifications

| Parameter | Dynamic Stage (`FishEnv`) | CFD Stage (`FluentEnv`) |
| :--- | :--- | :--- |
| **Observation** | 7D: [*x, y, ψ, vx, vy, ωz, t*] | 7D: [*x, y, θ, vx, vy, ωz, t*] |
| **Action** | 2D: [*amplitude, frequency*] | 2D: [*frequency, amplitude*] |
| **Physics** | Reduced-order articulated dynamics | Unsteady Navier–Stokes (Fluent) |
| **Speed** | >10³ steps/s | ~50 steps/s |


### Baseline Comparison

| Method | Trajectory Tracking RMSE | Station-keeping Deviation | Escape Success | Energy Cost |
| :--- | :--- | :--- | :--- | :--- |
| PID | 1.00 | 1.00 | 0.82 | 1.00 |
| MPC | 0.78 | 0.83 | 0.87 | 0.92 |
| **DyCFish-Gym** | **0.60** | **0.68** | **0.96** | **0.78** |

<!-- ### Demo Videos

<p align="center">
  <table>
    <tr>
      <td align="center"><b>Trajectory Tracking</b></td>
      <td align="center"><b>Predator Escape</b></td>
      <td align="center"><b>Station-Keeping</b></td>
    </tr>
    <tr>
      <td align="center"><a href="./assets/demos/video1.mp4">🎬 Video 1</a></td>
      <td align="center"><a href="./assets/demos/video2.mp4">🎬 Video 2</a></td>
      <td align="center"><a href="./assets/demos/video3.mp4">🎬 Video 3</a></td>
    </tr>
  </table>
</p> -->

---

## 📝 TODO List
- \[x\] Release CFD_stage training code.
- \[x\] Release dynamic_stage code.
- \[\] Release the demo videos.
- \[ \] Release the paper.

---

<!-- ## 🔗 Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{sun2025dycfishgym,
  title={DyCFish-Gym: An Intelligent Control Platform Bridging Reduced-Order Dynamics and Computational Fluid Dynamics for Thunniform Propulsion},
  author={Sun, Weiyuan and Zhan, Ruixin and Jiang, Haokui and Li, Xiaofan and Huang, Qingdong and Cao, Shunxiang},
  journal={International Journal of Mechanical Sciences},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

--- -->

## 👏 Acknowledgements

This work was supported by the **National Key R&D Program of China** (Grant No. 2024YFC3013200).

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO implementation
- [Gymnasium](https://gymnasium.farama.org/) — RL environment interface
- [ANSYS PyFluent](https://github.com/ansys/pyfluent) — Python interface for ANSYS Fluent
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Pygame](https://www.pygame.org/) — Real-time visualization
