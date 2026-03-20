<div>
  <h1>
    🐟 DyCFish-Gym&nbsp;&nbsp;&nbsp;
    <span style="float: right; font-size: 16px; font-weight: normal; margin-top: 10px;">
     <a href="README.md">English</a> | <b>中文</b>
    </span>
  </h1>
</div>

<p align="center">
  <em>融合降阶动力学与计算流体动力学的鲔形推进智能控制平台</em>
</p>

<p align="center">
  <img src="./assets/framework_figure/Framework.png" width="100%">
</p>


### 🧠 技术栈 / 标签

![](https://img.shields.io/badge/DeepRL-%23369FF7FF)  ![](https://img.shields.io/badge/BioRobotics-%23669FF7FF)  ![](https://img.shields.io/badge/CFD-%23766BF7FF)  ![](https://img.shields.io/badge/PPO-%23766BF7FF)  ![](https://img.shields.io/badge/ThunniformPropulsion-%23669FF7FF)  ![](https://img.shields.io/badge/Sim--to--Real-%2366BB66FF)  ![](https://img.shields.io/badge/GymEnv-%2366BB66FF)  ![](https://img.shields.io/badge/PyFluent-%23F7B93EFF)

---

## 📋 目录
- [🏠 项目简介](#-项目简介)
- [📚 快速开始](#-快速开始)
- [🚀 使用方法](#-使用方法)
- [📦 基准测试与方法](#-基准测试与方法)
- [📝 待办事项](#-待办事项)
- [👏 致谢](#-致谢)

---

## 🏠 项目简介

**DyCFish-Gym** 是一个面向仿生鲔形推进的统一智能控制平台。它通过两阶段深度强化学习（DRL）框架，融合**降阶动力学模型（ROM）**与**高保真计算流体动力学（CFD）**，在保持物理一致性的同时实现高效的策略学习。

在全阶 CFD 求解器中训练 DRL 智能体需要数百万次环境交互，计算成本极为高昂。DyCFish-Gym 通过分层架构解决了这一核心的**效率-保真度权衡**问题：

1. **第一阶段 — ROM 预训练：** 采用降阶铰接体动力学模型进行快速策略预训练。ROM 以 CPG 参数化驱动方式捕捉关键自由度（鱼体 + 尾部关节），模拟速度可达每秒 >10³ 步。
2. **第二阶段 — CFD 微调：** 将预训练策略通过 PyFluent 接口迁移到高保真 CFD 环境（ANSYS Fluent）中，在完整求解 Navier–Stokes 方程的条件下进行策略精调。此阶段仅占总训练时间的 10–15%。

平台在**三个代表性闭环控制任务**上进行了系统验证：
- 🎯 **轨迹跟踪** — 直线、半圆和 W 形路径
- 🏃 **快速逃逸** — 通过 C-start 机制重建实现捕食者规避
- ⚓ **定点保持** — 在均匀来流扰动下保持位置

核心特性：

* **🧬 生物可解释行为：** DRL 智能体自主学习仿生机制，例如利用反向 Kármán 涡街最小化运输能耗（COT）。
* **📊 超越经典控制器：** 相比 PID 和 MPC 基线，轨迹误差降低 40%、稳态偏差减少 30%、能耗降低 20%。
* **🤖 仿真到现实迁移：** 学习策略成功部署在物理双关节机器鱼上进行轨迹跟踪实验。

---
<!-- 
## 📁 项目结构

```
DyCFish-Gym/
├── README.md                          # 项目文档（英文）
├── README_zh.md                       # 项目文档（中文）
├── dynamic_stage/                     # 第一阶段：降阶模型预训练
│   ├── fish_env.py                    # 基于 Gymnasium 的 ROM 环境 (FishEnv)
│   ├── train.py                       # PPO 预训练脚本
│   └── test.py                        # 评估与视频录制
├── CFD_stage/                         # 第二阶段：CFD 微调
│   ├── EnvFluent.py                   # ANSYS Fluent Gymnasium 环境 (FluentEnv)
│   ├── training.py                    # 多进程 CFD 训练（支持断点续训）
│   └── lstm_policy.py                 # 自定义 LSTM 策略网络
└── assets/
    ├── framework_figure/              # 架构图
    │   └── Framework.png
    └── demos/                         # 演示视频
        ├── video1.mp4
        ├── video2.mp4
        └── video3.mp4
``` -->

---

## 📚 快速开始

### 前置条件

* 操作系统：Windows 或 Linux（推荐 Ubuntu 20.04+）
* NVIDIA GPU（可选，但推荐用于 PyTorch 训练）
* **ANSYS Fluent**（第二阶段需要，须安装并配置 `ansys-fluent-core`）
* Conda
* Python 3.9

### 安装

#### 1. Python 环境配置

```bash
# 创建并激活 Conda 环境
conda create -n dycfish python=3.9.13
conda activate dycfish

# 升级 pip
pip install --upgrade pip

# 安装核心库（深度学习 + 强化学习）
pip install numpy==2.0.2
pip install torch==2.1.0
pip install stable-baselines3[extra]

# 安装可视化和工具包
pip install pygame opencv-python matplotlib

# 安装 Pyfluent 包（CFD 阶段需要）
pip install ansys-fluent-core

# 调整 Pandas 版本（防止冲突）
pip uninstall pandas -y
pip install pandas==2.2.2
```

#### 2. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/DyCFish-Gym.git
cd DyCFish-Gym
```

#### 3. Pyfluent 配置（第二阶段需要）

`ansys-fluent-core` 包允许通过 Python 无缝控制 ANSYS Fluent。

* 详细文档和安装指南请参考官方仓库：[PyFluent](https://github.com/ansys/pyfluent)。

**通过 Python 控制 Fluent：**

| 操作 | 命令 | 说明 |
| :--- | :--- | :--- |
| **导入** | `import ansys.fluent.core as pyfluent` | 导入核心库 |
| **启动（无界面）** | `session = pyfluent.launch_fluent()` | 无图形界面启动 Fluent |
| **启动（有界面）** | `session = pyfluent.launch_fluent(show_gui=True)` | 带 GUI 启动 Fluent（仅网格模式） |
| **退出** | `session.exit()` | 关闭 Fluent 会话 |

**Pyfluent 日志录制（代码生成）：**

1. 启动 Fluent：`session = pyfluent.launch_fluent()`
2. 开始录制：`(api-start-python-journal "python_journal.py")`
3. 执行 TUI 命令 → Python 代码自动生成到 `python_journal.py`
4. 停止录制：`(api-stop-python-journal)`

---

## 🚀 使用方法

### 第一阶段：ROM 预训练

第一阶段使用降阶动力学环境进行快速策略学习。

**训练智能体：**

```bash
cd dynamic_stage
python train.py
```

运行后将：
- 启动 `FishEnv` 环境并进行实时 Pygame 渲染
- 训练 PPO 智能体，总步数 1,000,000
- 每 10,000 步保存模型检查点到 `./models/`
- 记录回合奖励到 `./logs/episode_rewards.txt`
- 支持 TensorBoard 可视化监控（`./tensorboard/`）

**监控训练过程：**

```bash
tensorboard --logdir ./tensorboard/
```

**评估训练模型：**

```bash
cd dynamic_stage
python test.py
```

加载训练好的模型，运行 5 个评估回合，并将结果保存为视频到 `./logs/test_escape.mp4`。

### 第二阶段：CFD 微调

第二阶段在高保真 ANSYS Fluent CFD 环境中微调预训练策略。

> ⚠️ **注意：** 运行此阶段前必须安装并正确配置 ANSYS Fluent。

**运行 CFD 训练：**

```bash
cd CFD_stage
python training.py
```

运行后将：
- 启动 ANSYS Fluent（双精度 2D 求解器，6 个处理器）
- 支持多进程训练和自动断点续训
- 保存模型并自动追踪局部/全局最优到 `./saved_models/`
- 记录每回合的详细性能指标

---

## 📦 基准测试与方法

### 平台架构

DyCFish-Gym 由四个紧密耦合的功能模块组成：

| 模块 | 说明 |
| :--- | :--- |
| **CPG 参数化模块** | 将生物运动学编码为低维驱动空间（振幅 *A*、频率 *f*） |
| **模型构建模块** | ROM 用于快速预训练 + CFD 用于高保真精调 |
| **DRL 控制模块** | 基于 PPO 的策略学习，支持稳定的跨模型迁移 |
| **策略仿生模块** | 行为评估：解码优化后的游泳策略，提供机制性洞察 |

### 环境参数

| 参数 | 动力学阶段 (`FishEnv`) | CFD 阶段 (`FluentEnv`) |
| :--- | :--- | :--- |
| **观测空间** | 7 维：[*x, y, ψ, vx, vy, ωz, t*] | 7 维：[*x, y, θ, vx, vy, ωz, t*] |
| **动作空间** | 2 维：[*振幅, 频率*] | 2 维：[*频率, 振幅*] |
| **物理引擎** | 降阶铰接体动力学 | 非定常 Navier–Stokes（Fluent） |
| **仿真速度** | >10³ 步/秒 | ~50 步/秒 |

### 基线对比

| 方法 | 轨迹跟踪 RMSE | 定点保持偏差 | 逃逸成功率 | 能耗 |
| :--- | :--- | :--- | :--- | :--- |
| PID | 1.00 | 1.00 | 0.82 | 1.00 |
| MPC | 0.78 | 0.83 | 0.87 | 0.92 |
| **DyCFish-Gym** | **0.60** | **0.68** | **0.96** | **0.78** |

<!-- ### 演示视频

<p align="center">
  <table>
    <tr>
      <td align="center"><b>轨迹跟踪</b></td>
      <td align="center"><b>捕食者逃逸</b></td>
      <td align="center"><b>定点保持</b></td>
    </tr>
    <tr>
      <td align="center"><a href="./assets/demos/video1.mp4">🎬 视频 1</a></td>
      <td align="center"><a href="./assets/demos/video2.mp4">🎬 视频 2</a></td>
      <td align="center"><a href="./assets/demos/video3.mp4">🎬 视频 3</a></td>
    </tr>
  </table>
</p> -->

---

## 📝 待办事项
- \[x\] 发布 CFD_stage 训练代码。
- \[x\] 发布 dynamic_stage 代码。
- \[\] 发布演示视频。
- \[ \] 发布论文。

---

<!-- ## 🔗 引用

如果您觉得本工作对您的研究有帮助，请考虑引用我们的论文：

```bibtex
@article{sun2025dycfishgym,
  title={DyCFish-Gym: An Intelligent Control Platform Bridging Reduced-Order Dynamics and Computational Fluid Dynamics for Thunniform Propulsion},
  author={Sun, Weiyuan and Zhan, Ruixin and Jiang, Haokui and Li, Xiaofan and Huang, Qingdong and Cao, Shunxiang},
  journal={International Journal of Mechanical Sciences},
  year={2025}
}
```

---

## 📄 许可证

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。

--- -->

## 👏 致谢

本研究得到了**国家重点研发计划**（项目编号：2024YFC3013200）的资助。

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO 算法实现
- [Gymnasium](https://gymnasium.farama.org/) — 强化学习环境接口
- [ANSYS PyFluent](https://github.com/ansys/pyfluent) — ANSYS Fluent Python 接口
- [PyTorch](https://pytorch.org/) — 深度学习框架
- [Pygame](https://www.pygame.org/) — 实时可视化
