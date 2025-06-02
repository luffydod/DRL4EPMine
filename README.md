# DRL4EPMine

本项目作为强化学习-机器人视觉导航大作业，主要使用PPO（近端策略优化）算法来训练智能体在仿真矿场环境中进行视觉探索任务，包含了训练、测试和行为克隆等多种功能模块，支持不同的策略网络架构和训练方法。

## 任务描述

在固定环境内，根据第一视角图像输入，找到指定目标。

状态：第一视角图像，尺寸为128x128

动作：机器人横向速度、纵向速度和旋转角速度

奖励：机器人到达指定位置会返回+10奖励，在`envs/SingleAgent/mine_toy.py`中设置了一种简易稠密奖励方式。

注意：机器人如果出生点位置异常，如何操作都会提前终止并返回-10的奖励，我们在实际训练中尝试了两种解决思路：将-10的奖励置为0，忽略这种负奖励；改写PPO的收集数据方法，将包含-10的轨迹直接移除，认为这种异常翻车轨迹对训练没影响。

## 运行说明

```bash

python train_ppo.py --help
# 命令行参数

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        训练设备 (auto, cpu, cuda)
  -a ACTION, --action ACTION
                        运行模式 (train, test, test_random,
                        test_keyboard, collect_expert_data)
  -mp MODEL_PATH, --model_path MODEL_PATH
                        模型加载或者保存路径
  -v, --video           是否保存视频 （将要弃用）
```

模型测试：

```bash
# 使用训练好的模型测试
python train_ppo.py -a test -mp models/your_model_path

# 随机动作测试
python train_ppo.py -a test_random

# 键盘控制测试
python train_ppo.py -a test_keyboard

```

行为克隆训练：

```bash
# 使用专家数据进行行为克隆
python bc_train.py --mode bc --expert_data_path expert_data/your_data.pkl

# 使用专家模型生成数据并进行行为克隆
python bc_train.py --mode bc_expert --expert_model_path models/expert_model.zip
```

键盘交互收集专家数据：

```bash
python train_ppo.py -a collect_expert_data
```

键盘控制说明：

- `w`：向前移动
- `s`：向后移动
- `a`：向左移动
- `d`：向右移动
- `q`：左转
- `e`：右转
- `space`：不动
- `ESC`：退出
- `r`：手动重置环境
- `y`: 保存当前回合的专家数据
- `n`: 不保存当前回合的专家数据

参数配置说明：
在`config.py`中可以修改各种训练参数：

- 环境相关配置：环境ID、环境数量、随机种子等
- PPO算法超参数：学习率、步数、批量大小、折扣因子等
- 训练相关配置：总步数、策略网络类型、日志路径等
- 模型保存相关：保存路径、保存频率等

## 环境配置

### pip依赖整理

```bash

stable-baselines3[extra]

mlagents-envs

sb3-contrib

```

### Windows端

```bash
pip install stable-baselines3[extra]
# 注意这个包安装后可能需要降级 protobuf==3.20.3
pip install mlagents-envs

```

### linux端配置

(1) opencv-python系统依赖缺失

```bash
# error
File "/workspace/drl_ep/envs/singleAgent/mine_toy.py", line 9, 
    in <module> import cv2 as cv 
    ImportError: libGL.so.1: cannot open shared object file: No such file or directory

# 解决
apt install libgl1
```

(2) 使用xvfb用于无头训练

```bash

apt update && apt install -y xvfb

# 运行方式
xvfb-run python3 train_ppo.py

```

```bash
# error
mlagents_envs.exception.UnityEnvironmentException: Error when trying to launch environment - make sure permissions are set correctly
# 解决
chmod -R 775 MineField_Linux-0510-random/drl.x86_64
```

### docker记录

```bash
# 基于 novnc镜像
docker pull crpi-c30rbdvbl28uwiva.cn-beijing.personal.cr.aliyuncs.com/luffydod/novnc:base

# 容器创建
docker run -itd \
  -p 50004:80 \
  --security-opt seccomp=unconfined \
  --shm-size=512m \
  --gpus all \
  -v /home/disk/sdb/one/zwb/workspace:/home/ubuntu/workspace \
  --name demo1 \
  novnc_torch:ep.mine

```

一些依赖配置问题，

```bash

# protobuf依赖降版本问题，原有操作似乎重装不干净

# 先手动卸载
rm -rf /usr/local/lib/python3.12/dist-packages/google/protobuf

# 重装
pip install --no-cache-dir --break-system-packages protobuf==3.20.3
```

运行指令，环境变量

```bash
# 修复 XDG_RUNTIME_DIR 错误
# 创建用户运行时目录（临时修复）
sudo mkdir -p /run/user/$(id -u)  # 需 root 权限创建

# 配置
export XDG_RUNTIME_DIR=/run/user/$(id -u)

# 设置目录所有者（替换为你的实际用户名） 
chown $(whoami):$(whoami) /run/user/$(id -u)

export XAUTHORITY=$HOME/.Xauthority

# 【已解决】 添加到 ~/.bashrc
```

### 关闭可视化界面

mlagents-envs提供了`no-graphics`仿真模式，但是在该模式下图像不会被正常渲染。
这里我们提供了一种通过修改mlagents-envs源码的方式，让它们支持不显示可视化窗口。
具体的，找到当前python环境的库安装路径，并找到`site-packages/mlagents_envs/environment.py`，将第272行
`args += ["-nographcis", "-batchmode"]` 修改为 `args += ["-batchmode"]`。

补充：实际测试发现错误。

![mlagents_error](docs/mlagents_e1.png)

然后再代码（`envs/SingleAgent/mine_toy.py`）中 `no_graph = True`。

需要注意的是，上述修改方式虽然支持关闭可视化窗口，但是在服务器（无显示）端仅修改上述代码而不适用docker的情况下，仍然不能正常渲染图像。

***警告***：上述代码涉及修改mlagents-envs源码，请谨慎使用。

补充(沉痛教训): `no_graph = True` 的情况下获取观测空间的图像数据会出现问题,也就是恒定的黑屏图片数据!

```bash
# 调试发现
obs_data min: 205, max: 205, mean: 205.0, std: 0.0
```

## 仿真环境下载

在[release](https://github.com/DRL-CASIA/EpMineEnv/releases)标签下，下载最新的系统对应的仿真环境，解压到`envs/SingleAgent/`路径下，并检查`envs/SingleAgent/mine_toy.py`中的`file_name`路径是否正确。
