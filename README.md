# DRL4EPMine

## 任务描述

在固定环境内，根据第一视角图像输入，找到指定目标。

状态：第一视角图像，尺寸为128x128

动作：机器人横向速度、纵向速度和旋转角速度

奖励：机器人到达指定位置会返回+10奖励，在`envs/SingleAgent/mine_toy.py`中设置了一种简易稠密奖励方式。

对环境的测试可以参考`envs/SingleAgent/mine_toy.py`文件。

## 环境配置

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

(2)

```bash
# error
mlagents_envs.exception.UnityEnvironmentException: Error when trying to launch environment - make sure permissions are set correctly
# 解决
chmod -R 775 MineField_Linux-0510-random/drl.x86_64
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

## 仿真环境下载

在[release](https://github.com/DRL-CASIA/EpMineEnv/releases)标签下，下载最新的系统对应的仿真环境，解压到`envs/SingleAgent/`路径下，并检查`envs/SingleAgent/mine_toy.py`中的`file_name`路径是否正确。
