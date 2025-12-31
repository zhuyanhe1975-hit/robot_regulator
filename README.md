# robot_regulator

工业机械臂（6轴）MuJoCo 仿真 + 力矩前馈 PID +（后续）神经网络变增益控制器。

## 环境

本仓库默认使用你已有的虚拟环境：

`/home/yhzhu/Industrial Robot/robot_regulator/env`

验证：

```bash
env/bin/python -c "import mujoco; print(mujoco.__version__)"
```

## 第1阶段：先跑起来（可视化仿真）

运行内置的 6-DoF 示例模型（MJCF）：

```bash
env/bin/python -m scripts.run_viewer --model models/six_dof.xml
```

如果你有自己的 URDF：

```bash
env/bin/python -m scripts.run_viewer --model /path/to/robot.urdf
```

（若你的 MuJoCo 版本/URDF 特性不兼容，会在终端提示加载失败；此时我们再做 URDF->MJCF 的转换流程。）

### ABB IRB2400（本仓库已有）

```bash
env/bin/python -m scripts.run_viewer --model models/abb_irb2400/urdf/irb2400.urdf
```

说明：
- `irb2400.urdf` 的 visual mesh 是 `.dae`，MuJoCo 不直接支持，程序会调用本机 `blender` 自动转成 `.stl` 并缓存到 `.cache/mujoco/`。

## 强化学习（mjlab，路线1：变增益 PD）

本仓库提供了一个 `mjlab` 任务插件（不再 vendoring mjlab 源码）：让 policy 输出每个关节的 `(kp,kd)`，用 mjlab 的 `IdealPdActuator` 在 GPU 上并行仿真训练。

- 任务 ID：`Mjlab-JointGain-ABB-IRB2400`
- 机器人 MJCF：`models/abb_irb2400/mjcf/irb2400_mjlab.xml`（引用已有 collision STL，默认关闭接触）

运行训练（前提：你已经 `pip install -e ~/AI/mjlab/ --no-deps` 并安装好 mujoco-warp/torch 等依赖）：

```bash
python -m scripts.mjlab_train Mjlab-JointGain-ABB-IRB2400 --env.scene.num-envs 4096
```

## 强化学习（mjlab，回到稳定基线：固定 PD + ID + 残差 FF）

先回到一个更稳定、更容易收敛的基线：PD 增益固定，使用逆动力学力矩前馈（ID），策略只学习残差力矩（FF，按摩擦模型参数化）。

- 任务 ID：`Mjlab-PDIDFF-ABB-IRB2400`
- 控制/环境步长：20ms（物理 2ms：`timestep=0.002`，`decimation=10`；ID/FF 每 20ms 更新一次）

运行训练：

```bash
python -m scripts.mjlab_train Mjlab-PDIDFF-ABB-IRB2400 --env.scene.num-envs 4096
```

### 去掉 ID 的对照组（固定 PD + 残差 FF）

用于验证“问题是否来自逆动力学 ID 项”：

```bash
python -m scripts.mjlab_train Mjlab-PDFF-ABB-IRB2400 --env.scene.num-envs 4096
```

### 粗物理步长对照组（同控制器，仅 timestep=20ms）

用于量化 `timestep` 变大带来的数值阻尼/低通效应（physics 20ms，decimation=1）：

```bash
python -m scripts.mjlab_train Mjlab-PDFF-CoarseDt-ABB-IRB2400 --env.scene.num-envs 4096
```

### TensorBoard

训练日志会写到 `logs/rsl_rl/<experiment_name>/<run_timestamp>/`，用 TensorBoard 查看：

```bash
conda activate mjwarp_env
tensorboard --logdir "logs/rsl_rl" --port 6006
```

## 复现（封存版 three-way 对比）

如果你想回到我们之前“固定增益 PD vs 变增益 PD vs 变增益+FF”那次自洽对比：

- 快照：`reproduce/v1_irb2400_threeway.json`
- 运行对比（会打印一条完整命令）：`python -m scripts.reproduce_v1_threeway`

### 复现（冻结任务 ID，不受后续实验改动影响）

为防止后续改动 `mjlab_irb2400` 影响旧 checkpoint 的可加载性，我们提供了 `mjlab_irb2400_v1` 插件和冻结 task id：

- 快照：`reproduce/v1_frozen_irb2400_threeway.json`
- 运行：`python -m scripts.reproduce_v1_frozen_threeway`

### 对比 Demo：变增益 vs 固定增益

在训练日志目录里选一个 checkpoint（例如 `model_999.pt`），用下面脚本做同任务同随机种子的 A/B 对比：

```bash
python -m scripts.mjlab_compare \
  --checkpoint logs/rsl_rl/irb2400_joint_gain/2025-12-26_22-00-23/model_999.pt \
  --num-envs 1024 \
  --episodes 5 \
  --out compare/irb2400_gain_compare.json
```

多 seed 汇总：

```bash
python -m scripts.mjlab_compare \
  --checkpoint logs/rsl_rl/irb2400_joint_gain/2025-12-26_22-00-23/model_999.pt \
  --seeds 0,1,2,3,4 \
  --num-envs 1024 \
  --episodes 5 \
  --out compare/irb2400_gain_compare.json
```

录制视频（单环境）：加 `--video`，会在 `compare/videos/baseline/` 和 `compare/videos/trained/` 输出 mp4。

对比输出中还会额外包含“真实关节跟踪误差”（单位 rad）：
- `JointError/mae_mean_rad`：全关节平均绝对误差（mean over envs & time）
- `JointError/rmse_mean_rad`：全关节 RMSE（mean over envs & time）
- `JointError/per_joint_*_rad`：每个关节的误差统计（list）

## 目录结构

- `models/`：示例模型（先用 MJCF，后续可替换为你的 URDF）
- `robot_regulator/`：仿真与控制代码（PID/NN 在这里扩展）
- `scripts/`：可运行入口（viewer、训练脚本等）
