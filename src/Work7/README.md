202411998349 孙多艺 人工智能
# SMPL LBS 蒙皮过程可视化

## 实验概述

基于 SMPL 模型完成 LBS (Linear Blend Skinning) 蒙皮过程可视化，手写实现四个核心阶段并验证与官方结果的一致性。同时完成了选做内容，实现了姿态动画，在outputs_animation中。

## 文件结构

```
Work7/
├── lbs_visualization.py      # 主实验：LBS 四阶段可视化
├── lbs_animation.py          # 选做：姿态动画
├── models/
│   └── smpl/
│       └── SMPL_NEUTRAL.pkl # 模型文件（需自行下载）
├── outputs/                  # 可视化结果
│   ├── stage_a_template_weights.png  # 模板网格 + 蒙皮权重
│   ├── stage_b_shaped_joints.png     # 形状校正 + 关节回归
│   ├── stage_c_pose_offsets.png      # 姿态校正
│   ├── stage_d_lbs_result.png        # 最终蒙皮结果
│   ├── comparison_grid.png           # 四阶段对比图
│   ├── all_joint_weights.png         # 全关节权重分布
│   └── summary.txt                   # 误差报告
└── outputs_animation/        # 动画输出（选做）
    ├── anim_left_elbow.gif
    ├── anim_right_shoulder.gif
    └── anim_left_knee.gif
```

## 环境配置

```bash
pip install smplx torch numpy matplotlib
```

从 [SMPL 官网](https://smpl.is.tue.mpg.de/) 下载 `SMPL_NEUTRAL.pkl`，放入 `models/smpl/` 目录。

## 运行方法

主实验：
```bash
python lbs_visualization.py --model-dir ./models --out-dir ./outputs --joint-id 18
```

参数说明：
- `--model-dir`：模型目录（默认 `./models`）
- `--out-dir`：输出目录（默认 `./outputs`）
- `--joint-id`：要可视化权重的关节编号（默认 18，左肘）
- `--num-betas`：shape 参数数量（默认 10）

选做动画：
```bash
python lbs_animation.py
```

## LBS 四个阶段

**阶段 (a)：模板网格与蒙皮权重**

展示 T-pose 模板网格，热力图显示指定关节对各顶点的影响权重。颜色越深，权重越大。

**阶段 (b)：形状校正与关节回归**

应用 shape 参数（betas）改变体型，从校正后的网格回归关节位置。关节点随体型变化而调整。

**阶段 (c)：姿态校正**

将姿态参数转为旋转矩阵，通过 `pose_feature = R - I` 和 `posedirs` 计算姿态偏移量。颜色表示偏移量大小，集中在弯曲部位（肘、膝等）。

**阶段 (d)：最终蒙皮结果**

根据运动学树计算全局刚体变换，用蒙皮权重加权平均得到最终顶点位置。

## 五个核心变量

| 变量 | 含义 |
|------|------|
| `v_template` | 模板顶点（T-pose） |
| `v_shaped` | 加了形状形变后的顶点 |
| `J` | 由 v_shaped 回归出的关节 |
| `v_posed` | 加了姿态校正后的顶点 |
| `verts` | 完成 LBS 之后的最终顶点 |

## 验证结果

手写 LBS 与官方 `model.forward()` 对比：
- 平均绝对误差 (MAE) < 1e-6
- 最大绝对误差 (Max AE) < 1e-5

误差极小，验证了手写实现的正确性。

## 选做：姿态动画

固定 shape 参数，让指定关节从 0° 逐渐旋转到目标角度，生成 GIF。支持的关节：`left_elbow`、`right_shoulder`、`left_knee` 等。

动画中红色球高亮旋转关节，热力图显示权重分布，可观察权重区域如何随骨骼运动被平滑带动。

## 参考资料

- SMPL 论文: Loper et al., "SMPL: A Skinned Multi-Person Linear Model", SIGGRAPH Asia 2015
- SMPL 官网: https://smpl.is.tue.mpg.de/
- SMPL-X GitHub: https://github.com/vchoutas/smplx
