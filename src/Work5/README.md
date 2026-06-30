202411998349 孙多艺 人工智能
# 可微渲染

## 实验概述

基于 PyTorch3D 实现可微光栅化（Differentiable Rendering），通过软剪影拟合将球体网格逐步变形为目标形状（奶牛）。使用梯度下降优化顶点位置，并加入拉普拉斯平滑、边长一致性和法线一致性三种正则化防止网格退化。

## 文件结构

```
Work5/
├── differentiable_rendering.py    # 主程序（训练 + 可视化）
├── visualize_deformation.py       # 可视化脚本（读取结果生成动画）
├── cow.obj                        # 目标奶牛模型
├── output_meshes/                 # 训练输出
│   ├── mesh_epoch_000.obj        # 中间模型
│   ├── mesh_epoch_020.obj
│   ├── ...
│   ├── final_mesh.obj            # 最终结果
│   ├── loss_curves.png           # 损失曲线
│   ├── comparison_epoch_*.png    # 每20轮的剪影对比
│   └── deformation_animation.html # 交互式变形动画
├── output_textured/              # 纹理优化尝试（未完成）
├── 形变过程-正面.gif               # 形变动画
├── 形变过程-侧面.gif               # 形变动画
└── 形变过程-背面.gif               # 形变动画
```

## 环境配置

```bash
pip install torch pytorch3d numpy matplotlib plotly
```

## 运行方法

### 训练

```bash
python differentiable_rendering.py
```

注意：`cow.obj` 需要放在代码同级目录下。如果找不到目标模型，会自动使用环面作为测试目标。

### 可视化变形过程

训练完成后，代码末尾的可视化部分会自动执行，生成 `deformation_animation.html`，可在浏览器中交互式查看变形过程（拖拽旋转、播放动画、拖动滑块）。

## 已实现的功能

### 1. 软光栅化形状优化

- 使用 SoftSilhouetteShader 实现可微剪影渲染
- 20 个视角均匀分布在 360 度范围内
- 从细分球体（ico_sphere level=4，2562 顶点）开始优化
- SGD 优化器 + 余弦退火学习率调度
- 训练 300 轮，每 20 轮保存中间模型

### 2. 三种正则化损失

| 正则化 | 权重 | 作用 |
|--------|------|------|
| 拉普拉斯平滑 | 1.0 | 防止表面出现尖锐突起 |
| 边长一致性 | 0.1 | 防止三角形严重拉伸 |
| 法线一致性 | 0.01 | 保持相邻面法线接近 |

### 3. 训练监控

- 控制台实时输出损失值
- 每 20 轮保存剪影对比图（目标 vs 预测 vs 差异图）
- 保存损失曲线（总损失 + 各分量）

### 4. 交互式变形动画

- 使用 Plotly 生成 HTML 动画
- 支持拖拽旋转、缩放、平移
- 可播放/暂停、拖动滑块跳转到任意步骤
- 顶点法线计算 + 简单光照着色，面片结构清晰可见

### 5. 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_([deform_verts], max_norm=10.0)
```
防止梯度爆炸导致顶点突然移动过远。

## 未完成的内容

### 1. 纹理/颜色优化（选做）

代码中曾尝试加入 SoftPhongShader 进行 RGB 颜色拟合，但遇到两个问题：

- `TexturesVertex` 导入错误：正确导入路径是 `from pytorch3d.renderer import TexturesVertex`，而非 `from pytorch3d.structures`
- 设备不一致错误：`SoftPhongShader` 内部的 materials 张量与光栅化结果在不同设备上（CPU vs GPU），在部分 PyTorch3D 版本中无法解决

最终只实现了基于剪影的形状优化，没有完成纹理/顶点颜色优化。

**改进方向**：
- 使用 PyTorch3D 更新版本
- 改用 SoftGouraudShader 替代 SoftPhongShader
- 或者只用剪影优化形状，颜色通过后处理方式赋予

### 2. 网格拓扑保持

虽然加入了三种正则化，但极端视角下仍可能出现局部面片翻转或自交。正则化权重是手动调参，没有自适应机制。

### 3. 收敛速度

300 轮训练在 CPU 上需要较长时间，且后期收敛缓慢。可以考虑：
- 增加学习率或使用 Adam 优化器
- 渐进式增加顶点数（从低面数到高面数）
- 多分辨率策略

## 实验效果

- 球体逐渐变形，大致拟合目标形状
- 凸起部分（如牛角、四肢）拟合较好
- 凹陷部分（如腹部）拟合较困难，需要更多视角或更长训练
- 正则化有效防止了网格退化成尖刺

## 参考资料

- Liu et al., "Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning", ICCV 2019
- PyTorch3D 文档: https://pytorch3d.org/
- 可微渲染教程: https://pytorch3d.org/tutorials/
