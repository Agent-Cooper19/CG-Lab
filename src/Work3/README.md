202411998349 孙多艺 人工智能
# Phong 光照模型

## 实验概述

基于 Taichi 框架实现 Phong 光照模型和 Blinn-Phong 光照模型，通过光线投射（Ray Casting）隐式定义球体和圆锥，实现局部光照渲染，并对比两种高光模型和硬阴影的视觉效果。

## 文件结构

```
Work3/
├── Phong.py                    # 基础 Phong 模型（红球+紫色圆锥）
├── Blinn-Phong.py              # Blinn-Phong + 硬阴影（红球+紫色圆锥）
├── new-Blinn.py                # Blinn-Phong + 硬阴影（红球+蓝色圆锥+地面）
├── Blinn-Phong与硬阴影.gif       # new-Blinn.py的运行结果
└── 基础Phong模型.gif             # Phong.py的运行结果
```

## 环境配置

```bash
pip install taichi
```

## 运行方法

```bash
# 基础 Phong 模型
python Phong.py

# Blinn-Phong + 阴影
python Blinn-Phong.py

# Blinn-Phong + 阴影 + 地面
python new-Blinn.py
```

GUI 面板说明：
- **Ka/ Kd/ Ks 滑块**：调节环境光、漫反射、镜面高光系数
- **Shininess 滑块**：调节高光指数（1~128）
- **Use Blinn-Phong 复选框**：切换 Phong / Blinn-Phong 高光模型
- **Enable Hard Shadow 复选框**：开关硬阴影

## 已实现的功能

### 1. 场景搭建（纯数学定义，无外部模型）

- 红色球体：圆心 (-1.2, -0.2, 0)，半径 1.2
- 紫色/蓝色圆锥：顶点 (1.2, 1.2, 0)，底面 y=-1.4，底面半径 1.2
- 灰色地平面（new-Blinn.py）：y=-1.5
- 相机 (0, 0, 5)，点光源 (2, 3, 4)

### 2. 光线投射与深度测试

- 每个像素发射一条射线
- 对场景中所有物体求交，取最近的交点（最小正 t 值）
- 正确实现遮挡关系

### 3. Phong 光照模型

三个分量独立计算后叠加：
```
I = I_ambient + I_diffuse + I_specular
```

- 环境光：`Ka × light_color × object_color`
- 漫反射：`Kd × max(0, N·L) × light_color × object_color`
- 镜面高光：`Ks × max(0, R·V)^n × light_color`

### 4. Blinn-Phong 升级（选做）

- 使用半程向量 H = normalize(L + V) 替代反射向量 R
- 高光计算：`max(0, N·H)^n`
- 可实时切换两种模型，对比高光区域差异

### 5. 硬阴影（选做）

- 从交点向光源发射阴影射线
- 检测路径上是否有其他物体遮挡
- 被遮挡区域仅保留环境光
- 球体在地面投射阴影，圆锥在球体和地面投射阴影

### 6. GUI 交互面板

- 材质参数实时调节（Ka, Kd, Ks, Shininess）
- Phong / Blinn-Phong 一键切换
- 阴影开关
- 当前模式状态显示

## Phong vs Blinn-Phong 对比

| 特性 | Phong | Blinn-Phong |
|------|-------|-------------|
| 高光计算 | R·V | N·H |
| 计算量 | 较大（需计算反射向量） | 较小（半程向量计算简单） |
| 大入射角表现 | 高光形状拉长 | 高光更圆润自然 |
| 物理准确性 | 不符合 Helmholtz 互反律 | 更接近真实材质 |

## 实验效果

- 球体：红色表面，左侧受光，高光集中在朝向光源的区域
- 圆锥：紫色/蓝色表面，曲面法线渐变，高光呈带状分布
- 阴影：球体和圆锥在地面投射清晰硬阴影，遮挡关系正确
- 高光指数：n 越小高光越分散，n 越大高光越集中锐利

## 关键实现细节

光线-圆锥求交：
- 将光线转换到圆锥顶点为原点的局部坐标系
- 构建一元二次方程 A·t² + B·t + C = 0
- 验证交点是否在圆锥高度范围内
- 法线公式：normalize(p_local.x, -k·p_local.y, p_local.z)

阴影射线偏移：
- 起点沿法线偏移 0.001~0.002 防止自交

## 参考资料

- Phong, "Illumination for Computer Generated Pictures", CACM 1975
- Blinn, "Models of Light Reflection for Computer Synthesized Pictures", SIGGRAPH 1977
- Taichi 文档: https://docs.taichi-lang.org/
