"""
可视化脚本：展示球体变形为奶牛的过程（改进版）
读取 output_meshes/ 中的中间模型文件，创建交互式动画
使用 Mesh3d 显示完整三角网格，支持材质和光照
"""

import os
import torch
import numpy as np
import plotly.graph_objects as go
from pytorch3d.io import load_objs_as_meshes
import webbrowser

# ============================================================
# 1. 配置
# ============================================================
OUTPUT_DIR = "output_meshes"
ANIMATION_FILE = os.path.join(OUTPUT_DIR, "deformation_animation.html")

# 检查输出目录是否存在
if not os.path.exists(OUTPUT_DIR):
    raise FileNotFoundError(f"找不到 {OUTPUT_DIR} 目录！请先运行训练脚本。")

# ============================================================
# 2. 读取所有中间模型
# ============================================================
print("正在扫描模型文件...")

mesh_files = []
for filename in sorted(os.listdir(OUTPUT_DIR)):
    if filename.startswith("mesh_epoch_") and filename.endswith(".obj"):
        epoch = int(filename.replace("mesh_epoch_", "").replace(".obj", ""))
        mesh_files.append((epoch, os.path.join(OUTPUT_DIR, filename)))

mesh_files.sort(key=lambda x: x[0])

if not mesh_files:
    raise FileNotFoundError(f"在 {OUTPUT_DIR} 中没有找到 mesh_epoch_*.obj 文件！")

print(f"找到 {len(mesh_files)} 个中间模型文件")
print(f"范围: Epoch {mesh_files[0][0]} -> Epoch {mesh_files[-1][0]}")

# 如果文件太多，只选择关键帧
MAX_FRAMES = 15
if len(mesh_files) > MAX_FRAMES:
    indices = np.linspace(0, len(mesh_files) - 1, MAX_FRAMES, dtype=int)
    mesh_files = [mesh_files[i] for i in indices]
    print(f"已采样到 {MAX_FRAMES} 个关键帧")

# ============================================================
# 3. 读取模型顶点和面
# ============================================================
print("正在加载模型数据...")

frames_data = []

for epoch, file_path in mesh_files:
    mesh = load_objs_as_meshes([file_path], device=torch.device("cpu"))
    verts = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()

    # 计算顶点法线用于光照
    triangles = verts[faces]
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8

    # 计算顶点法线（相邻面法线的平均）
    vertex_normals = np.zeros_like(verts)
    for f_idx, face in enumerate(faces):
        for v_idx in face:
            vertex_normals[v_idx] += face_normals[f_idx]
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-8

    # 简单光照强度
    light_dir = np.array([-0.5, -0.7, 0.8])
    light_dir /= np.linalg.norm(light_dir)
    intensity = 0.3 + 0.7 * np.clip(np.dot(vertex_normals, light_dir), 0, 1)
    vertex_colors = np.column_stack([intensity * 0.7, intensity * 0.8, intensity * 1.0])

    frames_data.append({
        'epoch': epoch,
        'verts': verts,
        'faces': faces,
        'file_path': file_path,
        'vertex_colors': vertex_colors
    })
    print(f"  已加载: Epoch {epoch:03d} ({len(verts)} 顶点, {len(faces)} 面)")

# ============================================================
# 4. 创建 Plotly 交互式动画
# ============================================================
print("正在创建交互式动画...")

# 计算所有模型的边界
all_verts = np.concatenate([d['verts'] for d in frames_data], axis=0)
x_min, x_max = all_verts[:, 0].min() - 0.2, all_verts[:, 0].max() + 0.2
y_min, y_max = all_verts[:, 1].min() - 0.2, all_verts[:, 1].max() + 0.2
z_min, z_max = all_verts[:, 2].min() - 0.2, all_verts[:, 2].max() + 0.2

# 创建帧
frames = []
for data in frames_data:
    frame = go.Frame(
        data=[
            go.Mesh3d(
                x=data['verts'][:, 0],
                y=data['verts'][:, 1],
                z=data['verts'][:, 2],
                i=data['faces'][:, 0],
                j=data['faces'][:, 1],
                k=data['faces'][:, 2],

                # 顶点颜色（带光照效果）
                vertexcolor=data['vertex_colors'],

                # 材质属性
                opacity=1.0,
                flatshading=True,  # 平面着色，面片更明显

                # 光照设置
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.9,
                    specular=0.3,
                    roughness=0.7,
                    fresnel=0.1
                ),

                # 颜色和外观
                color='lightblue',
                colorscale='Blues',
                intensity=None,

                # 线框设置
                showscale=False,
                name=f"Epoch {data['epoch']}",
                hoverinfo='none'
            )
        ],
        name=f"epoch_{data['epoch']}",
        layout=go.Layout(
            title=dict(
                text=f"球体变形为奶牛 - Epoch {data['epoch']}",
                font=dict(size=18, color='#333333')
            )
        )
    )
    frames.append(frame)

# 初始状态
first_data = frames_data[0]
fig = go.Figure(
    data=[
        go.Mesh3d(
            x=first_data['verts'][:, 0],
            y=first_data['verts'][:, 1],
            z=first_data['verts'][:, 2],
            i=first_data['faces'][:, 0],
            j=first_data['faces'][:, 1],
            k=first_data['faces'][:, 2],

            # 顶点颜色
            vertexcolor=first_data['vertex_colors'],

            # 材质
            opacity=1.0,
            flatshading=True,

            # 光照
            lighting=dict(
                ambient=0.4,
                diffuse=0.9,
                specular=0.3,
                roughness=0.7,
                fresnel=0.1
            ),

            color='lightblue',
            colorscale='Blues',
            intensity=None,
            showscale=False,
            name=f"Epoch {first_data['epoch']}",
            hoverinfo='none'
        )
    ],
    frames=frames
)

# 配置布局
fig.update_layout(
    title=dict(
        text="球体变形为奶牛的过程（可拖拽旋转）",
        font=dict(size=22, color='#222222'),
        x=0.5,
        xanchor='center'
    ),
    scene=dict(
        xaxis=dict(
            range=[x_min, x_max],
            autorange=False,
            title='X',
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[y_min, y_max],
            autorange=False,
            title='Y',
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        zaxis=dict(
            range=[z_min, z_max],
            autorange=False,
            title='Z',
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        aspectmode='cube',
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.8),
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0)
        ),
        bgcolor='#f0f0f0'
    ),
    showlegend=False,
    updatemenus=[
        {
            "type": "buttons",
            "buttons": [
                {
                    "label": "▶ 播放动画",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 500, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 300, "easing": "quadratic-in-out"}
                    }],
                },
                {
                    "label": "⏸ 暂停",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }],
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
            "bgcolor": 'rgba(255,255,255,0.8)',
            "bordercolor": '#cccccc',
            "borderwidth": 1
        }
    ],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 16, "color": '#666666'},
            "prefix": "Epoch: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "bgcolor": 'rgba(255,255,255,0.8)',
        "bordercolor": '#cccccc',
        "borderwidth": 1,
        "steps": [
            {
                "args": [
                    [f"epoch_{data['epoch']}"],
                    {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}
                    }
                ],
                "label": f"{data['epoch']}",
                "method": "animate"
            }
            for data in frames_data
        ]
    }],
    margin=dict(l=10, r=10, t=60, b=10),
    template="plotly_white",
    paper_bgcolor='white'
)

# ============================================================
# 5. 保存动画
# ============================================================
print(f"正在保存动画到: {ANIMATION_FILE}")
fig.write_html(ANIMATION_FILE)
print(f"✅ 动画已保存！")

# ============================================================
# 6. 自动在浏览器中打开
# ============================================================
try:
    webbrowser.open(ANIMATION_FILE)
    print(f"🌐 正在浏览器中打开动画...")
except:
    print(f"请手动打开: {ANIMATION_FILE}")

print("\n" + "=" * 60)
print("使用说明：")
print("- 🖱️ 左键拖拽：旋转视角")
print("- 🔍 滚轮：缩放")
print("- ↔️ 右键拖拽：平移")
print("- ▶️ 点击播放按钮：自动播放动画")
print("- 🎚️ 拖动滑块：跳转到指定步骤")
print("=" * 60)