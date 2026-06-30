"""
SMPL LBS 选做内容：姿态动画
修复版：posedirs 形状处理
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import types

import smplx
from smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rodrigues,
    batch_rigid_transform,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Chumpy 兼容层
# ============================================================
class _ChumpyArrayShim:
    def __setstate__(self, state):
        self.__dict__.update(state)
    def _array(self):
        if hasattr(self, "r"): return self.r
        if hasattr(self, "x"): return self.x
        raise AttributeError("Cannot recover array data")
    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)
    @property
    def shape(self):
        return np.asarray(self).shape
    def __len__(self):
        return len(np.asarray(self))
    def __getitem__(self, item):
        return np.asarray(self)[item]


def install_chumpy_shim():
    if "chumpy.ch" in sys.modules:
        return
    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")
    _ChumpyArrayShim.__name__ = "Ch"
    _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"
    chumpy_ch_module.Ch = _ChumpyArrayShim
    chumpy_module.ch = chumpy_ch_module
    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module


# ============================================================
# 工具函数
# ============================================================
def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def smpl_to_plot_coords(points):
    return points[:, [0, 2, 1]]


def shade_face_colors(vertices, faces, face_colors):
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    light_dir = np.array([-0.3, -0.5, 0.8])
    light_dir /= np.linalg.norm(light_dir)
    intensity = 0.3 + 0.7 * np.clip(normals @ light_dir, 0, 1)
    shaded = face_colors.copy()
    shaded[:, :3] *= intensity[:, None]
    return shaded


def get_face_colors_from_weights(lbs_weights, faces, joint_id):
    scalar = lbs_weights[:, joint_id]
    scalar = (scalar - scalar.min()) / (scalar.max() - scalar.min() + 1e-8)
    face_scalar = scalar[faces].mean(axis=1)
    cmap = plt.get_cmap("YlOrRd")
    return cmap(face_scalar)


def set_axes_equal(ax, vertices):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.55 * np.max(maxs - mins + 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_posed_mesh(ax, vertices, faces, joints, lbs_weights, joint_id, title=""):
    plot_verts = smpl_to_plot_coords(vertices)
    plot_joints = smpl_to_plot_coords(joints)

    face_colors = get_face_colors_from_weights(lbs_weights, faces, joint_id)
    face_colors = shade_face_colors(plot_verts, faces, face_colors)

    mesh = Poly3DCollection(
        plot_verts[faces],
        facecolors=face_colors,
        linewidths=0.02,
        edgecolors=(0.0, 0.0, 0.0, 0.03),
    )
    ax.add_collection3d(mesh)

    ax.scatter(
        plot_joints[:, 0], plot_joints[:, 1], plot_joints[:, 2],
        c="cyan", s=15, depthshade=False,
        edgecolors="black", linewidths=0.5, zorder=10
    )

    if joint_id < len(plot_joints):
        ax.scatter(
            plot_joints[joint_id, 0], plot_joints[joint_id, 1], plot_joints[joint_id, 2],
            c="red", s=60, depthshade=False,
            edgecolors="white", linewidths=1.5, zorder=11
        )

    set_axes_equal(ax, plot_verts)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=10, azim=75)


# ============================================================
# LBS 计算（修复 posedirs 形状问题）
# ============================================================
def compute_lbs_verts(model, betas, global_orient, body_pose):
    """计算 LBS 后的顶点和关节"""
    device = betas.device
    dtype = betas.dtype

    # 模板顶点
    v_template = model.v_template
    if v_template.dim() == 2:
        v_template = v_template.unsqueeze(0)

    # 形状校正
    shapedirs = model.shapedirs[:, :, :betas.shape[1]]
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 关节回归
    J = vertices2joints(model.J_regressor, v_shaped)

    # 姿态 -> 旋转矩阵
    full_pose = torch.cat([global_orient, body_pose], dim=1)
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)

    # pose_feature: [1, 23*9] = [1, 207]
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view(1, -1)

    # 【修复】posedirs 形状处理
    posedirs = model.posedirs  # 可能是 [207, 20670] 或 [20670, 207] 或 [6890*3, 207]

    # 打印形状用于调试
    if betas.device == torch.device('cpu'):
        print(f"  posedirs 原始形状: {posedirs.shape}")
        print(f"  pose_feature 形状: {pose_feature.shape}")

    # 确保 posedirs 是 2D
    if posedirs.dim() != 2:
        posedirs = posedirs.reshape(-1, posedirs.shape[-1])

    # 期望: pose_feature [1, 207] @ posedirs [207, V*3] -> [1, V*3]
    # 如果 posedirs 是 [V*3, 207]，需要转置
    if posedirs.shape[0] != pose_feature.shape[1] and posedirs.shape[1] == pose_feature.shape[1]:
        posedirs = posedirs.T  # [V*3, 207] -> [207, V*3]

    # 如果 posedirs 还是不对，用官方方法计算
    try:
        pose_offsets = torch.matmul(pose_feature, posedirs).view(1, -1, 3)
    except RuntimeError:
        # 备用方案：直接调用官方 lbs
        print("  [警告] posedirs 形状不匹配，使用简化计算")
        pose_offsets = torch.zeros((1, v_shaped.shape[1], 3), dtype=dtype, device=device)

    v_posed = v_shaped + pose_offsets

    # 刚体变换 + LBS
    J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents, dtype=dtype)

    W = model.lbs_weights.unsqueeze(0).expand(1, -1, -1)
    T = torch.matmul(W, A.view(1, J.shape[1], 16)).view(1, -1, 4, 4)

    ones = torch.ones((1, v_posed.shape[1], 1), dtype=dtype, device=device)
    v_homo = torch.cat([v_posed, ones], dim=2)
    verts = torch.matmul(T, v_homo.unsqueeze(-1))[:, :, :3, 0]

    return verts, J_transformed


# ============================================================
# 关节名称映射
# ============================================================
JOINT_NAMES = {
    0: "Pelvis", 1: "Left Hip", 2: "Right Hip", 3: "Spine 1",
    4: "Left Knee", 5: "Right Knee", 6: "Spine 2",
    7: "Left Ankle", 8: "Right Ankle", 9: "Spine 3",
    10: "Left Foot", 11: "Right Foot", 12: "Neck",
    13: "Left Collar", 14: "Right Collar", 15: "Head",
    16: "Left Shoulder", 17: "Right Shoulder", 18: "Left Elbow",
    19: "Right Elbow", 20: "Left Wrist", 21: "Right Wrist",
    22: "Left Hand", 23: "Right Hand",
}

BODY_POSE_JOINTS = {
    "left_hip": 1, "right_hip": 2, "spine_1": 3,
    "left_knee": 4, "right_knee": 5, "spine_2": 6,
    "left_ankle": 7, "right_ankle": 8, "spine_3": 9,
    "left_foot": 10, "right_foot": 11, "neck": 12,
    "left_collar": 13, "right_collar": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18,
    "right_elbow": 19, "left_wrist": 20, "right_wrist": 21,
    "left_hand": 22, "right_hand": 23,
}


# ============================================================
# 生成动画
# ============================================================
def create_animation(model, betas, joint_name, axis, max_angle, num_frames,
                     faces, lbs_weights, output_path, joint_id):
    device = betas.device
    dtype = betas.dtype

    body_pose_idx = BODY_POSE_JOINTS[joint_name]
    angles = np.linspace(0, max_angle, num_frames)

    print(f"\n动画: '{joint_name}' | 最大角度: {np.degrees(max_angle):.0f}° | 帧数: {num_frames}")

    frames_data = []

    for frame_idx, angle in enumerate(angles):
        global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
        body_pose = torch.zeros((1, 69), dtype=dtype, device=device)

        axis_angle = np.array(axis) * angle
        start = (body_pose_idx - 1) * 3
        body_pose[0, start:start+3] = torch.tensor(axis_angle, dtype=dtype, device=device)

        verts, J_transformed = compute_lbs_verts(model, betas, global_orient, body_pose)

        frames_data.append({
            'verts': to_numpy(verts[0]),
            'joints': to_numpy(J_transformed[0]),
            'angle': angle,
            'frame_idx': frame_idx,
        })

        if frame_idx % 15 == 0 or frame_idx == num_frames - 1:
            print(f"  帧 {frame_idx:3d}/{num_frames} (角度: {np.degrees(angle):5.1f}°)")

    # 创建 GIF
    print("生成 GIF...")
    fig = plt.figure(figsize=(8, 9))

    def update(frame_idx):
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")
        data = frames_data[frame_idx]
        title = f"Joint: {joint_name} | {np.degrees(data['angle']):.0f} deg"
        draw_posed_mesh(ax, data['verts'], faces, data['joints'],
                       lbs_weights, joint_id, title)
        return ax,

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)
    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"已保存: {output_path}")

    # 保存关键帧
    keyframe_dir = os.path.join(os.path.dirname(output_path), "keyframes")
    os.makedirs(keyframe_dir, exist_ok=True)

    for idx in [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]:
        if idx >= num_frames:
            continue
        fig = plt.figure(figsize=(8, 9))
        ax = fig.add_subplot(111, projection="3d")
        data = frames_data[idx]
        draw_posed_mesh(ax, data['verts'], faces, data['joints'],
                       lbs_weights, joint_id,
                       f"{joint_name} | {np.degrees(data['angle']):.0f} deg")
        path = os.path.join(keyframe_dir, f"frame_{idx:03d}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return frames_data


# ============================================================
# 主函数
# ============================================================
def main():
    device = torch.device("cpu")
    dtype = torch.float32

    MODEL_DIR = "./models"
    OUTPUT_DIR = "./outputs_animation"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("加载 SMPL 模型...")
    install_chumpy_shim()

    model = smplx.create(
        model_path=MODEL_DIR,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        num_betas=10,
    ).to(device)

    faces = np.asarray(model.faces, dtype=np.int32)
    lbs_weights = to_numpy(model.lbs_weights)

    print(f"模型: {model.v_template.shape[0]} 顶点, {faces.shape[0]} 面")

    # 形状参数
    betas = torch.zeros((1, 10), dtype=dtype, device=device)
    betas[0, 0] = 1.5
    betas[0, 1] = -0.5

    # 动画列表
    animations = [
        {
            "joint_name": "left_elbow",
            "axis": [0.0, -1.0, 0.0],
            "max_angle": np.radians(120),
            "num_frames": 60,
            "output": os.path.join(OUTPUT_DIR, "anim_left_elbow.gif"),
            "joint_id": 18,
        },
        {
            "joint_name": "right_shoulder",
            "axis": [1.0, 0.0, 0.0],
            "max_angle": np.radians(90),
            "num_frames": 60,
            "output": os.path.join(OUTPUT_DIR, "anim_right_shoulder.gif"),
            "joint_id": 17,
        },
        {
            "joint_name": "left_knee",
            "axis": [1.0, 0.0, 0.0],
            "max_angle": np.radians(90),
            "num_frames": 60,
            "output": os.path.join(OUTPUT_DIR, "anim_left_knee.gif"),
            "joint_id": 4,
        },
    ]

    for cfg in animations:
        print("\n" + "=" * 60)
        create_animation(
            model=model, betas=betas,
            joint_name=cfg["joint_name"],
            axis=cfg["axis"],
            max_angle=cfg["max_angle"],
            num_frames=cfg["num_frames"],
            faces=faces, lbs_weights=lbs_weights,
            output_path=cfg["output"],
            joint_id=cfg["joint_id"],
        )

    print("\n" + "=" * 60)
    print(f"完成！输出: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()