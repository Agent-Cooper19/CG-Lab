import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# 尝试导入 IPython（仅在 Jupyter 环境中需要）
try:
    from IPython.display import clear_output
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    print("未检测到 IPython，将使用控制台输出模式")

import pytorch3d
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams
)

# 确认设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")
print(f"PyTorch3D 版本: {pytorch3d.__version__}")

# ============================================================
# 1. 加载目标模型并预处理
# ============================================================
print("\n" + "="*60)
print("1. 加载目标奶牛模型")
print("="*60)

obj_path = "cow.obj"
if not os.path.exists(obj_path):
    print(f"警告：未找到 {obj_path} 文件！")
    print(f"当前工作目录: {os.getcwd()}")
    print("将使用一个简单的测试形状代替...")

    # 创建一个简单的测试目标（环面）
    from pytorch3d.utils import torus
    target_mesh = torus(r=0.5, R=1.0, sides=64, rings=64, device=device)
    print("已创建测试环面作为目标模型")
else:
    # 加载目标网格
    verts, faces, aux = load_obj(obj_path)
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # 归一化处理
    print(f"原始顶点范围: [{verts.min().item():.3f}, {verts.max().item():.3f}]")
    verts = (verts - verts.mean(0)) / max(verts.abs().max(0)[0])
    print(f"归一化后顶点范围: [{verts.min().item():.3f}, {verts.max().item():.3f}]")

    # 创建目标网格
    target_mesh = Meshes(verts=[verts], faces=[faces_idx])

print(f"目标网格: {target_mesh.verts_packed().shape[0]} 个顶点, {target_mesh.faces_packed().shape[0]} 个面")

# ============================================================
# 2. 配置渲染管线和摄像机
# ============================================================
print("\n" + "="*60)
print("2. 配置渲染管线")
print("="*60)

# 超参数配置
IMAGE_SIZE = 256
NUM_VIEWS = 20
SIGMA = 1e-4
GAMMA = 1e-4

# 创建摄像机
elevation = torch.zeros(NUM_VIEWS)
azimuth = torch.linspace(-180, 180, NUM_VIEWS)
distance = 2.7

R, T = look_at_view_transform(distance, elevation, azimuth)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

print(f"创建了 {NUM_VIEWS} 个摄像机视角")

# 配置光栅化器
raster_settings = RasterizationSettings(
    image_size=IMAGE_SIZE,
    blur_radius=np.log(1.0 / 1e-4 - 1.0) * SIGMA,
    faces_per_pixel=50,
    bin_size=0,
    max_faces_per_bin=100000
)

rasterizer = MeshRasterizer(
    cameras=cameras,
    raster_settings=raster_settings
)

# 创建软剪影着色器
blend_params = BlendParams(sigma=SIGMA, gamma=GAMMA)
silhouette_shader = SoftSilhouetteShader(blend_params=blend_params)

# 渲染目标剪影
print("渲染目标剪影图...")
target_silhouettes = silhouette_shader(
    rasterizer(target_mesh.extend(NUM_VIEWS)),
    target_mesh.extend(NUM_VIEWS)
)[..., 3]

print(f"目标剪影形状: {target_silhouettes.shape}")

# ============================================================
# 3. 初始化源模型（球体）
# ============================================================
print("\n" + "="*60)
print("3. 初始化源模型")
print("="*60)

ICO_SPHERE_LEVEL = 4
src_mesh = ico_sphere(ICO_SPHERE_LEVEL, device)
print(f"初始球体: {src_mesh.verts_packed().shape[0]} 个顶点, {src_mesh.faces_packed().shape[0]} 个面")

# 创建可训练的顶点偏移量
deform_verts = torch.zeros_like(src_mesh.verts_packed(), requires_grad=True)
print(f"可训练参数数量: {deform_verts.numel()}")

# ============================================================
# 4. 配置优化器
# ============================================================
print("\n" + "="*60)
print("4. 配置优化器")
print("="*60)

LEARNING_RATE = 1.0
MOMENTUM = 0.9
NUM_EPOCHS = 300

W_LAPLACIAN = 1.0
W_EDGE = 0.1
W_NORMAL = 0.01

print(f"学习率: {LEARNING_RATE}, 动量: {MOMENTUM}")
print(f"正则化权重 - 拉普拉斯: {W_LAPLACIAN}, 边长: {W_EDGE}, 法线: {W_NORMAL}")

optimizer = torch.optim.SGD([deform_verts], lr=LEARNING_RATE, momentum=MOMENTUM)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.1)

# ============================================================
# 5. 创建输出目录
# ============================================================
output_dir = "output_meshes"
os.makedirs(output_dir, exist_ok=True)
print(f"\n中间模型将保存在: ./{output_dir}/")

# ============================================================
# 6. 可微优化循环
# ============================================================
print("\n" + "="*60)
print("6. 开始优化循环")
print("="*60)

loss_history = {
    'total': [],
    'silhouette': [],
    'laplacian': [],
    'edge': [],
    'normal': []
}

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()

    # 应用顶点偏移
    new_src_mesh = src_mesh.offset_verts(deform_verts)

    # 渲染当前网格的剪影
    pred_silhouettes = silhouette_shader(
        rasterizer(new_src_mesh.extend(NUM_VIEWS)),
        new_src_mesh.extend(NUM_VIEWS)
    )[..., 3]

    # 计算损失
    loss_silhouette = ((pred_silhouettes - target_silhouettes) ** 2).mean()
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    loss_edge = mesh_edge_loss(new_src_mesh)
    loss_normal = mesh_normal_consistency(new_src_mesh)

    total_loss = (
        loss_silhouette +
        W_LAPLACIAN * loss_laplacian +
        W_EDGE * loss_edge +
        W_NORMAL * loss_normal
    )

    # 反向传播
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_([deform_verts], max_norm=10.0)
    optimizer.step()
    scheduler.step()

    # 记录损失
    loss_history['total'].append(total_loss.item())
    loss_history['silhouette'].append(loss_silhouette.item())
    loss_history['laplacian'].append(loss_laplacian.item())
    loss_history['edge'].append(loss_edge.item())
    loss_history['normal'].append(loss_normal.item())

    # 定期输出
    if epoch % 20 == 0 or epoch == NUM_EPOCHS - 1:
        elapsed_time = time.time() - start_time

        # 控制台输出（始终可用）
        print(f"\n{'='*40}")
        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | 用时: {elapsed_time:.1f}s")
        print(f"总损失:     {total_loss.item():.6f}")
        print(f"  剪影损失: {loss_silhouette.item():.6f}")
        print(f"  拉普拉斯: {loss_laplacian.item():.6f}")
        print(f"  边长损失: {loss_edge.item():.6f}")
        print(f"  法线损失: {loss_normal.item():.6f}")

        # 保存当前网格
        current_verts = new_src_mesh.verts_list()[0].detach()
        current_faces = new_src_mesh.faces_list()[0].detach()

        save_path = os.path.join(output_dir, f"mesh_epoch_{epoch:03d}.obj")
        save_obj(save_path, current_verts, current_faces)
        print(f"[保存] 模型已保存至: {save_path}")

        # 可视化（matplotlib 在两种环境下都能工作）
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 第一行：目标 vs 预测剪影
        axes[0, 0].imshow(target_silhouettes[0].cpu().numpy(), cmap='gray')
        axes[0, 0].set_title("Target Silhouette (View 0)")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(pred_silhouettes[0].detach().cpu().numpy(), cmap='gray')
        axes[0, 1].set_title(f"Predicted Silhouette (Epoch {epoch})")
        axes[0, 1].axis('off')

        # 差异图
        diff = (pred_silhouettes[0] - target_silhouettes[0]).detach().cpu().numpy()
        axes[0, 2].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, 2].set_title("Difference Map")
        axes[0, 2].axis('off')

        # 第二行：不同视角的预测剪影
        for idx, view_idx in enumerate([5, 10, 15]):
            axes[1, idx].imshow(
                pred_silhouettes[view_idx].detach().cpu().numpy(),
                cmap='gray'
            )
            axes[1, idx].set_title(f"Predicted View {view_idx}")
            axes[1, idx].axis('off')

        plt.tight_layout()

        # 保存图片而不是显示（避免阻塞）
        plt.savefig(os.path.join(output_dir, f"comparison_epoch_{epoch:03d}.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形释放内存

        # 如果是 Jupyter 环境，才使用 clear_output
        if IPYTHON_AVAILABLE:
            clear_output(wait=True)

# ============================================================
# 7. 训练完成后保存结果和损失曲线
# ============================================================
print("\n" + "="*60)
print("7. 优化完成！")
print("="*60)

total_time = time.time() - start_time
print(f"总训练时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")

# 保存最终模型
final_verts = new_src_mesh.verts_list()[0].detach()
final_faces = new_src_mesh.faces_list()[0].detach()
final_path = os.path.join(output_dir, "final_mesh.obj")
save_obj(final_path, final_verts, final_faces)
print(f"最终模型已保存至: {final_path}")

# 绘制损失曲线
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(loss_history['total'], label='Total Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Total Loss over Training')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(loss_history['silhouette'], label='Silhouette', alpha=0.8)
axes[1].plot(loss_history['laplacian'], label='Laplacian', alpha=0.8)
axes[1].plot(loss_history['edge'], label='Edge', alpha=0.8)
axes[1].plot(loss_history['normal'], label='Normal', alpha=0.8)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Loss Components')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n所有输出文件已保存在 ./{output_dir}/ 目录下")