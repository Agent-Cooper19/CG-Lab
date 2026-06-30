import taichi as ti
import taichi.math as tm

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# ============================================================
# 物理与网格参数
# ============================================================
N = 20                      # 布料网格分辨率 N x N
mass = 1.0                  # 质点质量
dt = 5e-4                   # 时间步长
k_s = 10000.0               # 弹簧劲度系数（结构弹簧）
k_s_shear = 5000.0          # 剪切弹簧劲度系数
k_s_bend = 2000.0           # 弯曲弹簧劲度系数
k_d = 1.0                   # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0         # 速度上限

# ============================================================
# 碰撞检测参数
# ============================================================
enable_collision = True
sphere_center = ti.Vector([0.0, -0.2, 0.0])
sphere_radius = 0.3
collision_stiffness = 20000.0
collision_damping = 500.0
collision_restitution = 0.1

# ============================================================
# Taichi 数据场定义
# ============================================================
x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

# 用于 CCD 的旧位置缓存（在 Python 层定义一次）
x_old = ti.Vector.field(3, dtype=float, shape=N * N)

# 隐式欧拉专用
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

# 弹簧数据场
max_springs = N * N * 12
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_types = ti.field(dtype=int, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())

SPRING_STRUCTURAL = 0
SPRING_SHEAR = 1
SPRING_BEND = 2

# ============================================================
# 初始化函数
# ============================================================

@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        x_old[idx] = x[idx]
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        
        # 结构弹簧
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
            spring_types[c] = SPRING_STRUCTURAL
        
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()
            spring_types[c] = SPRING_STRUCTURAL
        
        # 剪切弹簧
        if i < N - 1 and j < N - 1:
            idx_diag = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag])
            spring_lengths[c] = (x[idx] - x[idx_diag]).norm()
            spring_types[c] = SPRING_SHEAR
        
        if i < N - 1 and j > 0:
            idx_diag2 = (i + 1) * N + (j - 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag2])
            spring_lengths[c] = (x[idx] - x[idx_diag2]).norm()
            spring_types[c] = SPRING_SHEAR
        
        # 弯曲弹簧
        if i < N - 2:
            idx_bend_h = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_bend_h])
            spring_lengths[c] = (x[idx] - x[idx_bend_h]).norm()
            spring_types[c] = SPRING_BEND
        
        if j < N - 2:
            idx_bend_v = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_bend_v])
            spring_lengths[c] = (x[idx] - x[idx_bend_v]).norm()
            spring_types[c] = SPRING_BEND

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()
    print(f"布料初始化完成: {N}x{N} 质点, {num_springs[None]} 根弹簧")

# ============================================================
# 力学计算 (ti.func)
# ============================================================

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    """计算所有力"""
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        
        if dist > 1e-6:
            d_normalized = d / dist
            
            spring_k = k_s
            if spring_types[i] == SPRING_SHEAR:
                spring_k = k_s_shear
            elif spring_types[i] == SPRING_BEND:
                spring_k = k_s_bend
            
            f_spring = -spring_k * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)
    
    # 强碰撞力
    if enable_collision:
        for i in range(N * N):
            if is_fixed[i] == 0:
                to_sphere = pos[i] - sphere_center
                dist_to_sphere = to_sphere.norm()
                
                if dist_to_sphere < sphere_radius and dist_to_sphere > 1e-8:
                    normal = to_sphere / dist_to_sphere
                    penetration = sphere_radius - dist_to_sphere
                    
                    # 指数级排斥力
                    force_magnitude = collision_stiffness * penetration * ti.exp(penetration * 5.0)
                    force[i] += force_magnitude * normal
                    
                    vel_normal = vel[i].dot(normal)
                    if vel_normal < 0:
                        force[i] -= collision_damping * vel_normal * normal

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.func
def resolve_collision(pos: ti.template(), vel: ti.template(), idx: int):
    """碰撞约束修正"""
    if enable_collision and is_fixed[idx] == 0:
        to_sphere = pos[idx] - sphere_center
        dist_to_sphere = to_sphere.norm()
        
        if dist_to_sphere < sphere_radius and dist_to_sphere > 1e-8:
            normal = to_sphere / dist_to_sphere
            
            # 推出到球体表面外
            pos[idx] = sphere_center + normal * (sphere_radius + 1e-4)
            
            # 法向速度修正
            vel_normal = vel[idx].dot(normal)
            if vel_normal < 0:
                vel[idx] -= vel_normal * normal
                vel[idx] += abs(vel_normal) * normal * collision_restitution
                
                # 切向摩擦
                tangent_vel = vel[idx] - vel[idx].dot(normal) * normal
                vel[idx] -= tangent_vel * 0.1

# ============================================================
# 积分求解器
# ============================================================

@ti.kernel
def step_explicit():
    """显式欧拉"""
    # 保存旧位置
    for i in range(N * N):
        x_old[i] = x[i]
    
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            resolve_collision(x, v, i)

@ti.kernel
def step_semi_implicit():
    """半隐式欧拉"""
    # 保存旧位置
    for i in range(N * N):
        x_old[i] = x[i]
    
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            resolve_collision(x, v, i)

@ti.kernel
def step_implicit_iter():
    """隐式欧拉"""
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
                resolve_collision(x_next, v_next, i)
    
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]
        resolve_collision(x, v, i)

# ============================================================
# 主函数
# ============================================================

def main():
    init_cloth()

    window = ti.ui.Window("Mass-Spring Cloth Simulation", (900, 700))
    canvas = window.get_canvas()
    scene = window.get_scene()
    
    camera = ti.ui.Camera()
    camera.position(0.0, 0.3, 2.0)
    camera.lookat(0.0, -0.1, 0.0)

    current_method = 1
    paused = False
    method_names = {0: "Explicit Euler", 1: "Semi-Implicit", 2: "Implicit"}

    while window.running:
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.35, 0.48)

        window.GUI.text("=== Integration Method ===")
        
        prefix_0 = "> " if current_method == 0 else "  "
        prefix_1 = "> " if current_method == 1 else "  "
        prefix_2 = "> " if current_method == 2 else "  "

        if window.GUI.button(prefix_0 + "Explicit Euler"):
            current_method = 0
            init_cloth()
        
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler"):
            current_method = 1
            init_cloth()
        
        if window.GUI.button(prefix_2 + "Implicit Euler"):
            current_method = 2
            init_cloth()

        window.GUI.text("")
        window.GUI.text("=== Simulation Control ===")
        
        if window.GUI.button("Pause" if not paused else "Resume"):
            paused = not paused
        
        if window.GUI.button("Reset Cloth"):
            init_cloth()
        
        window.GUI.text("")
        window.GUI.text("=== Parameters ===")
        
        global k_d
        k_d = window.GUI.slider_float("Damping", k_d, 0.1, 10.0)
        
        window.GUI.text("")
        window.GUI.text(f"Method: {method_names[current_method]}")
        window.GUI.text(f"Status: {'Paused' if paused else 'Running'}")
        window.GUI.text(f"Damping: {k_d:.1f}")
        window.GUI.text(f"Springs: {num_springs[None]}")

        window.GUI.end()

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(1.0, 1.5, 2.0), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(-1.0, 1.0, -1.0), color=(0.5, 0.5, 0.8))

        scene.particles(x, radius=0.012, color=(0.2, 0.5, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))

        if enable_collision:
            ball_pos = ti.Vector.field(3, dtype=float, shape=1)
            ball_pos[0] = sphere_center
            scene.particles(ball_pos, radius=sphere_radius, color=(1.0, 0.6, 0.2))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()