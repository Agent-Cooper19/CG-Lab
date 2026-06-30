import taichi as ti
import taichi.math as tm

# 初始化 Taichi GPU 后端
ti.init(arch=ti.gpu, random_seed=42)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
sample_count = ti.field(ti.i32, shape=())  # 抗锯齿采样数

# 材质常量枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2  # 折射玻璃材质（选做）

# ============ 基础工具函数 ============

@ti.func
def normalize(v):
    """安全的向量归一化"""
    return v / (v.norm() + 1e-8)

@ti.func
def reflect(I, N):
    """计算反射方向
    I: 入射方向（指向表面）
    N: 表面法线（指向外侧）
    返回：反射方向（指向外侧）
    """
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, ior):
    """计算折射方向（斯涅尔定律）
    I: 入射方向（指向表面）
    N: 表面法线（指向外侧）
    ior: 相对折射率（入射介质折射率/透射介质折射率）
    返回：折射方向，如果全反射则返回零向量
    """
    cos_i = -I.dot(N)  # 入射角的余弦
    sin_t2 = ior * ior * (1.0 - cos_i * cos_i)  # sin²(透射角)
    
    # 初始化为零向量（表示全反射）
    refracted = ti.Vector([0.0, 0.0, 0.0])
    
    # 只有不发生全反射时才计算折射方向
    if sin_t2 <= 1.0:
        cos_t = ti.sqrt(1.0 - sin_t2)
        refracted = ior * I + (ior * cos_i - cos_t) * N
    
    return refracted

@ti.func
def fresnel_schlick(cos_i, ior):
    """Schlick近似的菲涅尔反射率
    cos_i: 入射角余弦（正值）
    ior: 相对折射率
    """
    r0 = (1.0 - ior) / (1.0 + ior)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * ti.pow(1.0 - cos_i, 5.0)

# ============ 场景求交函数 ============

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """球体求交，返回 (距离 t, 法线 normal)"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        t2 = (-b + ti.sqrt(delta)) / 2.0
        # 取最小的正解
        if t1 > 1e-4:
            t = t1
        elif t2 > 1e-4:
            t = t2
        if t > 0:
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    """水平无限大平面求交"""
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-6:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 1e-4:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    """
    遍历场景，寻找最近交点。
    返回: (命中标志 hit, 距离 t, 法线 N, 基础色 color, 材质 mat_id)
    """
    min_t = 1e10
    hit = False
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 红色漫反射球（左侧）
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.5, 0.0, 0.0]), 1.0)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.8, 0.1, 0.1])  # 红色
        hit_mat = MAT_DIFFUSE
        hit = True

    # 2. 银色镜面球（右侧）
    t, n = intersect_sphere(ro, rd, ti.Vector([1.5, 0.0, 0.0]), 1.0)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.95, 0.95, 0.95])  # 银色镜面
        hit_mat = MAT_MIRROR
        hit = True

    # 3. 地板平面（y = -1.0）
    t, n = intersect_plane(ro, rd, -1.0)
    if 0.0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        # 生成棋盘格纹理
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.2, 0.2, 0.2])  # 深灰色格子
        else:
            hit_c = ti.Vector([0.85, 0.85, 0.85])  # 浅灰色格子
        hit = True

    return hit, min_t, hit_n, hit_c, hit_mat

# ============ 渲染核心 ============

@ti.func
def trace_ray(ro, rd):
    """追踪一条射线，返回颜色"""
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.25])
    
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])
    
    # 迭代式光线追踪
    for bounce in range(max_bounces[None]):
        hit, t, N, obj_color, mat_id = scene_intersect(ro, rd)
        
        # 未命中任何物体
        if not hit:
            final_color += throughput * bg_color
            break
        
        p = ro + rd * t  # 交点位置
        
        # === 分支 1：镜面反射材质 ===
        if mat_id == MAT_MIRROR:
            # 偏移起点防止自相交
            ro = p + N * 1e-4
            rd = normalize(reflect(rd, N))
            # 反射率为0.8，物体颜色为银色
            throughput *= 0.8 * obj_color
            # 继续循环（不break）
        
        # === 分支 2：漫反射材质 ===
        elif mat_id == MAT_DIFFUSE:
            L_dir = light_pos - p
            L_dist = L_dir.norm()
            L = L_dir / (L_dist + 1e-8)
            
            # 阴影检测射线
            shadow_ro = p + N * 1e-4  # 关键：偏移防止自相交
            shadow_hit, shadow_t, _, _, _ = scene_intersect(shadow_ro, L)
            
            # Phong光照模型
            ambient = 0.15 * obj_color
            direct_light = ambient
            
            # 如果阴影射线没有在光源之前命中物体，则计算光照
            in_shadow = (shadow_hit and shadow_t < L_dist - 1e-4)
            
            if not in_shadow:
                # 漫反射分量
                diff = ti.max(0.0, N.dot(L))
                diffuse_term = 0.7 * diff * obj_color
                
                # 镜面高光（Blinn-Phong）
                V = normalize(-rd)  # 视线方向
                H = normalize(L + V)  # 半角向量
                spec = ti.pow(ti.max(0.0, N.dot(H)), 64.0)
                specular_term = 0.3 * spec * ti.Vector([1.0, 1.0, 1.0])
                
                direct_light += diffuse_term + specular_term
            
            final_color += throughput * direct_light
            break  # 漫反射终止
        
        # === 分支 3：玻璃材质（选做） ===
        elif mat_id == MAT_GLASS:
            I = rd
            
            # 在分支外部预先声明并初始化所有变量
            outward_n = N
            ior = 1.0 / 1.5
            cos_i_val = -I.dot(N)  # 外部入射
            
            # 判断射线是从外部射入还是从内部射出
            if I.dot(N) < 0.0:  # 从外部射入（法线朝外）
                outward_n = N
                ior = 1.0 / 1.5  # 空气到玻璃
                cos_i_val = -I.dot(N)
            else:  # 从内部射出
                outward_n = -N
                ior = 1.5 / 1.0  # 玻璃到空气
                cos_i_val = I.dot(N)
            
            # 菲涅尔反射率
            refl_prob = fresnel_schlick(cos_i_val, ior)
            
            # 折射方向
            refracted = refract(rd, outward_n, ior)
            is_total_internal = (refracted.norm() < 1e-6)
            
            # 俄罗斯轮盘赌决定反射还是折射
            if ti.random() < refl_prob or is_total_internal:
                # 反射
                ro = p + outward_n * 1e-4
                rd = normalize(reflect(rd, outward_n))
            else:
                # 折射
                ro = p - outward_n * 1e-4  # 向法线反方向偏移
                rd = normalize(refracted)
            
            throughput *= 0.95  # 轻微能量损失
            # 继续循环

    return final_color

@ti.kernel
def render():
    """主渲染函数"""
    # 预计算相机参数
    ro = ti.Vector([0.0, 1.0, 5.0])  # 相机位置
    look_at = ti.Vector([0.0, 0.0, 0.0])
    forward = normalize(look_at - ro)
    world_up = ti.Vector([0.0, 1.0, 0.0])
    right = normalize(world_up.cross(forward))
    up = forward.cross(right)
    
    for i, j in pixels:
        final_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 抗锯齿：每个像素多次采样
        samples = sample_count[None]
        for s in range(samples):
            # 在 if-else 外部声明变量
            offset_x = 0.0
            offset_y = 0.0
            
            # 在像素内随机偏移
            if samples > 1:
                offset_x = ti.random() - 0.5
                offset_y = ti.random() - 0.5
            
            # 计算射线方向（标准化坐标）
            u = (float(i) + offset_x - float(res_x) / 2.0) / float(res_y) * 2.0
            v = (float(j) + offset_y - float(res_y) / 2.0) / float(res_y) * 2.0
            
            # 构建射线方向
            rd = normalize(u * right + v * up + 2.0 * forward)
            
            final_color += trace_ray(ro, rd)
        
        # 平均所有采样
        final_color = final_color / float(samples)
        
        # Gamma校正
        final_color = ti.math.pow(final_color, 1.0 / 2.2)
        
        # 写入像素
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)

# ============ 主函数和UI ============

def main():
    window = ti.ui.Window("Whitted-Style Ray Tracer", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化参数
    light_pos_x[None] = 3.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 2.0
    max_bounces[None] = 3
    sample_count[None] = 1
    
    while window.running:
        render()
        canvas.set_image(pixels)
        
        # 控制面板 - 光源控制
        with gui.sub_window("Light Controls", 0.02, 0.02, 0.25, 0.22):
            gui.text("Light Position")
            light_pos_x[None] = gui.slider_float("Light X", light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float("Light Y", light_pos_y[None], 0.5, 8.0)
            light_pos_z[None] = gui.slider_float("Light Z", light_pos_z[None], -5.0, 5.0)
        
        # 控制面板 - 光线追踪设置
        with gui.sub_window("Ray Tracing Controls", 0.02, 0.27, 0.25, 0.18):
            gui.text("Ray Settings")
            max_bounces[None] = gui.slider_int("Max Bounces", max_bounces[None], 1, 5)
            sample_count[None] = gui.slider_int("AA Samples", sample_count[None], 1, 16)
        
        # 信息面板
        with gui.sub_window("Scene Info", 0.73, 0.02, 0.25, 0.18):
            gui.text("Scene Description")
            gui.text("Left: Red Diffuse Sphere")
            gui.text("Right: Silver Mirror Sphere")
            gui.text("Floor: Checkerboard Pattern")
            gui.text(f"Bounces: {max_bounces[None]}")

        window.show()

if __name__ == "__main__":
    main()