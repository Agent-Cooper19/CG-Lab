import taichi as ti
import math

# 初始化 Taichi
ti.init(arch=ti.gpu)

# 窗口分辨率
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 定义全局交互参数
Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())
use_blinn_phong = ti.field(ti.i32, shape=())  # 0=Phong, 1=Blinn-Phong
enable_shadow = ti.field(ti.i32, shape=())     # 0=No Shadow, 1=Hard Shadow

@ti.func
def normalize(v):
    return v / (v.norm() + 1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

# --- 几何体相交测试函数 ---

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """测试光线与球体相交"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0.001:  # 使用小偏移避免自交
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    """
    测试光线与竖直圆锥相交
    apex: 圆锥顶点坐标
    base_y: 圆锥底面的世界坐标 Y 值
    """
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2
    
    # 转换到以顶点为原点的局部坐标系
    ro_local = ro - apex
    
    # 构建一元二次方程 At^2 + Bt + C = 0
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    
    # 避免 A 为 0 时的除零错误
    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)
            
            # 保证 t_first 是较近的交点
            t_first = t1
            t_second = t2
            if t1 > t2:
                t_first, t_second = t_second, t_first
                
            # 验证交点是否在圆锥的高范围内 (局部 Y 坐标在 [-H, 0] 之间)
            y1 = ro_local.y + t_first * rd.y
            if t_first > 0.001 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 0.001 and -H <= y2 <= 0:
                    t = t_second
                    
            if t > 0:
                p_local = ro_local + rd * t
                # 圆锥表面的法线计算
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))
                
    return t, normal

@ti.func
def shadow_ray(ro, light_pos):
    """
    发射阴影射线检测是否有物体遮挡光源
    返回 True 表示被遮挡（在阴影中）
    """
    in_shadow = False
    
    # 计算到光源的距离和方向
    light_dir = light_pos - ro
    light_dist = light_dir.norm()
    light_dir = light_dir / light_dist
    
    # 检查球体遮挡
    t_sph, _ = intersect_sphere(ro, light_dir, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
    if 0 < t_sph and t_sph < light_dist:
        in_shadow = True
    
    # 检查圆锥遮挡
    t_cone, _ = intersect_cone(ro, light_dir, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
    if 0 < t_cone and t_cone < light_dist:
        in_shadow = True
    
    return in_shadow

@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        # 用于记录光线击中的最近物体
        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 1. 渲染红球 (放在左边)
        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t_sph and t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])
            
        # 2. 渲染紫色圆锥 (放在右边)
        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone and t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])

        # 背景色
        color = ti.Vector([0.05, 0.15, 0.15]) 

        # 如果击中了任何物体
        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal
            
            # 光源设置
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])
            
            # 检查阴影
            in_shadow = False
            if enable_shadow[None] == 1:
                # 从交点稍微偏移，避免自交
                shadow_origin = p + N * 0.001
                in_shadow = shadow_ray(shadow_origin, light_pos)
            
            # 初始化 spec 变量
            spec = 0.0
            
            if in_shadow:
                # 阴影中只计算环境光
                ambient = Ka[None] * light_color * hit_color
                color = ambient
            else:
                L = normalize(light_pos - p)
                V = normalize(ro - p)
                
                # 环境光分量
                ambient = Ka[None] * light_color * hit_color
                
                # 漫反射分量
                diff = ti.max(0.0, N.dot(L))
                diffuse = Kd[None] * diff * light_color * hit_color
                
                # 高光分量
                if use_blinn_phong[None] == 1:
                    # Blinn-Phong 模型：使用半程向量 H
                    H = normalize(L + V)
                    spec = ti.max(0.0, N.dot(H)) ** shininess[None]
                else:
                    # Phong 模型：使用反射向量 R
                    R = normalize(reflect(-L, N))
                    spec = ti.max(0.0, R.dot(V)) ** shininess[None]
                
                specular = Ks[None] * spec * light_color
                
                # 组合光照
                color = ambient + diffuse + specular
                
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Blinn-Phong & Hard Shadow Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化材质参数
    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 32.0
    use_blinn_phong[None] = 1  # 默认使用 Blinn-Phong
    enable_shadow[None] = 1   # 默认开启阴影

    while window.running:
        # 执行并行渲染
        render()
        
        # 将渲染结果绘制到画布
        canvas.set_image(pixels)
        
        # 绘制交互面板
        with gui.sub_window("Lighting Model", 0.7, 0.05, 0.28, 0.3):
            gui.text("Material Parameters")
            Ka[None] = gui.slider_float('Ka (Ambient)', Ka[None], 0.0, 1.0)
            Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 128.0)
            
            gui.text("")  # 空行
            gui.text("Rendering Options")
            
            # Phong/Blinn-Phong 切换
            phong_mode = gui.checkbox("Use Blinn-Phong", use_blinn_phong[None])
            use_blinn_phong[None] = 1 if phong_mode else 0
            
            # 阴影开关
            shadow_mode = gui.checkbox("Enable Hard Shadow", enable_shadow[None])
            enable_shadow[None] = 1 if shadow_mode else 0
            
            gui.text("")
            # 修复颜色参数 - 使用元组格式 (R, G, B)
            if use_blinn_phong[None]:
                gui.text("Current: Blinn-Phong Model", (0.27, 1.0, 0.27))  # 绿色
            else:
                gui.text("Current: Phong Model", (1.0, 0.27, 0.27))  # 红色
                
            if enable_shadow[None]:
                gui.text("Shadows: Enabled", (0.27, 1.0, 0.27))  # 绿色
            else:
                gui.text("Shadows: Disabled", (1.0, 0.27, 0.27))  # 红色

        # 显示窗口
        window.show()

if __name__ == '__main__':
    main()