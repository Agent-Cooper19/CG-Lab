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
use_blinn_phong = ti.field(ti.i32, shape=())
enable_shadow = ti.field(ti.i32, shape=())

@ti.func
def normalize(v):
    return v / (v.norm() + 1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

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
        if t1 > 0.001:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, point, normal):
    """测试光线与平面相交"""
    t = -1.0
    denom = rd.dot(normal)
    if ti.abs(denom) > 1e-5:
        t = (point - ro).dot(normal) / denom
    return t

@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    """测试光线与竖直圆锥相交"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2
    
    ro_local = ro - apex
    
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    
    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)
            
            t_first = t1
            t_second = t2
            if t1 > t2:
                t_first, t_second = t_second, t_first
                
            y1 = ro_local.y + t_first * rd.y
            if t_first > 0.001 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 0.001 and -H <= y2 <= 0:
                    t = t_second
                    
            if t > 0:
                p_local = ro_local + rd * t
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))
                
    return t, normal

@ti.func
def shadow_ray(ro, light_pos):
    """
    发射阴影射线检测是否有物体遮挡光源
    """
    in_shadow = False
    
    light_dir = light_pos - ro
    light_dist = light_dir.norm()
    light_dir = light_dir / light_dist
    
    # 检查球体遮挡
    t_sph, _ = intersect_sphere(ro, light_dir, ti.Vector([0.0, 0.5, -1.5]), 1.0)
    if 0 < t_sph and t_sph < light_dist:
        in_shadow = True
    
    # 检查圆锥遮挡
    t_cone, _ = intersect_cone(ro, light_dir, ti.Vector([0.0, 1.5, 1.5]), -1.5, 1.0)
    if 0 < t_cone and t_cone < light_dist:
        in_shadow = True
    
    return in_shadow

@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        
        ro = ti.Vector([0.0, 1.0, 5.0])  # 提高相机位置
        rd = normalize(ti.Vector([u, v - 0.3, -1.0]))  # 稍微向下看

        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 1. 地面平面
        t_plane = intersect_plane(ro, rd, ti.Vector([0.0, -1.5, 0.0]), ti.Vector([0.0, 1.0, 0.0]))
        if 0 < t_plane and t_plane < min_t:
            min_t = t_plane
            hit_normal = ti.Vector([0.0, 1.0, 0.0])
            hit_color = ti.Vector([0.5, 0.5, 0.5])  # 灰色地面
        
        # 2. 红色球（悬浮在空中，会在地面投射阴影）
        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([0.0, 0.5, -1.5]), 1.0)
        if 0 < t_sph and t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.9, 0.2, 0.2])
            
        # 3. 蓝色圆锥（会在地面和球上投射阴影）
        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([0.0, 1.5, 1.5]), -1.5, 1.0)
        if 0 < t_cone and t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.2, 0.4, 0.9])

        color = ti.Vector([0.1, 0.1, 0.2])  # 深色背景

        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal
            
            # 光源设置在侧面，更容易产生明显阴影
            light_pos = ti.Vector([3.0, 4.0, -2.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])
            
            # 环境光降低，让阴影更明显
            ambient_strength = 0.15
            
            in_shadow = False
            if enable_shadow[None] == 1:
                shadow_origin = p + N * 0.002
                in_shadow = shadow_ray(shadow_origin, light_pos)
            
            spec = 0.0
            
            if in_shadow:
                # 阴影中只有很暗的环境光
                ambient = ambient_strength * light_color * hit_color
                color = ambient
            else:
                L = normalize(light_pos - p)
                V = normalize(ro - p)
                
                ambient = ambient_strength * light_color * hit_color
                
                diff = ti.max(0.0, N.dot(L))
                diffuse = Kd[None] * diff * light_color * hit_color
                
                if use_blinn_phong[None] == 1:
                    H = normalize(L + V)
                    spec = ti.max(0.0, N.dot(H)) ** shininess[None]
                else:
                    R = normalize(reflect(-L, N))
                    spec = ti.max(0.0, R.dot(V)) ** shininess[None]
                
                specular = Ks[None] * spec * light_color
                
                color = ambient + diffuse + specular
                
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Blinn-Phong & Hard Shadow Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    Ka[None] = 0.15
    Kd[None] = 0.8
    Ks[None] = 0.4
    shininess[None] = 32.0
    use_blinn_phong[None] = 1
    enable_shadow[None] = 1

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Lighting Model", 0.7, 0.05, 0.28, 0.3):
            gui.text("Material Parameters")
            Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 128.0)
            
            gui.text("")
            gui.text("Rendering Options")
            
            phong_mode = gui.checkbox("Use Blinn-Phong", use_blinn_phong[None])
            use_blinn_phong[None] = 1 if phong_mode else 0
            
            shadow_mode = gui.checkbox("Enable Hard Shadow", enable_shadow[None])
            enable_shadow[None] = 1 if shadow_mode else 0
            
            gui.text("")
            if use_blinn_phong[None]:
                gui.text("Current: Blinn-Phong Model", (0.27, 1.0, 0.27))
            else:
                gui.text("Current: Phong Model", (1.0, 0.27, 0.27))
                
            if enable_shadow[None]:
                gui.text("Shadows: Enabled", (0.27, 1.0, 0.27))
            else:
                gui.text("Shadows: Disabled", (1.0, 0.27, 0.27))

        window.show()

if __name__ == '__main__':
    main()