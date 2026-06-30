import taichi as ti
import math

ti.init(arch=ti.cpu)

# 两个立方体的顶点（世界坐标）
cube1_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)  # 立方体1原始顶点
cube2_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)  # 立方体2原始顶点
# 变换后的屏幕坐标
screen_coords1 = ti.Vector.field(2, dtype=ti.f32, shape=8)  # 立方体1屏幕坐标
screen_coords2 = ti.Vector.field(2, dtype=ti.f32, shape=8)  # 立方体2屏幕坐标
# 立方体的12条边
edges = ti.Vector.field(2, dtype=ti.i32, shape=12)

@ti.func
def get_rotation_matrix(angle_x: ti.f32, angle_y: ti.f32, angle_z: ti.f32):
    """
    完整的旋转矩阵：绕X、Y、Z轴旋转
    """
    rad_x = angle_x * math.pi / 180.0
    rad_y = angle_y * math.pi / 180.0
    rad_z = angle_z * math.pi / 180.0
    
    cx = ti.cos(rad_x)
    sx = ti.sin(rad_x)
    cy = ti.cos(rad_y)
    sy = ti.sin(rad_y)
    cz = ti.cos(rad_z)
    sz = ti.sin(rad_z)
    
    # 旋转矩阵：先绕Y轴，再绕X轴，最后绕Z轴
    # RX * RY * RZ (注意顺序：先Z，再Y，再X)
    rot = ti.Matrix([
        [cy*cz, -cy*sz, sy, 0],
        [cx*sz + sx*sy*cz, cx*cz - sx*sy*sz, -sx*cy, 0],
        [sx*sz - cx*sy*cz, sx*cz + cx*sy*sz, cx*cy, 0],
        [0, 0, 0, 1]
    ])
    return rot

@ti.func
def get_translation_matrix(translation: ti.template()):
    """
    平移矩阵
    """
    return ti.Matrix([
        [1, 0, 0, translation[0]],
        [0, 1, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1]
    ])

@ti.func
def get_view_matrix(eye_pos, center, up):
    """
    视图变换矩阵
    """
    f = (center - eye_pos).normalized()
    s = f.cross(up).normalized()
    u = s.cross(f)
    
    return ti.Matrix([
        [s[0], s[1], s[2], -s.dot(eye_pos)],
        [u[0], u[1], u[2], -u.dot(eye_pos)],
        [-f[0], -f[1], -f[2], f.dot(eye_pos)],
        [0, 0, 0, 1]
    ])

@ti.func
def get_projection_matrix(fov: ti.f32, aspect: ti.f32, near: ti.f32, far: ti.f32):
    """
    透视投影矩阵
    """
    n = -near
    f = -far
    t = ti.tan(fov * math.pi / 360.0) * ti.abs(n)
    r = aspect * t
    
    return ti.Matrix([
        [n/r, 0, 0, 0],
        [0, n/t, 0, 0],
        [0, 0, (n+f)/(n-f), -2*n*f/(n-f)],
        [0, 0, 1, 0]
    ])

@ti.kernel
def compute_transform(t: ti.f32):
    """
    计算两个立方体的顶点变换
    t: 插值参数，范围[0, 1]
    """
    # 相机设置
    eye_pos = ti.Vector([0.0, 3.0, 8.0])
    center = ti.Vector([0.0, 0.0, 0.0])
    up = ti.Vector([0.0, 1.0, 0.0])
    
    # 立方体1的姿态：绕X轴30度，绕Y轴45度
    angle1_x = 30.0
    angle1_y = 45.0
    angle1_z = 0.0
    # 立方体2的姿态：绕X轴-60度，绕Y轴120度
    angle2_x = -60.0
    angle2_y = 120.0
    angle2_z = 30.0
    
    # 插值角度
    angle_interp_x = angle1_x + t * (angle2_x - angle1_x)
    angle_interp_y = angle1_y + t * (angle2_y - angle1_y)
    angle_interp_z = angle1_z + t * (angle2_z - angle1_z)
    
    # 平移插值：立方体1在左侧(-2, 0, 0)，立方体2在右侧(2, 0, 0)
    trans1 = ti.Vector([-2.0, 0.0, 0.0])
    trans2 = ti.Vector([2.0, 0.0, 0.0])
    trans_interp = trans1 + t * (trans2 - trans1)
    
    # 构建变换矩阵
    model1 = get_translation_matrix(trans_interp) @ get_rotation_matrix(angle_interp_x, angle_interp_y, angle_interp_z)
    
    # 立方体2使用相反的插值
    trans_interp2 = trans2 + t * (trans1 - trans2)
    angle_interp2_x = angle2_x + t * (angle1_x - angle2_x)
    angle_interp2_y = angle2_y + t * (angle1_y - angle2_y)
    angle_interp2_z = angle2_z + t * (angle1_z - angle2_z)
    
    model2 = get_translation_matrix(trans_interp2) @ get_rotation_matrix(angle_interp2_x, angle_interp2_y, angle_interp2_z)
    
    view = get_view_matrix(eye_pos, center, up)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp1 = proj @ view @ model1
    mvp2 = proj @ view @ model2
    
    # 变换立方体1的顶点
    for i in range(8):
        v = cube1_vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp1 @ v4
        v_ndc = v_clip / v_clip[3]
        
        screen_coords1[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords1[i][1] = 1.0 - (v_ndc[1] + 1.0) / 2.0
    
    # 变换立方体2的顶点
    for i in range(8):
        v = cube2_vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp2 @ v4
        v_ndc = v_clip / v_clip[3]
        
        screen_coords2[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords2[i][1] = 1.0 - (v_ndc[1] + 1.0) / 2.0

@ti.func
def lerp(a: ti.f32, b: ti.f32, t: ti.f32) -> ti.f32:
    """线性插值"""
    return a + t * (b - a)

@ti.func
def smoothstep(t: ti.f32) -> ti.f32:
    """平滑插值函数"""
    return t * t * (3.0 - 2.0 * t)

def init_cubes():
    """初始化两个立方体的顶点和边"""
    # 立方体1的标准顶点（边长为1）
    cube1_vertices[0] = [-0.5, -0.5, -0.5]
    cube1_vertices[1] = [0.5, -0.5, -0.5]
    cube1_vertices[2] = [0.5, 0.5, -0.5]
    cube1_vertices[3] = [-0.5, 0.5, -0.5]
    cube1_vertices[4] = [-0.5, -0.5, 0.5]
    cube1_vertices[5] = [0.5, -0.5, 0.5]
    cube1_vertices[6] = [0.5, 0.5, 0.5]
    cube1_vertices[7] = [-0.5, 0.5, 0.5]
    
    # 立方体2使用相同的顶点（变换会处理位置和旋转）
    for i in range(8):
        cube2_vertices[i] = cube1_vertices[i]
    
    # 立方体的12条边
    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 后面四条边
        (4, 5), (5, 6), (6, 7), (7, 4),  # 前面四条边
        (0, 4), (1, 5), (2, 6), (3, 7)   # 连接前后面的四条边
    ]
    
    for i in range(12):
        edges[i] = ti.Vector(edge_pairs[i])

def main():
    init_cubes()
    
    gui = ti.GUI("Two Cubes - Rotation Interpolation", (800, 600))
    
    t = 0.0  # 插值参数
    is_animating = True
    animation_speed = 0.01
    manual_mode = False
    
    while gui.running:
        # 处理键盘输入
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ' ':
                # 空格键切换动画/手动模式
                is_animating = not is_animating
                manual_mode = not is_animating
            elif e.key == 'r':
                t = 0.0
            elif e.key == ti.GUI.LEFT:
                if manual_mode:
                    t = max(0.0, t - 0.02)
            elif e.key == ti.GUI.RIGHT:
                if manual_mode:
                    t = min(1.0, t + 0.02)
        
        # 自动动画
        if is_animating:
            t += animation_speed
            if t > 1.0:
                t = 1.0
                animation_speed = -animation_speed
            elif t < 0.0:
                t = 0.0
                animation_speed = -animation_speed
        
        # 清除画面
        gui.clear(0x1a1a2e)
        
        # 计算变换
        compute_transform(t)
        
        # 绘制坐标轴参考
        gui.line((0.1, 0.5), (0.9, 0.5), radius=1, color=0x333355)  # X轴
        gui.line((0.5, 0.1), (0.5, 0.9), radius=1, color=0x333355)  # Y轴
        
        # 绘制立方体1的边（红色系）
        for i in range(12):
            edge = edges[i]
            p1 = screen_coords1[edge[0]]
            p2 = screen_coords1[edge[1]]
            
            # 根据边的位置给不同深浅的红色
            if i < 4:  # 后面
                color = 0xCC4444
            elif i < 8:  # 前面
                color = 0xFF6666
            else:  # 连接线
                color = 0xDD5555
            
            gui.line(p1, p2, radius=2, color=color)
        
        # 绘制立方体2的边（蓝色系）
        for i in range(12):
            edge = edges[i]
            p1 = screen_coords2[edge[0]]
            p2 = screen_coords2[edge[1]]
            
            if i < 4:  # 后面
                color = 0x3366CC
            elif i < 8:  # 前面
                color = 0x4488FF
            else:  # 连接线
                color = 0x3377DD
            
            gui.line(p1, p2, radius=2, color=color)
        
        # 绘制所有顶点
        for i in range(8):
            gui.circle(screen_coords1[i], radius=4, color=0xFFAA00)
            gui.circle(screen_coords2[i], radius=4, color=0x00CCFF)
        
        # 显示信息
        gui.text("Space: Toggle Animation | Left/Right: Manual Control | R: Reset | ESC: Exit", 
                pos=(0.02, 0.02), color=0xAAAAAA)
        gui.text(f"Interpolation t: {t:.3f}", 
                pos=(0.02, 0.95), color=0xFFFFFF)
        gui.text(f"Mode: {'Auto Animation' if is_animating else 'Manual Control'}", 
                pos=(0.02, 0.90), color=0xFFCC00)
        
        # 显示插值进度条
        bar_start = (0.2, 0.08)
        bar_end = (0.8, 0.08)
        gui.line(bar_start, bar_end, radius=3, color=0x333333)
        progress_pos = (0.2 + t * 0.6, 0.08)
        gui.line(bar_start, progress_pos, radius=4, color=0x44FF44)
        
        gui.show()

if __name__ == '__main__':
    main()