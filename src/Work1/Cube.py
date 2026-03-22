import taichi as ti
import math

ti.init(arch=ti.cpu)

# 立方体的8个顶点
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
# 立方体的12条边
edges = ti.Vector.field(2, dtype=ti.i32, shape=12)
# 变换后的屏幕坐标
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

@ti.func
def get_model_matrix(angle_x: ti.f32, angle_y: ti.f32):
    """
    模型变换矩阵：绕X轴和Y轴旋转
    """
    rad_x = angle_x * math.pi / 180.0
    rad_y = angle_y * math.pi / 180.0
    
    cx = ti.cos(rad_x)
    sx = ti.sin(rad_x)
    cy = ti.cos(rad_y)
    sy = ti.sin(rad_y)
    
    # 组合旋转：先绕Y轴，再绕X轴
    return ti.Matrix([
        [cy, 0, sy, 0],
        [sx*sy, cx, -sx*cy, 0],
        [-cx*sy, sx, cx*cy, 0],
        [0, 0, 0, 1]
    ])

@ti.func
def get_view_matrix(eye_pos, center, up):
    """
    完整的视图变换矩阵：使用lookAt函数
    """
    # 计算相机坐标系的基向量
    f = (center - eye_pos).normalized()  # 前向向量
    s = f.cross(up).normalized()          # 右向量
    u = s.cross(f)                         # 上向量
    
    # 构建视图矩阵
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
def compute_transform(angle_x: ti.f32, angle_y: ti.f32):
    """
    计算所有顶点的变换
    """
    # 相机位置：从正前方稍远一点观察
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    # 看向原点（立方体中心）
    center = ti.Vector([0.0, 0.0, 0.0])
    # 上方向
    up = ti.Vector([0.0, 1.0, 0.0])
    
    model = get_model_matrix(angle_x, angle_y)
    view = get_view_matrix(eye_pos, center, up)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model
    
    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = 1.0 - (v_ndc[1] + 1.0) / 2.0

def init_cube():
    """初始化立方体的顶点和边"""
    # 立方体的8个顶点（中心在原点）
    vertices[0] = [-1.0, -1.0, -1.0]  # 0: 后下左
    vertices[1] = [1.0, -1.0, -1.0]   # 1: 后下右
    vertices[2] = [1.0, 1.0, -1.0]    # 2: 后上右
    vertices[3] = [-1.0, 1.0, -1.0]   # 3: 后上左
    vertices[4] = [-1.0, -1.0, 1.0]   # 4: 前下左
    vertices[5] = [1.0, -1.0, 1.0]    # 5: 前下右
    vertices[6] = [1.0, 1.0, 1.0]     # 6: 前上右
    vertices[7] = [-1.0, 1.0, 1.0]    # 7: 前上左
    
    # 立方体的12条边
    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 后面四条边
        (4, 5), (5, 6), (6, 7), (7, 4),  # 前面四条边
        (0, 4), (1, 5), (2, 6), (3, 7)   # 连接前后面的四条边
    ]
    
    for i in range(12):
        edges[i] = ti.Vector(edge_pairs[i])

def main():
    # 初始化立方体
    init_cube()
    
    gui = ti.GUI("3D Cube - Centered", (700, 700))
    angle_x = 0.0
    angle_y = 0.0
    
    while gui.running:
        # 处理键盘输入
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == 'a':
                angle_y += 5
            elif e.key == 'd':
                angle_y -= 5
            elif e.key == 'w':
                angle_x += 5
            elif e.key == 's':
                angle_x -= 5
            elif e.key == 'r':
                angle_x = angle_y = 0.0
        
        # 清除画面
        gui.clear(0x000000)
        
        # 计算变换
        compute_transform(angle_x, angle_y)
        
        # 绘制坐标轴参考线（可选）
        # 中心点
        gui.circle((0.5, 0.5), radius=2, color=0xFFFFFF)
        
        # 绘制立方体的所有边
        for i in range(12):
            edge = edges[i]
            idx1 = edge[0]
            idx2 = edge[1]
            p1 = screen_coords[idx1]
            p2 = screen_coords[idx2]
            
            # 根据深度给不同颜色
            if i < 4:  # 后面
                color = 0xFF9999  # 浅红
            elif i < 8:  # 前面
                color = 0x99FF99  # 浅绿
            else:  # 连接线
                color = 0x9999FF  # 浅蓝
            
            gui.line(p1, p2, radius=2, color=color)
        
        # 绘制顶点
        for i in range(8):
            gui.circle(screen_coords[i], radius=3, color=0xFFFF00)
        
        # 显示控制说明
        gui.text("W/S: X-axis | A/D: Y-axis | R: Reset | ESC: Exit", 
                pos=(0.02, 0.02), color=0xAAAAAA)
        gui.text(f"Angle X:{angle_x:.0f}° Y:{angle_y:.0f}°", 
                pos=(0.02, 0.95), color=0xFFFFFF)
        
        gui.show()

if __name__ == '__main__':
    main()