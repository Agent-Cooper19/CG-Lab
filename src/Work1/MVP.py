import taichi as ti
import math

ti.init(arch=ti.cpu)

# 顶点数据
vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=3)

@ti.func
def get_model_matrix(angle):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

@ti.func
def get_view_matrix(eye):
    return ti.Matrix([[1, 0, 0, -eye[0]], [0, 1, 0, -eye[1]], 
                      [0, 0, 1, -eye[2]], [0, 0, 0, 1]])

@ti.func
def get_projection_matrix(fov, aspect, near, far):
    n = -near
    f = -far
    t = ti.tan(fov * math.pi / 360.0) * ti.abs(n)
    r = aspect * t
    return ti.Matrix([[n/r, 0, 0, 0], [0, n/t, 0, 0], 
                      [0, 0, (n+f)/(n-f), -2*n*f/(n-f)], [0, 0, 1, 0]])

@ti.kernel
def compute(angle: ti.f32):
    eye = ti.Vector([0.0, 0.0, 5.0])
    mvp = get_projection_matrix(45.0, 1.0, 0.1, 50.0) @ get_view_matrix(eye) @ get_model_matrix(angle)
    
    for i in range(3):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = 1.0 - (v_ndc[1] + 1.0) / 2.0

def main():
    # 初始化三角形顶点
    vertices[0] = [2.0, 0.0, -2.0]
    vertices[1] = [0.0, 2.0, -2.0]
    vertices[2] = [-2.0, 0.0, -2.0]
    
    gui = ti.GUI("MVP Transformation", (700, 700))
    angle = 0.0
    
    while gui.running:
        # 处理键盘输入
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == 'a':
                angle += 10  # A键：角度增加
            elif e.key == 'd':
                angle -= 10  # D键：角度减少
        
        # 清除画面
        gui.clear(0x000000)
        
        # 计算变换
        compute(angle)
        
        # 获取屏幕坐标
        a = screen_coords[0]
        b = screen_coords[1]
        c = screen_coords[2]
        
        # 绘制三角形边（使用不同颜色）
        gui.line(a, b, radius=2, color=0xFF5733)  # 红色
        gui.line(b, c, radius=2, color=0x33FF57)  # 绿色
        gui.line(c, a, radius=2, color=0x3357FF)  # 蓝色
        
        # 绘制顶点
        gui.circle(a, radius=4, color=0xFF5733)
        gui.circle(b, radius=4, color=0x33FF57)
        gui.circle(c, radius=4, color=0x3357FF)
        gui.show()

if __name__ == '__main__':
    main()