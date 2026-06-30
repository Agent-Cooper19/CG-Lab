import taichi as ti
import numpy as np

# 使用 gpu 后端
ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000  # 曲线采样点数量

# 反走样采样半径
ANTIALIAS_RADIUS = 1.5

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# GUI 绘制数据缓冲池
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# 曲线坐标缓冲区
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

# 控制点缓冲区（GPU端）
control_points_gpu = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)

# B样条曲线需要的更大缓冲区
MAX_BSPLINE_POINTS = (MAX_CONTROL_POINTS - 3) * (NUM_SEGMENTS + 1)
curve_points_bspline = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BSPLINE_POINTS)

# 曲线模式：0 = Bezier, 1 = B-Spline
curve_mode = 0


@ti.func
def de_casteljau_gpu(points, n, t):
    """GPU 上的 De Casteljau 算法"""
    # 使用局部数组
    temp_x = ti.Vector([0.0 for _ in range(MAX_CONTROL_POINTS)])
    temp_y = ti.Vector([0.0 for _ in range(MAX_CONTROL_POINTS)])
    
    for i in range(n):
        temp_x[i] = points[i].x
        temp_y[i] = points[i].y
    
    m = n
    while m > 1:
        for i in range(m - 1):
            temp_x[i] = (1.0 - t) * temp_x[i] + t * temp_x[i + 1]
            temp_y[i] = (1.0 - t) * temp_y[i] + t * temp_y[i + 1]
        m -= 1
    return ti.Vector([temp_x[0], temp_y[0]])


@ti.kernel
def generate_bezier_gpu(n: ti.i32):
    """GPU 并行生成贝塞尔曲线采样点"""
    for idx in range(NUM_SEGMENTS + 1):
        t = idx / NUM_SEGMENTS
        pt = de_casteljau_gpu(control_points_gpu, n, t)
        curve_points_field[idx] = pt


@ti.kernel
def generate_bspline_gpu(n: ti.i32):
    """GPU 并行生成 B样条曲线采样点"""
    # 只在控制点足够时生成
    if n >= 4:
        for seg in range(n - 3):
            p0 = control_points_gpu[seg]
            p1 = control_points_gpu[seg + 1]
            p2 = control_points_gpu[seg + 2]
            p3 = control_points_gpu[seg + 3]
            
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                t2 = t * t
                t3 = t2 * t
                
                # 均匀三次B样条基函数
                N0 = (1 - t) ** 3 / 6.0
                N1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
                N2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
                N3 = t3 / 6.0
                
                x = N0 * p0.x + N1 * p1.x + N2 * p2.x + N3 * p3.x
                y = N0 * p0.y + N1 * p1.y + N2 * p2.y + N3 * p3.y
                
                idx = seg * (NUM_SEGMENTS + 1) + t_int
                curve_points_bspline[idx] = ti.Vector([x, y])


@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def draw_bezier_antialias(n: ti.i32):
    """绘制贝塞尔曲线（带反走样）"""
    sigma = 0.6
    inv_2_sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    for idx in range(n):
        pt = curve_points_field[idx]
        
        x_float = pt[0] * WIDTH
        y_float = pt[1] * HEIGHT
        
        x_min = max(0, ti.cast(x_float - ANTIALIAS_RADIUS, ti.i32))
        x_max = min(WIDTH - 1, ti.cast(x_float + ANTIALIAS_RADIUS, ti.i32))
        y_min = max(0, ti.cast(y_float - ANTIALIAS_RADIUS, ti.i32))
        y_max = min(HEIGHT - 1, ti.cast(y_float + ANTIALIAS_RADIUS, ti.i32))
        
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                dx = ti.cast(i, ti.f32) + 0.5 - x_float
                dy = ti.cast(j, ti.f32) + 0.5 - y_float
                distance = ti.sqrt(dx * dx + dy * dy)
                weight = ti.exp(-distance * distance * inv_2_sigma2)
                
                if weight > 0.01:
                    pixels[i, j] += ti.Vector([0.0, weight, 0.0])


@ti.kernel
def draw_bspline_antialias(n: ti.i32):
    """绘制B样条曲线（带反走样）"""
    sigma = 0.6
    inv_2_sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    for idx in range(n):
        pt = curve_points_bspline[idx]
        
        x_float = pt[0] * WIDTH
        y_float = pt[1] * HEIGHT
        
        x_min = max(0, ti.cast(x_float - ANTIALIAS_RADIUS, ti.i32))
        x_max = min(WIDTH - 1, ti.cast(x_float + ANTIALIAS_RADIUS, ti.i32))
        y_min = max(0, ti.cast(y_float - ANTIALIAS_RADIUS, ti.i32))
        y_max = min(HEIGHT - 1, ti.cast(y_float + ANTIALIAS_RADIUS, ti.i32))
        
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                dx = ti.cast(i, ti.f32) + 0.5 - x_float
                dy = ti.cast(j, ti.f32) + 0.5 - y_float
                distance = ti.sqrt(dx * dx + dy * dy)
                weight = ti.exp(-distance * distance * inv_2_sigma2)
                
                if weight > 0.01:
                    pixels[i, j] += ti.Vector([0.0, weight, 0.0])


@ti.kernel
def draw_bezier_basic(n: ti.i32):
    """绘制贝塞尔曲线（无反走样）"""
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])


@ti.kernel
def draw_bspline_basic(n: ti.i32):
    """绘制B样条曲线（无反走样）"""
    for i in range(n):
        pt = curve_points_bspline[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])


def main():
    global curve_mode
    
    window = ti.ui.Window("Bezier/B-Spline Curve (Optimized)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    
    use_antialias = True
    
    print("=" * 60)
    print("交互说明:")
    print("  - 鼠标左键: 添加控制点")
    print("  - 按键 'c': 清除所有控制点")
    print("  - 按键 'b': 切换曲线模式 (Bezier <-> B-Spline)")
    print("  - 按键 'a': 切换反走样开关")
    print("=" * 60)
    print(f"当前曲线模式: {'B样条' if curve_mode == 1 else '贝塞尔'}")
    print(f"反走样: {'开启' if use_antialias else '关闭'}")
    print("=" * 60)
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    # 更新 GPU 控制点缓冲区
                    np_points = np.full((MAX_CONTROL_POINTS, 2), 0.0, dtype=np.float32)
                    np_points[:len(control_points)] = np.array(control_points, dtype=np.float32)
                    control_points_gpu.from_numpy(np_points)
                    print(f"Added control point {len(control_points)}")
            elif e.key == 'c':
                control_points = []
                # 清空 GPU 控制点缓冲区
                np_points = np.zeros((MAX_CONTROL_POINTS, 2), dtype=np.float32)
                control_points_gpu.from_numpy(np_points)
                print("Canvas cleared.")
            elif e.key == 'b':
                curve_mode = 1 - curve_mode
                mode_name = "B样条" if curve_mode == 1 else "贝塞尔"
                print(f"切换到 {mode_name} 模式")
                if curve_mode == 1 and len(control_points) < 4:
                    print("提示: B样条曲线需要至少4个控制点")
            elif e.key == 'a':
                use_antialias = not use_antialias
                print(f"反走样: {'开启' if use_antialias else '关闭'}")
        
        clear_pixels()
        
        current_count = len(control_points)
        
        # GPU 并行生成曲线
        if current_count >= 2 and curve_mode == 0:  # 贝塞尔
            generate_bezier_gpu(current_count)
            total_points = NUM_SEGMENTS + 1
            if use_antialias:
                draw_bezier_antialias(total_points)
            else:
                draw_bezier_basic(total_points)
                
        elif current_count >= 4 and curve_mode == 1:  # B样条
            generate_bspline_gpu(current_count)
            total_points = (current_count - 3) * (NUM_SEGMENTS + 1)
            if total_points > 0:
                if use_antialias:
                    draw_bspline_antialias(total_points)
                else:
                    draw_bspline_basic(total_points)
        
        canvas.set_image(pixels)
        
        # 绘制控制点和连线
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))
        
        window.show()


if __name__ == '__main__':
    main()