import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.integrate import odeint
import time

# --- 1. 页面基本设置 ---
st.set_page_config(page_title="高级地震液体晃动模拟", layout="wide")
st.title("🌊 高级液体晃动模拟 (Sloshing Pro+) ")
st.markdown("支持 **多种罐体形状** 与 **历史著名地震波** 响应分析。")

# --- 2. 工具函数：生成模拟地震波 ---
def generate_synthetic_quake(name, t, pga_g):
    """
    生成模拟的地震加速度时程 (单位: m/s^2)
    为了保持代码单文件运行，这里使用随机噪声+包络函数+滤波
    来模拟著名地震波的频谱特性和持续时间，而非读取外部CSV。
    """
    g = 9.81
    np.random.seed(42) # 固定种子，保证每次生成的波形一样
    
    # 基础白噪声
    noise = np.random.normal(0, 1, len(t))
    
    if name == "正弦波 (Sine Wave)":
        # 纯正弦波用于理论验证
        freq = 0.6 
        envelope = np.ones_like(t)
        envelope[:int(len(t)*0.1)] = np.linspace(0, 1, int(len(t)*0.1)) # 渐入
        acc_raw = np.sin(2 * np.pi * freq * t) * envelope
        
    elif name == "El Centro (1940)":
        # 特点：持续时间长，频谱丰富
        envelope = np.exp(-0.15 * t) * (t ** 1.5)
        # 模拟低频为主
        acc_raw = np.convolve(noise, np.ones(5)/5, mode='same') * envelope
        
    elif name == "Kobe (1995)":
        # 特点：近场脉冲，猛烈但短促
        envelope = np.exp(-0.5 * (t - 3)**2) * 5 # 脉冲型
        acc_raw = noise * envelope
        
    elif name == "Northridge (1994)":
        # 特点：高频成分多
        envelope = np.exp(-0.2 * t) * t
        acc_raw = noise * envelope
        
    elif name == "Chi-Chi (1999)":
        # 特点：非常长的周期和持续时间
        envelope = (np.sin(t/3) + 1.2) * np.exp(-0.05*t) * (t>1)
        acc_raw = np.convolve(noise, np.ones(15)/15, mode='same') * envelope
        
    else:
        acc_raw = noise

    # 归一化并缩放至目标 PGA (Peak Ground Acceleration)
    current_max = np.max(np.abs(acc_raw))
    if current_max == 0: current_max = 1
    acc_normalized = acc_raw / current_max
    
    return acc_normalized * pga_g * g

# --- 3. 侧边栏：参数设置 ---
with st.sidebar:
    st.header("🏗️ 模型参数")
    
    # --- 形状选择 ---
    shape_type = st.selectbox(
        "罐体形状", 
        ["矩形 (Rectangular)", "圆柱形 (Cylindrical)", "圆环形 (Annular)"]
    )
    
    # 尺寸变量初始化
    L, R, R_in, R_out = 0, 0, 0, 0
    
    if "矩形" in shape_type:
        L = st.number_input("长度 L (m)", 2.0, 10.0, 2.0, step=0.5)
        H = st.number_input("高度 H (m)", 1.0, 10.0, 1.5, step=0.5)
        h_fill = st.slider("液面深度 h (m)", 0.1, H, 1.0)
    elif "圆柱" in shape_type:
        R = st.number_input("半径 R (m)", 0.5, 5.0, 1.0, step=0.1)
        H = st.number_input("高度 H (m)", 1.0, 10.0, 1.5, step=0.5)
        h_fill = st.slider("液面深度 h (m)", 0.1, H, 1.0)
    elif "圆环" in shape_type:
        c1, c2 = st.columns(2)
        R_out = c1.number_input("外半径 Rout", 1.0, 10.0, 2.0)
        R_in = c2.number_input("内半径 Rin", 0.5, 9.0, 1.0)
        H = st.number_input("高度 H (m)", 1.0, 10.0, 1.5)
        h_fill = st.slider("液面深度 h (m)", 0.1, H, 1.0)

    st.markdown("---")
    st.header("📉 地震输入")
    
    # --- 地震波选择 ---
    quake_name = st.selectbox(
        "选择地震波记录",
        ["正弦波 (Sine Wave)", "El Centro (1940)", "Northridge (1994)", "Kobe (1995)", "Chi-Chi (1999)"]
    )
    
    # --- PGA 输入 (使用 g) ---
    pga_g = st.slider("PGA (峰值加速度) [g]", 0.05, 1.0, 0.2, step=0.05)
    st.caption(f"当前峰值加速度: {pga_g * 9.81:.2f} m/s²")
    
    duration = st.slider("模拟时长 (s)", 10, 40, 20)
    
    # --- 预览地震波 ---
    # 预计算地震波并在侧边栏展示
    t_preview = np.linspace(0, duration, 200)
    acc_preview = generate_synthetic_quake(quake_name, t_preview, pga_g)
    
    fig_prev, ax_prev = plt.subplots(figsize=(4, 1.5))
    ax_prev.plot(t_preview, acc_preview / 9.81, color='red', lw=1)
    ax_prev.set_title("输入加速度时程 (g)", fontsize=8)
    ax_prev.axis('off')
    st.pyplot(fig_prev)

    st.markdown("---")
    speed_factor = st.select_slider("动画播放速度", options=["慢速", "正常", "快速"], value="正常")

# --- 4. 物理求解核心 ---
class SloshingSolver:
    def __init__(self, shape, h, **kwargs):
        self.shape = shape
        self.h = h
        self.g = 9.81
        self.xi = 0.03 # 阻尼比
        self.kwargs = kwargs
        self.omega_n = self.calc_freq()
        self.freq_n = self.omega_n / (2 * np.pi)

    def calc_freq(self):
        # 计算固有频率
        if "矩形" in self.shape:
            L = self.kwargs.get('L')
            k = np.pi / L
            return np.sqrt(self.g * k * np.tanh(k * self.h))
        elif "圆柱" in self.shape:
            R = self.kwargs.get('R')
            ep = 1.8412
            return np.sqrt((self.g * ep / R) * np.tanh(ep * self.h / R))
        elif "圆环" in self.shape:
            # 简化近似：基于特征宽度的矩形类比
            w = self.kwargs.get('R_out') - self.kwargs.get('R_in')
            k = np.pi / w
            return np.sqrt(self.g * k * np.tanh(k * self.h))
        return 0

    def solve(self, t_eval, acc_array):
        # acc_array 必须与 t_eval 长度对应
        def equations(y, t):
            eta, v = y
            # 线性插值获取当前时刻加速度
            ground_acc = np.interp(t, t_eval, acc_array)
            
            # 模态参与系数 (简化)
            gamma = 0.83 * np.tanh(np.pi * self.h / 1.0) # 这里的1.0应为特征长度，简化处理
            forcing = -gamma * ground_acc
            
            deta_dt = v
            dv_dt = forcing - 2 * self.xi * self.omega_n * v - (self.omega_n**2) * eta
            return [deta_dt, dv_dt]
        
        sol = odeint(equations, [0,0], t_eval)
        
        # 物理量恢复
        scale = 1.0
        if "矩形" in self.shape: scale = self.kwargs.get('L') / 2
        elif "圆柱" in self.shape: scale = self.kwargs.get('R')
        elif "圆环" in self.shape: scale = (self.kwargs.get('R_out') - self.kwargs.get('R_in'))/2
            
        return sol[:, 0] * scale

# --- 5. 主逻辑 ---

# 初始化参数字典
params = {'L': L, 'R': R, 'R_in': R_in, 'R_out': R_out}
solver = SloshingSolver(shape_type, h_fill, **params)

# 顶部指标栏
c1, c2, c3 = st.columns(3)
c1.metric("自然频率 (Hz)", f"{solver.freq_n:.3f} Hz")
c2.metric("输入PGA (g)", f"{pga_g} g")
c3.metric("总水深", f"{h_fill} m")

if st.button("▶️ 运行模拟", type="primary"):
    
    # 1. 生成时间步和地震波
    dt = 0.05
    t_eval = np.arange(0, duration, dt)
    acc_array = generate_synthetic_quake(quake_name, t_eval, pga_g)
    
    # 2. 求解微分方程
    wave_heights = solver.solve(t_eval, acc_array)
    max_h = np.max(np.abs(wave_heights))
    
    st.info(f"计算完成。最大波高响应: {max_h:.3f} m (相对于静水面)")
    
    # 3. 动画与绘图
    col_anim, col_static = st.columns([3, 2])
    
    plot_ph = col_anim.empty()
    chart_ph = col_static.empty()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 动画循环
    sleep_t = 0.05
    if speed_factor == "慢速": sleep_t = 0.1
    elif speed_factor == "快速": sleep_t = 0.01
        
    prog = st.progress(0)
    
    for i, t in enumerate(t_eval):
        eta = wave_heights[i]
        curr_acc = acc_array[i]
        
        ax.clear()
        
        # --- 绘图逻辑 ---
        if "矩形" in shape_type:
            ax.set_xlim(0, L)
            ax.set_ylim(0, H + 0.5)
            # 容器
            ax.add_patch(Rectangle((0, 0), L, H, fill=False, lw=3))
            # 晃动的水面
            x = np.linspace(0, L, 50)
            y = h_fill + eta * np.cos(np.pi * x / L)
            y = np.clip(y, 0, H)
            verts = [(0,0), (L,0)] + list(zip(L-x, y[::-1]))
            ax.add_patch(Polygon(verts, color='#4F90F0', alpha=0.8))
            
        elif "圆柱" in shape_type:
            ax.set_xlim(-R, R)
            ax.set_ylim(0, H + 0.5)
            ax.add_patch(Rectangle((-R, 0), 2*R, H, fill=False, lw=3))
            ax.plot([0,0], [0,H], 'k--', alpha=0.2)
            
            x = np.linspace(-R, R, 50)
            y = h_fill + eta * (x/R) # 简化线性晃动
            y = np.clip(y, 0, H)
            verts = [(-R,0), (R,0)] + list(zip(x[::-1], y[::-1]))
            ax.add_patch(Polygon(verts, color='#4F90F0', alpha=0.8))
            
        elif "圆环" in shape_type:
            ax.set_xlim(-R_out, R_out)
            ax.set_ylim(0, H + 0.5)
            # 墙体
            ax.vlines([-R_out, -R_in, R_in, R_out], 0, H, color='k', lw=2)
            ax.hlines(0, -R_out, R_out, color='k', lw=2)
            ax.add_patch(Rectangle((-R_in, 0), 2*R_in, H, color='#DDDDDD')) # 内岛
            
            # 左右水面 (反向晃动)
            x_l = np.linspace(-R_out, -R_in, 20)
            y_l = h_fill + eta * (x_l/R_out)
            y_l = np.clip(y_l, 0, H)
            
            x_r = np.linspace(R_in, R_out, 20)
            y_r = h_fill + eta * (x_r/R_out)
            y_r = np.clip(y_r, 0, H)
            
            # 左水
            v_l = [(-R_out,0), (-R_in,0)] + list(zip(x_l[::-1], y_l[::-1]))
            ax.add_patch(Polygon(v_l, color='#4F90F0', alpha=0.8))
            # 右水
            v_r = [(R_in,0), (R_out,0)] + list(zip(x_r[::-1], y_r[::-1]))
            ax.add_patch(Polygon(v_r, color='#4F90F0', alpha=0.8))

        # 标注
        ax.set_title(f"Time: {t:.2f}s | Ground Acc: {curr_acc/9.81:.2f}g")
        ax.text(0, H+0.2, f"Max Wave: {eta:.3f}m", ha='center', fontsize=9, color='blue')
        ax.set_aspect('equal')
        
        plot_ph.pyplot(fig)
        
        # 动态更新右侧曲线
        if i % 5 == 0:
            # 绘制双轴图：波高 vs 地震加速度
            fig2, ax2 = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
            
            # 上图：波高
            ax2[0].plot(t_eval[:i], wave_heights[:i], color='blue')
            ax2[0].set_ylabel("Wave (m)")
            ax2[0].grid(True, alpha=0.3)
            
            # 下图：输入加速度
            ax2[1].plot(t_eval[:i], acc_array[:i]/9.81, color='red', lw=1)
            ax2[1].set_ylabel("Input (g)")
            ax2[1].set_xlabel("Time (s)")
            ax2[1].grid(True, alpha=0.3)
            
            chart_ph.pyplot(fig2)

        prog.progress((i+1)/len(t_eval))
        time.sleep(sleep_t)
        
    st.success("✅ 模拟结束")