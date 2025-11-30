import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

# --- 1. 设置页面布局 ---
st.set_page_config(page_title="地震液体晃动模拟", layout="wide")

st.title("🌊 罐体液体晃动模拟 (Sloshing Demo)")
st.markdown("基于线性波浪理论与Housner模型演示地震下的液面响应。")

# --- 2. 侧边栏：参数设置 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    st.subheader("罐体尺寸")
    L = st.number_input("长度 L (m)", value=2.0, min_value=0.5, step=0.1)
    H = st.number_input("高度 H (m)", value=1.5, min_value=0.5, step=0.1)
    h_fill = st.slider("液面深度 h (m)", 0.1, H, 1.0)
    
    st.subheader("地震激励")
    acc_amp = st.slider("地震幅值 (m/s²)", 0.1, 5.0, 1.0)
    freq_exc = st.slider("地震频率 (Hz)", 0.1, 2.0, 0.6)
    
    st.subheader("模拟设置")
    duration = st.slider("模拟时长 (s)", 5, 30, 20)
    speed_factor = st.select_slider("动画速度", options=["慢速", "正常", "快速"], value="正常")

# --- 3. 物理计算核心 (后端逻辑) ---
class SloshingSolver:
    def __init__(self, L, h):
        self.L = L
        self.h = h
        self.g = 9.81
        self.xi = 0.05 # 阻尼比
        
        # 自然频率计算
        if h > 0 and L > 0:
            term = np.tanh(np.pi * h / L)
            self.omega_n = np.sqrt((self.g * np.pi / L) * term)
            self.freq_n = self.omega_n / (2 * np.pi)
        else:
            self.omega_n = 0
            self.freq_n = 0

    def get_wave(self, t, acc, freq):
        omega_exc = 2 * np.pi * freq
        # 简单的包络函数，让地震波逐渐开始
        envelope = np.minimum(t / 2.0, 1.0)
        return acc * envelope * np.sin(omega_exc * t)

    def solve(self, t_eval, acc, freq):
        def equations(y, t):
            eta, v = y
            ground_acc = self.get_wave(t, acc, freq)
            gamma = 0.83 * np.tanh(np.pi * self.h / self.L)
            forcing = -gamma * ground_acc
            deta_dt = v
            dv_dt = forcing - 2 * self.xi * self.omega_n * v - (self.omega_n**2) * eta
            return [deta_dt, dv_dt]
        
        y0 = [0.0, 0.0]
        sol = odeint(equations, y0, t_eval)
        return sol[:, 0] * (self.L / 2) # 返回波高历史

# --- 4. 主界面逻辑 ---

# 实例化并计算自然频率
solver = SloshingSolver(L, h_fill)
st.metric(label="一阶自然频率 (共振点)", value=f"{solver.freq_n:.3f} Hz", 
          delta=f"当前激励: {freq_exc} Hz", delta_color="inverse")

if abs(solver.freq_n - freq_exc) < 0.1:
    st.warning("⚠️ 警告：激励频率接近自然频率，将发生共振！")

# 占位符，用于动画和图表
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🌊 实时动画")
    plot_placeholder = st.empty()

with col2:
    st.subheader("📈 波高时程曲线")
    chart_placeholder = st.empty()

# --- 5. 运行按钮 ---
if st.button("▶️ 开始模拟", type="primary"):
    # 预计算数据
    dt = 0.05
    t_eval = np.arange(0, duration, dt)
    wave_heights = solver.solve(t_eval, acc_amp, freq_exc)
    
    # 准备绘图对象
    fig, ax = plt.subplots(figsize=(6, 4))
    x_surface = np.linspace(0, L, 50)
    
    # 定义速度
    sleep_time = 0.05
    if speed_factor == "慢速": sleep_time = 0.1
    elif speed_factor == "快速": sleep_time = 0.01

    # 动画循环
    progress_bar = st.progress(0)
    
    for i, t in enumerate(t_eval):
        # 1. 更新动画帧
        eta = wave_heights[i]
        
        ax.clear()
        ax.set_xlim(0, L)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        
        # 绘制罐体
        rect = plt.Rectangle((0, 0), L, H, fill=False, lw=3, color='black')
        ax.add_patch(rect)
        
        # 绘制水面
        y_surface = h_fill + eta * np.cos(np.pi * x_surface / L)
        y_surface = np.clip(y_surface, 0, H) # 防止画出边界
        
        verts = [(0, 0), (L, 0)]
        verts.extend(list(zip(L - x_surface, y_surface[::-1])))
        poly = plt.Polygon(verts, color='skyblue', alpha=0.7)
        ax.add_patch(poly)
        
        ax.text(0.05 * L, 0.9 * H, f"Time: {t:.1f}s")
        ax.set_title(f"Wave Height: {eta:.3f} m")
        
        # 将Matplotlib图显示在Streamlit占位符中
        plot_placeholder.pyplot(fig)
        
        # 2. 更新右侧曲线图 (动态显示一段历史)
        # 为了性能，每5帧更新一次曲线，或者显示全部
        if i % 5 == 0:
            chart_placeholder.line_chart(wave_heights[:i+1])
            
        # 更新进度条
        progress_bar.progress((i + 1) / len(t_eval))
        
        # 控制速度
        time.sleep(sleep_time)

    st.success("模拟结束！")