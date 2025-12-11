import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.integrate import odeint
from scipy.special import jv  # 贝塞尔函数
import time

# --- 1. 页面基本设置 ---
st.set_page_config(page_title="高级地震液体晃动模拟", layout="wide")
st.title("🌊 高级液体晃动模拟 (Sloshing Pro+ 修正版)")
st.markdown("支持 **多种罐体形状** 与 **历史著名地震波** 响应分析 | ✅ 算法已修正")

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
        L = st.number_input("长度 L (m)", 2.0, 10.0, 4.0, step=0.5)
        H = st.number_input("高度 H (m)", 1.0, 10.0, 3.0, step=0.5)
        h_fill = st.slider("液面深度 h (m)", 0.1, H, 2.0)
    elif "圆柱" in shape_type:
        R = st.number_input("半径 R (m)", 0.5, 5.0, 2.0, step=0.1)
        H = st.number_input("高度 H (m)", 1.0, 10.0, 3.0, step=0.5)
        h_fill = st.slider("液面深度 h (m)", 0.1, H, 2.0)
    elif "圆环" in shape_type:
        c1, c2 = st.columns(2)
        R_out = c1.number_input("外半径 Rout", 1.0, 10.0, 3.0)
        R_in = c2.number_input("内半径 Rin", 0.5, 9.0, 1.5)
        H = st.number_input("高度 H (m)", 1.0, 10.0, 3.0)
        h_fill = st.slider("液面深度 h (m)", 0.1, H, 2.0)

    st.markdown("---")
    st.header("📉 地震输入")
    
    # --- 地震波选择 ---
    quake_name = st.selectbox(
        "选择地震波记录",
        ["正弦波 (Sine Wave)", "El Centro (1940)", "Northridge (1994)", "Kobe (1995)", "Chi-Chi (1999)"]
    )
    
    # --- PGA 输入 (使用 g) ---
    pga_g = st.slider("PGA (峰值加速度) [g]", 0.05, 1.0, 0.3, step=0.05)
    st.caption(f"当前峰值加速度: {pga_g * 9.81:.2f} m/s²")
    
    duration = st.slider("模拟时长 (s)", 10, 40, 20)
    
    # --- 阻尼比选项 ---
    st.markdown("---")
    st.subheader("高级选项")
    use_auto_damping = st.checkbox("自动计算阻尼", value=True)
    if not use_auto_damping:
        manual_damping = st.slider("阻尼比 ξ", 0.005, 0.10, 0.03, step=0.005)
    else:
        manual_damping = None
    
    # --- 预览地震波 ---
    t_preview = np.linspace(0, duration, 200)
    acc_preview = generate_synthetic_quake(quake_name, t_preview, pga_g)
    
    fig_prev, ax_prev = plt.subplots(figsize=(4, 1.5))
    ax_prev.plot(t_preview, acc_preview / 9.81, color='red', lw=1)
    ax_prev.set_title("输入加速度时程 (g)", fontsize=8)
    ax_prev.set_xlabel("时间 (s)", fontsize=7)
    ax_prev.set_ylabel("加速度 (g)", fontsize=7)
    ax_prev.grid(True, alpha=0.3)
    ax_prev.tick_params(labelsize=6)
    st.pyplot(fig_prev)
    plt.close()

    st.markdown("---")
    speed_factor = st.select_slider("动画播放速度", options=["慢速", "正常", "快速"], value="正常")

# --- 4. 物理求解核心（修正版）---
class SloshingSolver:
    def __init__(self, shape, h, auto_damping=True, manual_xi=None, **kwargs):
        self.shape = shape
        self.h = h
        self.g = 9.81
        self.kwargs = kwargs
        
        # 计算固有频率和模态参数
        self.omega_n = self.calc_natural_frequency()
        self.freq_n = self.omega_n / (2 * np.pi)
        
        # 计算阻尼比
        if auto_damping:
            self.xi = self.calc_damping_ratio()
        else:
            self.xi = manual_xi if manual_xi else 0.03
        
        # 计算模态参与系数
        self.gamma = self.calc_modal_participation()

    def calc_natural_frequency(self):
        """计算第一阶固有频率 (rad/s)"""
        if "矩形" in self.shape:
            L = self.kwargs.get('L')
            k = np.pi / L
            omega = np.sqrt(self.g * k * np.tanh(k * self.h))
            return omega
            
        elif "圆柱" in self.shape:
            R = self.kwargs.get('R')
            epsilon_1 = 1.8412  # 第一阶贝塞尔函数根 J'_1(ε) = 0
            omega = np.sqrt((self.g * epsilon_1 / R) * np.tanh(epsilon_1 * self.h / R))
            return omega
            
        elif "圆环" in self.shape:
            # 圆环形容器：使用平均半径的等效圆柱模型
            R_out = self.kwargs.get('R_out')
            R_in = self.kwargs.get('R_in')
            R_mean = (R_out + R_in) / 2
            epsilon_1 = 1.8412
            omega = np.sqrt((self.g * epsilon_1 / R_mean) * np.tanh(epsilon_1 * self.h / R_mean))
            return omega
            
        return 0

    def calc_modal_participation(self):
        """
        计算模态参与系数 γ
        定义：广义坐标与物理波高的关系 η(x,t) = γ * q(t) * φ(x)
        """
        if "矩形" in self.shape:
            L = self.kwargs.get('L')
            k = np.pi / L
            kh = k * self.h
            
            # 矩形容器第一阶模态参与系数
            # 来源：Housner (1963)
            if kh < 0.01:  # 极浅水
                gamma = 1.0
            else:
                gamma = np.tanh(kh) / kh
            return gamma
            
        elif "圆柱" in self.shape:
            R = self.kwargs.get('R')
            epsilon_1 = 1.8412
            x = epsilon_1 * self.h / R
            
            # 圆柱容器第一阶模态参与系数
            # gamma = 2 * J_1(ε) / [ε * J_0(ε)]
            # 使用近似公式避免数值问题
            if x < 0.1:  # 浅水近似
                gamma = 1.0
            else:
                J0 = jv(0, epsilon_1)
                J1 = jv(1, epsilon_1)
                if abs(J0) > 1e-10:
                    gamma = 2 * J1 / (epsilon_1 * J0)
                else:
                    gamma = 0.5  # 深水极限近似
            return gamma
            
        elif "圆环" in self.shape:
            # 圆环形：使用等效圆柱的参与系数
            R_out = self.kwargs.get('R_out')
            R_in = self.kwargs.get('R_in')
            R_mean = (R_out + R_in) / 2
            epsilon_1 = 1.8412
            x = epsilon_1 * self.h / R_mean
            
            if x < 0.1:
                gamma = 1.0
            else:
                J0 = jv(0, epsilon_1)
                J1 = jv(1, epsilon_1)
                if abs(J0) > 1e-10:
                    gamma = 2 * J1 / (epsilon_1 * J0)
                else:
                    gamma = 0.5
            return gamma
            
        return 0.8  # 默认值

    def calc_damping_ratio(self):
        """
        根据容器尺寸和频率估算阻尼比
        考虑：边界层阻尼 + 内部粘性阻尼
        """
        nu = 1e-6  # 水的运动粘度 (m²/s) at 20°C
        
        if "矩形" in self.shape:
            L = self.kwargs.get('L')
            # 基于边界层理论的阻尼估算
            # ξ ≈ 2√(ν/(ωL²))
            if self.omega_n > 0:
                xi_viscous = 2 * np.sqrt(nu / (self.omega_n * L**2))
            else:
                xi_viscous = 0.01
            
            # 加上结构阻尼（经验值）
            xi_structural = 0.005
            xi_total = xi_viscous + xi_structural
            
            # 限制在合理范围内
            return np.clip(xi_total, 0.005, 0.05)
            
        elif "圆柱" in self.shape:
            R = self.kwargs.get('R')
            if self.omega_n > 0:
                xi_viscous = 2 * np.sqrt(nu / (self.omega_n * R**2))
            else:
                xi_viscous = 0.01
            xi_structural = 0.005
            xi_total = xi_viscous + xi_structural
            return np.clip(xi_total, 0.005, 0.05)
            
        elif "圆环" in self.shape:
            R_out = self.kwargs.get('R_out')
            R_in = self.kwargs.get('R_in')
            R_mean = (R_out + R_in) / 2
            if self.omega_n > 0:
                xi_viscous = 2 * np.sqrt(nu / (self.omega_n * R_mean**2))
            else:
                xi_viscous = 0.01
            xi_structural = 0.005
            xi_total = xi_viscous + xi_structural
            return np.clip(xi_total, 0.005, 0.05)
            
        return 0.02  # 默认值

    def solve(self, t_eval, acc_array):
        """
        求解液体晃动运动方程（修正版）
        
        运动方程：
        q̈ + 2ξω_n q̇ + ω_n² q = -Γ * a_g(t)
        
        其中：
        - q: 广义模态坐标 (量纲: 米)
        - Γ: 模态参与系数
        - a_g(t): 地面加速度 (m/s²)
        
        物理波高：η(x,t) = q(t) * φ(x)
        对于矩形：φ(x) = cos(πx/L)
        """
        
        def equations(y, t):
            q, q_dot = y
            
            # 线性插值获取当前时刻地面加速度
            a_ground = np.interp(t, t_eval, acc_array)
            
            # 运动方程右端项
            forcing = -self.gamma * a_ground
            
            # 状态方程
            dq_dt = q_dot
            dq_dot_dt = forcing - 2 * self.xi * self.omega_n * q_dot - (self.omega_n**2) * q
            
            return [dq_dt, dq_dot_dt]
        
        # 初始条件：静止
        y0 = [0.0, 0.0]
        
        # 求解ODE
        solution = odeint(equations, y0, t_eval)
        
        # 广义坐标 q(t)
        q_modal = solution[:, 0]
        
        # 对于矩形容器，波高在中心处的最大值约为 q(t)
        # 对于圆柱容器，需要考虑模态形状函数
        # 这里返回的是广义坐标，单位已经是米
        
        return q_modal
    
    def get_wave_profile(self, q_value, x_positions):
        """
        根据广义坐标计算空间波形
        
        参数：
        - q_value: 当前时刻的广义坐标 (米)
        - x_positions: 空间位置数组
        
        返回：
        - 波高分布 (米)
        """
        if "矩形" in self.shape:
            L = self.kwargs.get('L')
            # 第一阶模态形状：cos(πx/L)
            phi = np.cos(np.pi * x_positions / L)
            return q_value * phi
            
        elif "圆柱" in self.shape:
            R = self.kwargs.get('R')
            # 简化为线性分布（x从-R到R）
            phi = x_positions / R
            return q_value * phi
            
        elif "圆环" in self.shape:
            # 简化处理
            R_out = self.kwargs.get('R_out')
            phi = x_positions / R_out
            return q_value * phi
            
        return np.zeros_like(x_positions)

# --- 5. 主逻辑 ---

# 初始化参数字典
params = {'L': L, 'R': R, 'R_in': R_in, 'R_out': R_out}
solver = SloshingSolver(
    shape_type, 
    h_fill, 
    auto_damping=use_auto_damping,
    manual_xi=manual_damping,
    **params
)

# 顶部指标栏
c1, c2, c3, c4 = st.columns(4)
c1.metric("固有频率", f"{solver.freq_n:.3f} Hz")
c2.metric("阻尼比 ξ", f"{solver.xi:.4f}")
c3.metric("参与系数 γ", f"{solver.gamma:.3f}")
c4.metric("输入PGA", f"{pga_g} g")

# 显示理论信息
with st.expander("📐 查看理论公式"):
    st.markdown(f"""
    ### 当前配置的理论参数
    
    **容器类型**: {shape_type}
    
    **固有频率**: 
    - 角频率 ω_n = {solver.omega_n:.4f} rad/s
    - 自然频率 f_n = {solver.freq_n:.4f} Hz
    - 周期 T = {1/solver.freq_n:.4f} s
    
    **模态参与系数**: γ = {solver.gamma:.4f}
    
    **阻尼比**: ξ = {solver.xi:.5f}
    
    **运动方程**:
    ```
    q̈ + 2ξω_n·q̇ + ω_n²·q = -γ·a_g(t)
    ```
    
    **共振放大倍数** (理论值):
    - Q = 1/(2ξ) ≈ {1/(2*solver.xi):.1f}
    
    **预期最大响应** (线性估算):
    - 若激励频率接近固有频率: η_max ≈ {solver.gamma * pga_g * 9.81 / (2 * solver.xi * solver.omega_n**2):.4f} m
    """)

if st.button("▶️ 运行模拟", type="primary"):
    
    # 1. 生成时间步和地震波
    dt = 0.05
    t_eval = np.arange(0, duration, dt)
    acc_array = generate_synthetic_quake(quake_name, t_eval, pga_g)
    
    # 2. 求解微分方程
    with st.spinner("正在求解运动方程..."):
        modal_coords = solver.solve(t_eval, acc_array)
    
    max_response = np.max(np.abs(modal_coords))
    
    st.success(f"✅ 计算完成！最大模态响应: {max_response:.4f} m")
    
    # 3. 动画与绘图
    col_anim, col_static = st.columns([3, 2])
    
    plot_ph = col_anim.empty()
    chart_ph = col_static.empty()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 动画循环参数
    sleep_t = 0.05
    if speed_factor == "慢速": sleep_t = 0.1
    elif speed_factor == "快速": sleep_t = 0.01
        
    prog = st.progress(0)
    
    # 计算所有时刻的波形（用于后续绘图）
    for i, t in enumerate(t_eval):
        q_current = modal_coords[i]
        curr_acc = acc_array[i]
        
        ax.clear()
        
        # --- 绘图逻辑 ---
        if "矩形" in shape_type:
            ax.set_xlim(0, L)
            ax.set_ylim(0, H + 0.5)
            ax.set_aspect('equal')
            
            # 容器边界
            ax.add_patch(Rectangle((0, 0), L, H, fill=False, lw=3, edgecolor='black'))
            
            # 计算晃动水面
            x = np.linspace(0, L, 100)
            wave = solver.get_wave_profile(q_current, x)
            y_surface = h_fill + wave
            y_surface = np.clip(y_surface, 0, H)
            
            # 绘制水体
            verts = [(0, 0), (L, 0)] + list(zip(x[::-1], y_surface[::-1]))
            ax.add_patch(Polygon(verts, color='#4F90F0', alpha=0.7, edgecolor='blue', lw=1.5))
            
            # 静水面参考线
            ax.plot([0, L], [h_fill, h_fill], 'k--', alpha=0.3, lw=1)
            
        elif "圆柱" in shape_type:
            ax.set_xlim(-R*1.2, R*1.2)
            ax.set_ylim(0, H + 0.5)
            ax.set_aspect('equal')
            
            # 容器边界
            ax.add_patch(Rectangle((-R, 0), 2*R, H, fill=False, lw=3, edgecolor='black'))
            ax.plot([0, 0], [0, H], 'k--', alpha=0.2, lw=1)
            
            # 计算晃动水面
            x = np.linspace(-R, R, 100)
            wave = solver.get_wave_profile(q_current, x)
            y_surface = h_fill + wave
            y_surface = np.clip(y_surface, 0, H)
            
            # 绘制水体
            verts = [(-R, 0), (R, 0)] + list(zip(x[::-1], y_surface[::-1]))
            ax.add_patch(Polygon(verts, color='#4F90F0', alpha=0.7, edgecolor='blue', lw=1.5))
            
            # 静水面参考线
            ax.plot([-R, R], [h_fill, h_fill], 'k--', alpha=0.3, lw=1)
            
        elif "圆环" in shape_type:
            ax.set_xlim(-R_out*1.1, R_out*1.1)
            ax.set_ylim(0, H + 0.5)
            ax.set_aspect('equal')
            
            # 容器边界
            ax.vlines([-R_out, -R_in, R_in, R_out], 0, H, color='black', lw=2)
            ax.hlines(0, -R_out, R_out, color='black', lw=2)
            ax.add_patch(Rectangle((-R_in, 0), 2*R_in, H, color='#CCCCCC', alpha=0.5))
            
            # 左侧水体
            x_l = np.linspace(-R_out, -R_in, 50)
            wave_l = solver.get_wave_profile(q_current, x_l)
            y_l = h_fill + wave_l
            y_l = np.clip(y_l, 0, H)
            
            verts_l = [(-R_out, 0), (-R_in, 0)] + list(zip(x_l[::-1], y_l[::-1]))
            ax.add_patch(Polygon(verts_l, color='#4F90F0', alpha=0.7, edgecolor='blue', lw=1.5))
            
            # 右侧水体
            x_r = np.linspace(R_in, R_out, 50)
            wave_r = solver.get_wave_profile(q_current, x_r)
            y_r = h_fill + wave_r
            y_r = np.clip(y_r, 0, H)
            
            verts_r = [(R_in, 0), (R_out, 0)] + list(zip(x_r[::-1], y_r[::-1]))
            ax.add_patch(Polygon(verts_r, color='#4F90F0', alpha=0.7, edgecolor='blue', lw=1.5))
            
            # 静水面参考线
            ax.plot([-R_out, -R_in], [h_fill, h_fill], 'k--', alpha=0.3, lw=1)
            ax.plot([R_in, R_out], [h_fill, h_fill], 'k--', alpha=0.3, lw=1)

        # 标注信息
        ax.set_title(f"时间: {t:.2f}s | 地面加速度: {curr_acc/9.81:.3f}g", fontsize=10)
        ax.text(0.02, 0.98, f"模态坐标: {q_current:.4f}m\n最大波高: {max_response:.4f}m", 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel("位置 (m)")
        ax.set_ylabel("高度 (m)")
        ax.grid(True, alpha=0.2)
        
        plot_ph.pyplot(fig)
        
        # 动态更新右侧曲线（每5帧更新一次以提高性能）
        if i % 5 == 0:
            fig2, ax2 = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
            
            # 上图：波高响应
            ax2[0].plot(t_eval[:i+1], modal_coords[:i+1], color='blue', lw=1.5)
            ax2[0].axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
            ax2[0].set_ylabel("模态坐标 q (m)", fontsize=9)
            ax2[0].set_title("液体晃动响应", fontsize=10)
            ax2[0].grid(True, alpha=0.3)
            ax2[0].set_ylim(-max_response*1.2, max_response*1.2)
            
            # 下图：输入加速度
            ax2[1].plot(t_eval[:i+1], acc_array[:i+1]/9.81, color='red', lw=1)
            ax2[1].axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
            ax2[1].set_ylabel("加速度 (g)", fontsize=9)
            ax2[1].set_xlabel("时间 (s)", fontsize=9)
            ax2[1].set_title("地震输入", fontsize=10)
            ax2[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            chart_ph.pyplot(fig2)
            plt.close(fig2)

        prog.progress((i+1)/len(t_eval))
        time.sleep(sleep_t)
    
    plt.close(fig)
    st.success("✅ 模拟结束")
    
    # 最终结果统计
    st.markdown("---")
    st.subheader("📊 结果统计")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("最大波高响应", f"{max_response:.4f} m")
        st.caption(f"相对水深: {max_response/h_fill*100:.1f}%")
    
    with col2:
        max_acc_idx = np.argmax(np.abs(acc_array))
        st.metric("峰值地面加速度", f"{np.max(np.abs(acc_array))/9.81:.3f} g")
        st.caption(f"发生在 t={t_eval[max_acc_idx]:.2f}s")
    
    with col3:
        # 计算能量相关指标
        energy_ratio = max_response / (pga_g * 9.81 / solver.omega_n**2)
        st.metric("动力放大系数", f"{energy_ratio:.2f}")
        st.caption(f"理论值: ~{1/(2*solver.xi):.1f}")
    
    # 频谱分析
    st.markdown("---")
    st.subheader("🔬 频谱分析")
    
    # FFT分析
    from scipy.fft import fft, fftfreq
    
    N = len(modal_coords)
    T_sample = t_eval[1] - t_eval[0]
    yf = fft(modal_coords)
    xf = fftfreq(N, T_sample)[:N//2]
    
    # 只取正频率部分
    power = 2.0/N * np.abs(yf[0:N//2])
    
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 3))
    
    # 响应频谱
    ax3a.plot(xf, power, 'b-', lw=1)
    ax3a.axvline(solver.freq_n, color='r', linestyle='--', lw=2, label=f'固有频率 {solver.freq_n:.3f} Hz')
    ax3a.set_xlabel('频率 (Hz)')
    ax3a.set_ylabel('幅值')
    ax3a.set_title('响应频谱')
    ax3a.set_xlim(0, 5)
    ax3a.grid(True, alpha=0.3)
    ax3a.legend()
    
    # 输入加速度频谱
    yf_input = fft(acc_array/9.81)
    power_input = 2.0/N * np.abs(yf_input[0:N//2])
    ax3b.plot(xf, power_input, 'r-', lw=1)
    ax3b.axvline(solver.freq_n, color='r', linestyle='--', lw=2, label=f'固有频率 {solver.freq_n:.3f} Hz')
    ax3b.set_xlabel('频率 (Hz)')
    ax3b.set_ylabel('幅值')
    ax3b.set_title('输入加速度频谱')
    ax3b.set_xlim(0, 5)
    ax3b.grid(True, alpha=0.3)
    ax3b.legend()
    
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    
    # 时程对比图
    st.markdown("---")
    st.subheader("📈 完整时程曲线")
    
    fig4, ax4 = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    
    # 波高时程
    ax4[0].plot(t_eval, modal_coords, 'b-', lw=1.5, label='液体晃动响应')
    ax4[0].axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
    ax4[0].axhline(max_response, color='r', lw=1, ls=':', alpha=0.5, label=f'最大值: {max_response:.4f}m')
    ax4[0].axhline(-max_response, color='r', lw=1, ls=':', alpha=0.5)
    ax4[0].set_ylabel('模态坐标 q (m)', fontsize=10)
    ax4[0].set_title('液体晃动响应时程', fontsize=11)
    ax4[0].grid(True, alpha=0.3)
    ax4[0].legend(loc='upper right')
    
    # 加速度时程
    ax4[1].plot(t_eval, acc_array/9.81, 'r-', lw=1, label='地震加速度输入')
    ax4[1].axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
    ax4[1].axhline(pga_g, color='darkred', lw=1, ls=':', alpha=0.5, label=f'PGA: {pga_g}g')
    ax4[1].axhline(-pga_g, color='darkred', lw=1, ls=':', alpha=0.5)
    ax4[1].set_ylabel('加速度 (g)', fontsize=10)
    ax4[1].set_xlabel('时间 (s)', fontsize=10)
    ax4[1].set_title('地震输入时程', fontsize=11)
    ax4[1].grid(True, alpha=0.3)
    ax4[1].legend(loc='upper right')
    
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)
    
    # 下载数据
    st.markdown("---")
    st.subheader("💾 导出数据")
    
    import pandas as pd
    
    df_results = pd.DataFrame({
        '时间(s)': t_eval,
        '模态坐标(m)': modal_coords,
        '地震加速度(g)': acc_array / 9.81,
        '地震加速度(m/s²)': acc_array
    })
    
    csv = df_results.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 下载结果CSV文件",
        data=csv,
        file_name=f'sloshing_results_{quake_name}_{pga_g}g.csv',
        mime='text/csv',
    )
    
    # 参数总结
    with st.expander("📋 计算参数总结"):
        st.markdown(f"""
        ### 模拟配置参数
        
        **容器几何**
        - 形状: {shape_type}
        - 特征尺寸: {L if '矩形' in shape_type else (R if '圆柱' in shape_type else f'Rout={R_out}, Rin={R_in}')} m
        - 总高度: {H} m
        - 液面深度: {h_fill} m
        - 充液率: {h_fill/H*100:.1f}%
        
        **动力学参数**
        - 固有频率: {solver.freq_n:.4f} Hz
        - 固有周期: {1/solver.freq_n:.4f} s
        - 角频率: {solver.omega_n:.4f} rad/s
        - 阻尼比: {solver.xi:.5f}
        - 模态参与系数: {solver.gamma:.4f}
        
        **地震输入**
        - 地震波: {quake_name}
        - PGA: {pga_g} g ({pga_g*9.81:.2f} m/s²)
        - 持续时间: {duration} s
        - 时间步长: {dt} s
        
        **计算结果**
        - 最大响应: {max_response:.4f} m
        - 相对水深比: {max_response/h_fill*100:.2f}%
        - 动力放大系数: {max_response / (pga_g * 9.81 / solver.omega_n**2):.2f}
        - 理论放大系数: {1/(2*solver.xi):.2f}
        
        **算法说明**
        - 模型: 单模态线性晃动理论
        - 求解器: scipy.integrate.odeint (LSODA)
        - 频率计算: 基于势流理论
        - 阻尼模型: 边界层阻尼 + 结构阻尼
        """)

# --- 底部说明 ---
st.markdown("---")
st.markdown("""
### 📚 理论基础

本程序基于**线性势流理论**求解液体晃动问题：

1. **运动方程**: 单自由度振子模型
   ```
   q̈ + 2ξω_n·q̇ + ω_n²·q = -γ·a_g(t)
   ```

2. **固有频率** (矩形容器):
   ```
   ω_n = √(g·k·tanh(k·h)), k = π/L
   ```

3. **模态参与系数** (矩形):
   ```
   γ = tanh(k·h) / (k·h)
   ```

4. **适用范围**:
   - 小幅晃动 (η/h < 0.1)
   - 单模态主导
   - 无粘性流体假设

5. **主要修正** (相比原版):
   - ✅ 修正模态参与系数计算
   - ✅ 移除不合理的响应缩放
   - ✅ 添加自适应阻尼估算
   - ✅ 改进贝塞尔函数计算

### ⚠️ 使用注意事项

- 当波高超过水深的10%时，应考虑非线性效应
- 浅水情况 (h/L < 0.1) 结果可能偏差较大
- 圆环形容器使用等效模型，精度相对较低
- 真实地震波建议使用实测数据替代模拟波形

### 🔗 参考文献

- Housner, G.W. (1963). "The Dynamic Behavior of Water Tanks"
- Ibrahim, R.A. (2005). "Liquid Sloshing Dynamics"
- Faltinsen, O.M. (1974). "A Nonlinear Theory of Sloshing"

---

**版本**: v2.0 (算法修正版) | **开发**: Streamlit + NumPy + SciPy | **许可**: MIT
""")