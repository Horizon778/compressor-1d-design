"""
Y-7-v2 算例（40000 r/min，40000 Pa 进口，ṁ≈0.374 kg/s）。

两个用途：
  DESIGN  —— 用当前工况重新做一遍一维设计，看"应该长什么样"
  MACHINE —— 你现在实际在算的那台（single_imp-stator1-volute_003），用来体检

MACHINE 里凡是标 ?? 的，是从 CFturbo 界面抄下来填的，没填就留 0，
程序会把对应判据标成 N/A，不会瞎猜。
"""

from ..audit import CFData, Geometry, Machine
from ..design import Case

# ============================================================
# 一维设计（正问题）
# ============================================================
# psi / phi2 / eta_tt 用你 003 算例反算出来的值，这样设计出来的 r2 应该
# 跟你的 99.32 mm 对得上 —— 对不上就说明一维模型跟你的实际几何有偏差，
# 那正是要找的东西。
DESIGN = Case(
    mdot=0.3738,
    n_rpm=40000.0,
    PR_tt=2.252,
    T0_in=300.0,
    p0_in=40000.0,
    eta_tt=0.764,          # 实测 Δh0=102.9 kJ/kg 反算
    psi=0.594,             # 实测
    phi2=0.257,            # 实测
    z_main=9,
    z_split=9,
    Kd=0.40,
    # 后置部件用推荐值，不用你现在的 1.0003 / 1.462
    r3_over_r2=1.10,
    r4_over_r3=1.32,
    theta_target_deg=9.0,
    beta4A_offset_deg=5.0,   # 出口偏离角
    M5_target=0.10,
)

# ============================================================
# 实际机器（逆向体检）
# ============================================================
GEOM = Geometry(
    r1t=0.0517,        # 进口延伸段外径，若叶轮 LE 的 shroud 半径不同请改
    r1h=0.0207,        # ?? 估的（Kd=0.40），请从 CFturbo 或 .gtm 量
    r2=0.09932,
    b2=0.00859,
    beta1A_deg=0.0,    # ?? CFturbo -> Blade properties -> Metal angles
    beta2A_deg=0.0,    # ?? 同上
    z_main=9,
    z_split=9,
    r3=0.09935,
    r4=0.14520,
    b3=0.00859,
    b4=0.00859,
    z_diff=17,
    beta3A_deg=None,   # ?? 扩压器叶片进口角
    beta4A_deg=None,   # ?? 扩压器叶片出口角
    A_vol_in=0.007836,
    A_vol_out=0.018412,
    r_tongue=0.0,      # ?? 蜗舌半径
)

CFD = CFData(
    mdot_in=0.3738,
    mdot_out=0.3593,
    p0_2=105_340.0,
    p_2=64_470.0,
    T0_2=401.47,
    T_2=None,          # 只给了 M=0.86，由程序按 M 反算
    M_2=0.86,
    p0_3=94_570.0,
    p_3=86_600.0,
    T0_3=401.41,
    p0_5=90_010.0,
    p_5=89_610.0,
    T0_5=408.46,
    M_5=0.08,
)

MACHINE = Machine(
    name="Y-7-v2 / single_imp-stator1-volute_003 (iter 2100)",
    n_rpm=40000.0,
    mdot=0.3738,
    T0_in=300.0,
    p0_in=40000.0,
    geom=GEOM,
    cfd=CFD,
)

CASES = {
    "y7v2": (DESIGN, MACHINE),
}
