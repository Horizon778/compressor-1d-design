"""
逆向体检：给你一台已经建好（或已经算完）的机器，反算所有设计参数并逐条对判据。

这是本工具最重要的部分。CFturbo 的 Preliminary Design 会把 r1、r2、b2、beta2A、
z 全部"自动算好"给你，但它算的依据是内部经验库 + 你输入的那三个数，
既不告诉你它选了什么经验系数，也不保证这些数在你的工况下合理。

做法：把几何量出来（从 CFturbo 界面抄、从 .gtm 量、或从 CFX 网格量都行），
喂进 Geometry；有 CFD 结果就再喂 CFD。程序把所有无量纲量反算出来，
跟判据表逐条对，超出范围的地方直接标 FAIL。

两条使用原则：
  1) 几何判据（k3、ky、b2/r2、叶片数）不需要 CFD 就能查，先把这些查完；
  2) 气动判据（M、psi、slip）需要 CFD 的截面平均值，不要用局部峰值冒充平均值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .gas import IdealGas
from . import criteria as cr


# ============================================================
# 输入
# ============================================================
@dataclass
class Geometry:
    """实测几何。长度单位 m，角度单位 degree（从径向测量）。"""

    # 叶轮
    r1t: float = 0.0          # 叶尖进口半径（shroud 侧）
    r1h: float = 0.0          # 轮毂进口半径
    r2: float = 0.0           # 出口半径
    b2: float = 0.0           # 出口宽度
    beta1A_deg: float = 0.0   # 叶片进口安装角
    beta2A_deg: float = 0.0   # 叶片出口安装角
    z_main: int = 0
    z_split: int = 0
    t_blade: float = 0.0015   # 叶片平均厚度

    # 扩压器
    r3: float = 0.0
    r4: float = 0.0
    b3: float = 0.0
    b4: float = 0.0
    z_diff: int = 0
    beta3A_deg: float | None = None
    beta4A_deg: float | None = None

    # 蜗壳
    A_vol_in: float = 0.0
    A_vol_out: float = 0.0
    r_tongue: float = 0.0

    @property
    def z_total(self) -> int:
        return self.z_main + self.z_split


@dataclass
class CFData:
    """
    CFD 截面平均值（massFlowAve），单位 Pa / K / kg/s。
    留 None 的项会被跳过，不参与判据。
    """
    mdot_in: float | None = None
    mdot_out: float | None = None
    # 叶轮出口 / 扩压器进口
    p0_2: float | None = None
    p_2: float | None = None
    T0_2: float | None = None
    T_2: float | None = None
    M_2: float | None = None       # 绝对马赫
    # 扩压器出口
    p0_3: float | None = None
    p_3: float | None = None
    T0_3: float | None = None
    T_3: float | None = None
    # 蜗壳出口
    p0_5: float | None = None
    p_5: float | None = None
    T0_5: float | None = None
    T_5: float | None = None
    M_5: float | None = None


@dataclass
class Machine:
    name: str = ""
    n_rpm: float = 0.0
    mdot: float = 0.0
    T0_in: float = 0.0
    p0_in: float = 0.0
    gas: IdealGas = field(default_factory=IdealGas)
    geom: Geometry = field(default_factory=Geometry)
    cfd: CFData = field(default_factory=CFData)

    @property
    def omega(self) -> float:
        return self.n_rpm * 2.0 * math.pi / 60.0


# ============================================================
# 反算
# ============================================================
@dataclass
class Derived:
    U2: float = 0.0
    dh0: float = 0.0
    psi: float = 0.0
    phi2: float = 0.0
    c_u2: float = 0.0
    c_r2: float = 0.0
    c2: float = 0.0
    rho2: float = 0.0
    beta2_flow_deg: float = 0.0
    M2_abs: float = 0.0
    M2_rel: float = 0.0
    slip_cfd: float = 0.0
    slip_wiesner: float = 0.0
    # 进口
    c_m1: float = 0.0
    U1t: float = 0.0
    w1: float = 0.0
    w2: float = 0.0
    M_w1: float = 0.0
    beta1_flow_deg: float = 0.0
    # 比值
    b2_over_r2: float = 0.0
    hub_tip: float = 0.0
    k3: float = 0.0
    ky: float = 0.0
    ak: float = 0.0
    blockage: float = 0.0
    solidity_imp: float = 0.0
    solidity_diff: float = 0.0
    theta_diff: float = 0.0
    # 气动
    M3: float = 0.0
    alpha3_deg: float = 0.0
    alpha4_deg: float = 0.0
    # 自洽性
    eta_p_imp: float = float("nan")
    dT0_diff: float | None = None
    dT0_vol: float | None = None
    a4_estimated: bool = False   # alpha4 是否为估算值


def derive(m: Machine) -> Derived:
    g = m.gas
    q = m.geom
    d = Derived()

    d.U2 = m.omega * q.r2

    # ---- 能量头：只认叶轮出口 ----
    # 不能用蜗壳出口的 T0：那个位置已经叠加了质量不平衡的能量残差
    # （本算例里 T0_5 比 T0_3 高 7 K），拿它算 psi 会虚高 5%。
    if m.cfd.T0_2 is not None:
        d.dh0 = g.dh_from_T0(m.T0_in, m.cfd.T0_2)
    elif m.cfd.T0_5 is not None:
        d.dh0 = g.dh_from_T0(m.T0_in, m.cfd.T0_5)
    else:
        d.dh0 = float("nan")

    if d.U2 > 0 and not math.isnan(d.dh0):
        d.psi = d.dh0 / d.U2 ** 2
        d.c_u2 = d.dh0 / d.U2
    else:
        d.c_u2 = float("nan")

    # ---- 出口密度与径向速度 ----
    # 静温优先用实测；只给了马赫数时由 T0/M 反算
    T2_eff = m.cfd.T_2
    if T2_eff is None and m.cfd.T0_2 is not None and m.cfd.M_2 is not None:
        T2_eff = g.T_from_T0_M(m.cfd.T0_2, m.cfd.M_2)

    if m.cfd.p_2 and T2_eff:
        d.rho2 = g.rho(m.cfd.p_2, T2_eff)
    elif m.cfd.p0_2 and m.cfd.T0_2:
        # 粗估：按等熵
        d.rho2 = g.rho(m.cfd.p0_2, m.cfd.T0_2)
    else:
        d.rho2 = float("nan")

    if q.r2 > 0 and q.b2 > 0 and d.rho2 > 0:
        A2 = 2.0 * math.pi * q.r2 * q.b2
        d.c_r2 = m.mdot / (d.rho2 * A2)
        d.phi2 = d.c_r2 / d.U2 if d.U2 > 0 else float("nan")
        d.c2 = math.hypot(d.c_u2, d.c_r2) if not math.isnan(d.c_u2) else float("nan")
        if T2_eff:
            a2 = g.a(T2_eff)
            d.M2_abs = d.c2 / a2
            w_u2 = d.U2 - d.c_u2
            d.w2 = math.hypot(d.c_r2, w_u2)
            d.M2_rel = d.w2 / a2
            d.beta2_flow_deg = math.degrees(math.atan2(d.c_r2, w_u2))
            # 滑移系数：实际 c_u2 与"无限叶片"理论值之比
            if q.beta2A_deg > 0:
                c_u_inf = d.U2 - d.c_r2 / math.tan(math.radians(q.beta2A_deg))
                if c_u_inf > 1e-6:
                    d.slip_cfd = d.c_u2 / c_u_inf
                from .design import wiesner_slip
                d.slip_wiesner = wiesner_slip(
                    math.radians(q.beta2A_deg), q.z_total or 18,
                    q.r1t / q.r2 if q.r2 > 0 else 0.0)
            else:
                # beta2A 没填就不知道叶片几何，滑移无法反算，
                # 宁可报 N/A 也不要给个 0 让人以为通过
                d.slip_cfd = float("nan")
                d.slip_wiesner = float("nan")

    # ---- 进口 ----
    if q.r1t > 0 and q.r1h > 0:
        A1 = math.pi * (q.r1t ** 2 - q.r1h ** 2)
        d.hub_tip = q.r1h / q.r1t
        d.U1t = m.omega * q.r1t
        # 进口密度：用总状态近似（M_w1 < 0.9 时误差 < 8%）
        rho1 = g.rho(m.p0_in, m.T0_in)
        d.c_m1 = m.mdot / (rho1 * A1)
        d.w1 = math.hypot(d.c_m1, d.U1t)
        # 音速必须用静温，用 T0 会把 M_w1 低估约 4%
        T1 = m.T0_in - d.c_m1 ** 2 / (2.0 * g.cp)
        d.M_w1 = d.w1 / g.a(T1)
        d.beta1_flow_deg = math.degrees(math.atan2(d.c_m1, d.U1t))
        # 堵塞
        b1 = q.r1t - q.r1h
        A_blk = q.z_main * q.t_blade * b1 / max(
            math.sin(math.radians(q.beta1A_deg or 45.0)), 0.2)
        d.blockage = 1.0 - min(A_blk / A1, 0.5)

    # ---- 几何比值 ----
    d.b2_over_r2 = q.b2 / q.r2 if q.r2 > 0 else float("nan")
    if q.r3 > 0 and q.r2 > 0:
        d.k3 = q.r3 / q.r2
    if q.r4 > 0 and q.r3 > 0:
        d.ky = q.r4 / q.r3
    if q.b3 > 0 and q.b4 > 0:
        d.ak = q.b4 / q.b3

    # 叶轮稠度（主叶片）
    if q.r1t > 0 and q.r2 > 0 and q.z_main > 0:
        b1 = q.beta1A_deg or d.beta1_flow_deg or 40.0
        b2 = q.beta2A_deg or 40.0
        bm = math.radians(0.5 * (b1 + b2))
        L = (q.r2 - q.r1t) / max(math.sin(bm), 0.2)
        d.solidity_imp = q.z_main * L / (2.0 * math.pi * 0.5 * (q.r1t + q.r2))

    # ---- 扩压器进口气动（自由涡近似） ----
    if q.r3 > 0 and q.r2 > 0 and not math.isnan(d.c_u2) and d.c_r2 == d.c_r2:
        cu3 = d.c_u2 * q.r2 / q.r3
        cr3 = d.c_r2 * q.r2 / q.r3
        d.alpha3_deg = math.degrees(math.atan2(cr3, cu3))
        if T2_eff:
            d.M3 = math.hypot(cu3, cr3) / g.a(T2_eff)

    # 出口气流角：有实测叶片角就用，没有就按"典型偏离 5 度"估，
    # 并把这个假设写进判据的 note，避免让人以为是实测值
    a4_estimated = False
    if q.beta4A_deg:
        d.alpha4_deg = q.beta4A_deg
    elif d.alpha3_deg > 0:
        d.alpha4_deg = d.alpha3_deg - 5.0
        a4_estimated = True

    if d.alpha3_deg > 0 and d.alpha4_deg is not None:
        b3 = q.beta3A_deg if q.beta3A_deg else d.alpha3_deg
        b4 = q.beta4A_deg if q.beta4A_deg else d.alpha4_deg
        if q.z_diff > 0 and q.r4 > q.r3 > 0:
            am = math.radians(0.5 * (b3 + b4))
            L = (q.r4 - q.r3) / max(math.cos(am), 0.2)
            rm = 0.5 * (q.r3 + q.r4)
            d.solidity_diff = q.z_diff * L / (2.0 * math.pi * rm)
            o3 = 2 * math.pi * q.r3 * math.cos(math.radians(b3)) / q.z_diff - q.t_blade
            o4 = 2 * math.pi * q.r4 * math.cos(math.radians(b4)) / q.z_diff - q.t_blade
            d.theta_diff = math.degrees(2.0 * math.atan((o4 - o3) / (2.0 * L)))
    d.a4_estimated = a4_estimated

    # ---- 自洽性 ----
    if all(v is not None for v in (m.cfd.p0_2, m.cfd.T0_2, m.cfd.p_2, T2_eff)):
        d.eta_p_imp = g.eta_p_from_states(
            m.cfd.p0_2, m.cfd.T0_2, m.cfd.p_2, T2_eff)

    if m.cfd.T0_3 is not None and m.cfd.T0_2 is not None:
        d.dT0_diff = m.cfd.T0_3 - m.cfd.T0_2
    if m.cfd.T0_5 is not None and m.cfd.T0_3 is not None:
        d.dT0_vol = m.cfd.T0_5 - m.cfd.T0_3

    return d


# ============================================================
# 体检
# ============================================================
def audit(m: Machine) -> tuple[Derived, list]:
    d = derive(m)
    q, f = m.geom, m.cfd

    checks = []
    checks += cr.impeller_checks(
        M_w1=_ok(d.M_w1),
        w2_over_w1=_ok(d.w2 / d.w1 if d.w1 > 0 else None),
        psi=_ok(d.psi),
        phi2=_ok(d.phi2),
        b2_over_r2=_ok(d.b2_over_r2),
        beta2A_deg=q.beta2A_deg or None,
        beta2_flow_deg=_ok(d.beta2_flow_deg),
        slip_wiesner=_ok(d.slip_wiesner),
        slip_cfd=_ok(d.slip_cfd),
        hub_tip_ratio=_ok(d.hub_tip),
        blockage_inlet=_ok(d.blockage),
        M2_rel=_ok(d.M2_rel),
    )
    checks += cr.diffuser_checks(
        k3=_ok(d.k3),
        ky=_ok(d.ky),
        M3=_ok(d.M3),
        theta_deg=_ok(d.theta_diff),
        ak=_ok(d.ak),
        z_diff=q.z_diff or None,
        z_imp=q.z_total or None,
        solidity=_ok(d.solidity_diff),
        alpha3_deg=_ok(d.alpha3_deg),
        alpha4_deg=_ok(d.alpha4_deg),
        theta_is_estimate=d.a4_estimated,
    )
    checks += cr.volute_checks(
        M5=_ok(f.M_5),
        A_out=q.A_vol_out or None,
        A_in=q.A_vol_in or None,
        area_ratio=_ok(q.A_vol_out / q.A_vol_in if q.A_vol_in > 0 else None),
        tongue_gap_ratio=_ok(q.r_tongue / q.r2 if q.r2 > 0 and q.r_tongue > 0 else None),
    )
    checks += cr.consistency_checks(
        mdot_in=f.mdot_in,
        mdot_out=f.mdot_out,
        eta_p_imp=_ok(d.eta_p_imp),
        T0_drift_diffuser=d.dT0_diff,
        T0_drift_volute=d.dT0_vol,
    )
    return d, checks


def _ok(v):
    """把 nan / 0 统一成 None，避免判据误报"""
    if v is None:
        return None
    try:
        if math.isnan(v):
            return None
    except TypeError:
        pass
    return v


def summary(m: Machine, d: Derived) -> str:
    q = m.geom
    L = []
    L.append("=" * 72)
    L.append(f"逆向体检：{m.name or '(未命名)'}")
    L.append("=" * 72)
    L.append(f"  n = {m.n_rpm:.0f} r/min   mdot = {m.mdot:.4f} kg/s   "
             f"U2 = {d.U2:.2f} m/s")
    L.append(f"  几何：r1t={q.r1t*1e3:.2f}  r1h={q.r1h*1e3:.2f}  "
             f"r2={q.r2*1e3:.2f}  b2={q.b2*1e3:.2f} mm")
    L.append(f"        r3={q.r3*1e3:.2f}  r4={q.r4*1e3:.2f}  "
             f"z_imp={q.z_total}  z_diff={q.z_diff}")
    L.append("")
    L.append("-- 反算无量纲量 " + "-" * 55)
    rows = [
        ("psi = dh0/U2^2", d.psi, "-"),
        ("phi2 = c_r2/U2", d.phi2, "-"),
        ("b2/r2", d.b2_over_r2, "-"),
        ("k3 = r3/r2", d.k3, "-"),
        ("ky = r4/r3", d.ky, "-"),
        ("hub_tip = r1h/r1t", d.hub_tip, "-"),
        ("beta1_flow", d.beta1_flow_deg, "deg"),
        ("beta2_flow", d.beta2_flow_deg, "deg"),
        ("M_w1", d.M_w1, "-"),
        ("M2_abs", d.M2_abs, "-"),
        ("M2_rel", d.M2_rel, "-"),
        ("M3 (自由涡估)", d.M3, "-"),
        ("alpha3", d.alpha3_deg, "deg"),
        ("slip_cfd", d.slip_cfd, "-"),
        ("slip_wiesner", d.slip_wiesner, "-"),
        ("solidity_imp", d.solidity_imp, "-"),
        ("solidity_diff", d.solidity_diff, "-"),
        ("theta_diff", d.theta_diff, "deg"),
        ("eta_p_imp (反算)", d.eta_p_imp, "-"),
        ("dT0 扩压器", d.dT0_diff, "K"),
        ("dT0 蜗壳", d.dT0_vol, "K"),
    ]
    for k, v, u in rows:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            L.append(f"    {k:<22} = {'-':>10} {u}")
        else:
            L.append(f"    {k:<22} = {v:>10.4f} {u}")
    return "\n".join(L)
