"""
单级离心压缩机一维设计（正问题）。

输入：流量、转速、总压比、进口总温总压 + 一组设计选择
输出：叶轮 / 扩压器 / 蜗壳的几何 + 判据表 + 子午线

与崔晓杰(2025) 的差别（也是本工具的立场）：
  1) 他的流程是"选压头系数 -> 算 U2 -> 一路算下去"，只对 r1 做了黄金分割优化。
     这里把 (psi, phi2, z, beta2A, Kd) 都作为显式输入，设计者自己承担选择，
     程序只负责把它们变成自洽的几何，并把每条经验判据的通过情况摆出来。
  2) 叶轮进口半径用解析解而不是黄金分割：
     w1^2 = (A/r^2)^2 + (omega*r)^2 对 r 求导令其为 0，
     得 r_opt = (2A^2/omega^2)^(1/6)，A = mdot/(rho*pi*(1-Kd^2))。
     比迭代快，而且能一眼看出 w1_min 与哪些量有关。
  3) 出口安装角用 Wiesner 滑移反解，保证"欧拉功 = 一维功"这条自洽性从一开始就成立，
     而不是先拍一个 beta2A 再指望 CFD 对得上。

角度定义统一为 **从径向测量**（与 CFD-Post 的 atan2(Vr, Vt) 一致）：
  alpha = atan2(c_r, c_u)，beta = atan2(w_r, w_u)。
  这样 CFD 后处理出来的数可以直接跟设计值对表。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy.optimize import brentq

from .gas import IdealGas
from . import bezier


# ============================================================
# 输入
# ============================================================
@dataclass
class Case:
    """设计工况 + 设计选择。带 ★ 的是真正影响结果的选择，其余是次要参数。"""

    # --- 工况 ---
    mdot: float                  # kg/s
    n_rpm: float                 # r/min
    PR_tt: float                 # 总压比
    T0_in: float                 # K
    p0_in: float                 # Pa
    gas: IdealGas = field(default_factory=IdealGas)

    # --- 设计选择 ★ ---
    eta_tt: float = 0.80         # 目标总-总效率（用于定 Δh）
    psi: float = 0.60            # 负荷系数 Δh/U2^2
    phi2: float = 0.26           # 出口流量系数 c_r2/U2
    z_main: int = 9              # 主叶片数
    z_split: int = 9             # 分流叶片数（0 = 无）
    Kd: float = 0.40             # 轮毂比 r1h/r1t
    beta2A_deg: float | None = None   # 若 None，由 Wiesner 滑移反解

    # --- 后置部件 ---
    sigma_rear: float = 0.86     # 扩压器+蜗壳总压恢复系数
    r3_over_r2: float = 1.10     # 无叶段长度比 ★
    r4_over_r3: float = 1.32     # 扩压器半径比 ★
    z_diff: int | None = None    # None -> 由扩张角反算，再取最近的互质整数
    theta_target_deg: float = 9.0     # 当量扩张角目标 ★
    beta4A_offset_deg: float = 5.0    # 出口偏离角：beta4A = alpha4_ideal + 本值
    M5_target: float = 0.10      # 蜗壳出口马赫目标

    # --- 数值/经验 ---
    eta_p_imp: float = 0.90      # 叶轮多变效率（定 p2 用）
    eta_p_vaneless: float = 0.95
    eta_p_diff: float = 0.92
    sigma_inlet: float = 0.99    # 进口总压恢复
    t_leading: float = 0.0015    # 叶片前缘厚度 m
    t_trailing: float = 0.0010   # 叶片尾缘厚度 m
    i_inlet_deg: float = 1.0     # 进口攻角
    z_out_z_in: float = 0.85     # 轴向长度 / 径向跨度

    # ---- 派生 ----
    @property
    def omega(self) -> float:
        return self.n_rpm * 2.0 * math.pi / 60.0

    @property
    def z_total(self) -> int:
        return self.z_main + self.z_split


# ============================================================
# 输出
# ============================================================
@dataclass
class Impeller:
    U2: float = 0.0
    r2: float = 0.0
    b2: float = 0.0
    c_u2: float = 0.0
    c_r2: float = 0.0
    c2: float = 0.0
    T2: float = 0.0
    p2: float = 0.0
    rho2: float = 0.0
    T02: float = 0.0
    p02: float = 0.0
    beta2_flow_deg: float = 0.0
    beta2A_deg: float = 0.0
    M2_abs: float = 0.0
    M2_rel: float = 0.0
    slip_wiesner: float = 0.0
    # 进口
    r1t: float = 0.0
    r1h: float = 0.0
    c_m1: float = 0.0
    w1: float = 0.0
    U1t: float = 0.0
    M_w1: float = 0.0
    beta1_flow_deg: float = 0.0
    beta1A_deg: float = 0.0
    w2_over_w1: float = 0.0
    blockage_inlet: float = 0.0
    # 整体
    dh: float = 0.0
    dh_s: float = 0.0
    power: float = 0.0
    solidity: float = 0.0
    # 子午线
    meridian: bezier.Meridian | None = None


@dataclass
class Diffuser:
    r3: float = 0.0
    r4: float = 0.0
    b3: float = 0.0
    b4: float = 0.0
    c_u3: float = 0.0
    c_r3: float = 0.0
    c3: float = 0.0
    alpha3_deg: float = 0.0
    M3: float = 0.0
    T3: float = 0.0
    p3: float = 0.0
    beta3A_deg: float = 0.0
    beta4A_deg: float = 0.0
    alpha4_deg: float = 0.0
    M4: float = 0.0
    c4: float = 0.0
    T4: float = 0.0
    p4: float = 0.0
    z_diff: int = 0
    theta_deg: float = 0.0
    solidity: float = 0.0
    blade_length: float = 0.0
    o3: float = 0.0
    o4: float = 0.0


@dataclass
class Volute:
    A_in: float = 0.0
    A_out: float = 0.0
    r5: float = 0.0
    c5: float = 0.0
    M5: float = 0.0
    T5: float = 0.0
    p5: float = 0.0
    area_ratio: float = 0.0


@dataclass
class Result:
    case: Case
    imp: Impeller
    dif: Diffuser
    vol: Volute
    checks: list = field(default_factory=list)


# ============================================================
# 叶轮
# ============================================================
def wiesner_slip(beta2A_rad: float, z: int, r1_over_r2: float = 0.0) -> float:
    """
    Wiesner (1967) 滑移系数。
    sigma = 1 - sqrt(cos beta2A)/z^0.7，并对 r1/r2 较大时做修正。
    """
    s = 1.0 - math.sqrt(max(math.cos(beta2A_rad), 1e-6)) / (z ** 0.7)
    if r1_over_r2 > 0.0:
        lim = math.exp(-8.16 / z)
        if r1_over_r2 > lim:
            s *= 1.0 - ((r1_over_r2 - lim) / (1.0 - lim)) ** 3
    return s


def solve_beta2A(c_u2: float, c_r2: float, U2: float, z: int,
                 r1_over_r2: float = 0.0) -> tuple[float, float]:
    """
    解 beta2A 使 Wiesner 滑移给出的 c_u2 与目标一致：
        c_u2 = sigma(beta2A) * (U2 - c_r2 / tan(beta2A))
    返回 (beta2A_deg, sigma)。
    """
    def f(beta_deg):
        b = math.radians(beta_deg)
        if math.tan(b) < 1e-6:
            return -c_u2
        return wiesner_slip(b, z, r1_over_r2) * (U2 - c_r2 / math.tan(b)) - c_u2

    lo, hi = 8.0, 75.0
    if f(lo) * f(hi) > 0:
        raise ValueError(
            f"beta2A 无解：psi={c_u2/U2:.3f} 超出该叶片数能提供的范围，"
            f"调 psi / phi2 / z 其中一个"
        )
    b = brentq(f, lo, hi, xtol=1e-10)
    return b, wiesner_slip(math.radians(b), z, r1_over_r2)


def design_impeller(c: Case) -> Impeller:
    g = c.gas
    imp = Impeller()

    # ---- 能量头 ----
    imp.dh_s = g.dh_s(c.T0_in, c.PR_tt)
    imp.dh = imp.dh_s / c.eta_tt
    imp.power = c.mdot * imp.dh

    # ---- 出口速度三角形 ----
    U2 = math.sqrt(imp.dh / c.psi)
    r2 = U2 / c.omega
    c_u2 = imp.dh / U2
    c_r2 = c.phi2 * U2
    c2 = math.hypot(c_u2, c_r2)

    T02 = g.T0_from_dh(c.T0_in, imp.dh)
    T2 = T02 - c2 ** 2 / (2.0 * g.cp)
    p02 = c.p0_in * c.PR_tt / c.sigma_rear
    p2 = g.p_static_from_T(p02, T02, T2, c.eta_p_imp)
    rho2 = g.rho(p2, T2)
    b2 = c.mdot / (2.0 * math.pi * r2 * rho2 * c_r2)

    a2 = g.a(T2)
    M2_abs = c2 / a2
    w_u2 = U2 - c_u2
    w2 = math.hypot(c_r2, w_u2)
    M2_rel = w2 / a2
    beta2_flow = math.degrees(math.atan2(c_r2, w_u2))

    imp.U2, imp.r2, imp.b2 = U2, r2, b2
    imp.c_u2, imp.c_r2, imp.c2 = c_u2, c_r2, c2
    imp.T2, imp.p2, imp.rho2 = T2, p2, rho2
    imp.T02, imp.p02 = T02, p02
    imp.M2_abs, imp.M2_rel = M2_abs, M2_rel
    imp.beta2_flow_deg = beta2_flow

    # ---- 进口（w1 最小化） ----
    # 注意：解析解 r_opt=(2A^2/w^2)^(1/6) 只在 rho1 固定时成立，
    # 但 rho1 本身随 c_m1 变（c_m1 大 -> T1 低 -> rho1 低 -> c_m1 更大），
    # 这个正反馈会把最优点往小半径推，所以必须把 rho1 的耦合算进去。
    # 这里对外层 r1t 做一维搜索，内层解自洽的 c_m1。
    from scipy.optimize import minimize_scalar

    def _state(r1t_try: float) -> tuple[float, float]:
        """给定 r1t，解出自洽的 (c_m1, rho1)"""
        A1 = math.pi * (r1t_try ** 2 - (c.Kd * r1t_try) ** 2)
        c_m = 100.0
        rho = g.rho(c.p0_in, c.T0_in)
        for _ in range(80):
            T = c.T0_in - c_m ** 2 / (2.0 * g.cp)
            p = c.p0_in * c.sigma_inlet * (T / c.T0_in) ** g.g_exp
            rho_new = g.rho(p, T)
            c_m_new = c.mdot / (rho_new * A1)
            d_ = abs(c_m_new - c_m)
            c_m = 0.5 * (c_m + c_m_new)
            rho = rho_new
            if d_ < 1e-9:
                break
        return c_m, rho

    def _w1(r1t_try: float) -> float:
        c_m, _ = _state(r1t_try)
        return math.hypot(c_m, c.omega * r1t_try)

    # 粗估初值再局部细化
    rho_guess = g.rho(c.p0_in * c.sigma_inlet, c.T0_in)
    A_guess = c.mdot / (rho_guess * math.pi * (1.0 - c.Kd ** 2))
    r0 = (2.0 * A_guess ** 2 / c.omega ** 2) ** (1.0 / 6.0)
    res = minimize_scalar(_w1, bounds=(0.75 * r0, 1.60 * r0), method="bounded",
                          options={"xatol": 1e-7})
    r1t = float(res.x)
    c_m1, rho1 = _state(r1t)

    r1h = c.Kd * r1t
    U1t = c.omega * r1t
    w1 = math.hypot(c_m1, U1t)
    T1 = c.T0_in - c_m1 ** 2 / (2.0 * g.cp)
    M_w1 = w1 / g.a(T1)
    beta1_flow = math.degrees(math.atan2(c_m1, U1t))

    imp.r1t, imp.r1h = r1t, r1h
    imp.c_m1, imp.w1, imp.U1t = c_m1, w1, U1t
    imp.M_w1 = M_w1
    imp.beta1_flow_deg = beta1_flow
    imp.beta1A_deg = beta1_flow + c.i_inlet_deg
    imp.w2_over_w1 = w2 / w1

    # 进口堵塞：叶片厚度在进口占的面积
    A1 = math.pi * (r1t ** 2 - r1h ** 2)
    b1 = r1t - r1h
    A_block = c.z_main * c.t_leading * b1 / math.sin(math.radians(imp.beta1A_deg))
    imp.blockage_inlet = 1.0 - min(A_block / A1, 0.5)

    # 稠度（用主叶片数，长短叶片时这样才有意义）
    r_m = 0.5 * (r1t + r2)
    beta_m = math.radians(0.5 * (imp.beta1A_deg + beta2_flow))
    L_blade = (r2 - r1t) / max(math.sin(beta_m), 0.2)
    imp.solidity = c.z_main * L_blade / (2.0 * math.pi * r_m)

    # ---- beta2A ----
    if c.beta2A_deg is not None:
        b2a = c.beta2A_deg
        sig = wiesner_slip(math.radians(b2a), c.z_total, r1t / r2)
        # 用给定角度反算实际 c_u2，检查与欧拉功差多少
        c_u2_pred = sig * (U2 - c_r2 / math.tan(math.radians(b2a)))
        imp.slip_wiesner = sig
        imp.beta2A_deg = b2a
        if abs(c_u2_pred - c_u2) / c_u2 > 0.02:
            imp.beta2_flow_deg = beta2_flow  # 保留
    else:
        b2a, sig = solve_beta2A(c_u2, c_r2, U2, c.z_total, r1t / r2)
        imp.beta2A_deg = b2a
        imp.slip_wiesner = sig

    # ---- 子午流道 ----
    # 用"面积驱动"：先定 shroud 的椭圆转弯，再按 A(s) 线性从进口到出口
    # 反求 hub 半径。这样面积精确单调，不会出现"先画线再检查面积"的反复。
    # 轴向长度按 hub 的径向跨度给（不是 shroud 的），否则转弯过急、曲率尖峰。
    span_hub = (r2 - b2) - r1h
    z_out = c.z_out_z_in * span_hub
    try:
        imp.meridian = bezier.build_by_area(
            r_hub_in=r1h, r_shroud_in=r1t,
            r_hub_out=r2 - b2, r_shroud_out=r2,
            z_in=0.0, z_out=z_out,
            lead_shroud=0.55, blend=1.8,
        )
    except Exception:
        imp.meridian = None

    return imp


# ============================================================
# 扩压器
# ============================================================
def _solve_station(r, b, mdot, T0, p0, gas, eta_p, c_u_guess, follow_angle_deg=None):
    """
    解一个截面的 (c_u, c_r, c, T, p, rho)。
    follow_angle_deg 给出时，c_u = c_r/tan(beta)（气流跟随叶片）；
    否则 c_u 由自由涡 c_u*r = const 传入。
    """
    c_r = c_u_guess * 0.3
    c_u = c_u_guess
    for _ in range(60):
        c = math.hypot(c_u, c_r)
        T = T0 - c ** 2 / (2.0 * gas.cp)
        p = gas.p_static_from_T(p0, T0, T, eta_p)
        rho = gas.rho(p, T)
        c_r_new = mdot / (2.0 * math.pi * r * b * rho)
        if follow_angle_deg is not None:
            c_u_new = c_r_new / math.tan(math.radians(follow_angle_deg))
        else:
            c_u_new = c_u
        d = abs(c_r_new - c_r) + abs(c_u_new - c_u)
        c_r, c_u = 0.5 * (c_r + c_r_new), 0.5 * (c_u + c_u_new)
        if d < 1e-8:
            break
    c = math.hypot(c_u, c_r)
    T = T0 - c ** 2 / (2.0 * gas.cp)
    p = gas.p_static_from_T(p0, T0, T, eta_p)
    return c_u, c_r, c, T, p, gas.rho(p, T)


def design_diffuser(c: Case, imp: Impeller) -> Diffuser:
    g = c.gas
    d = Diffuser()
    d.r3 = c.r3_over_r2 * imp.r2
    d.r4 = c.r4_over_r3 * d.r3
    d.b3 = d.b4 = imp.b2

    # ---- 无叶段：自由涡 c_u*r = const ----
    d.c_u3 = imp.c_u2 * imp.r2 / d.r3
    cu, cr, cc, T3, p3, rho3 = _solve_station(
        d.r3, d.b3, c.mdot, imp.T02, imp.p02, g, c.eta_p_vaneless, d.c_u3)
    d.c_u3, d.c_r3, d.c3 = cu, cr, cc
    d.T3, d.p3 = T3, p3
    d.alpha3_deg = math.degrees(math.atan2(cr, cu))
    d.M3 = cc / g.a(T3)

    # ---- 叶片角 ----
    # 进口零攻角（留 i_inlet）。
    # 出口角不能直接取等角：等宽 + 等角确实能自洽，但那样算出来的叶片数
    # 会到 35~40 片（稠度 >1.7），跟实际扩压器 15~25 片对不上。
    # 实际设计是让叶片更"径向"一点（beta4A > beta3A），流道宽度增长放慢，
    # 叶片数就下来了。beta4A 取"自由涡出口气流角 + 偏离角"，
    # 偏离角默认 5 度 —— 这是扩压器出口气流不完全跟随叶片的典型量级。
    d.beta3A_deg = d.alpha3_deg + c.i_inlet_deg

    p03 = imp.p02 * 0.985   # 无叶段总压损失
    # 先按自由涡算出口理想气流角
    cu4i, cr4i, c4i, T4i, p4i, rho4i = _solve_station(
        d.r4, d.b4, c.mdot, imp.T02, p03, g, c.eta_p_diff,
        d.c_u3 * d.r3 / d.r4)
    d.alpha4_deg = math.degrees(math.atan2(cr4i, cu4i))
    d.beta4A_deg = d.alpha4_deg + c.beta4A_offset_deg

    # ---- 出口实际状态（气流跟随叶片角） ----
    cu4, cr4, c4, T4, p4, rho4 = _solve_station(
        d.r4, d.b4, c.mdot, imp.T02, p03, g, c.eta_p_diff, d.c_r3,
        follow_angle_deg=d.beta4A_deg)
    d.c4, d.T4, d.p4 = c4, T4, p4
    d.M4 = c4 / g.a(T4)

    # ---- 叶片数：由当量扩张角反算 ----
    # theta = 2*atan((o4-o3)/(2L))，o = 2*pi*r*cos(beta)/z - t
    # 解 z 使 theta = theta_target，再取最近的、与叶轮互质且稠度合理的整数
    if c.z_diff is not None:
        d.z_diff = c.z_diff
    else:
        b3 = math.radians(d.beta3A_deg)
        b4 = math.radians(d.beta4A_deg)
        am = 0.5 * (b3 + b4)
        L = (d.r4 - d.r3) / max(math.cos(am), 0.2)
        num = 2.0 * math.pi * (d.r4 * math.cos(b4) - d.r3 * math.cos(b3))
        den = 2.0 * L * math.tan(math.radians(c.theta_target_deg) / 2.0) + (
            c.t_trailing - c.t_leading)
        z_ideal = num / den if den > 1e-9 else 25.0
        d.z_diff = _pick_int(z_ideal, imp_z=c.z_total, r3=d.r3, r4=d.r4,
                             a3=d.alpha3_deg, a4=d.alpha4_deg)

    d.blade_length = (d.r4 - d.r3) / math.cos(
        math.radians(0.5 * (d.alpha3_deg + d.alpha4_deg)))
    d.solidity = _diffuser_solidity(d.z_diff, d.r3, d.r4, d.alpha3_deg, d.alpha4_deg)

    # 喉部宽度（从径向测角，周向间距投影到叶片法向）
    d.o3 = 2.0 * math.pi * d.r3 * math.cos(math.radians(d.beta3A_deg)) / d.z_diff - c.t_leading
    d.o4 = 2.0 * math.pi * d.r4 * math.cos(math.radians(d.beta4A_deg)) / d.z_diff - c.t_trailing
    if d.blade_length > 0:
        d.theta_deg = math.degrees(2.0 * math.atan((d.o4 - d.o3) / (2.0 * d.blade_length)))

    return d


def _pick_int(z_ideal: float, imp_z: int, r3: float, r4: float,
              a3: float, a4: float, span: int = 4) -> int:
    """
    把理想的叶片数圆整成整数：优先与叶轮叶片数互质（最大公约数 <= 2），
    其次稠度落在 1.0~1.7。两个都不满足时退回最接近的整数。
    """
    lo = max(7, int(round(z_ideal)) - span)
    hi = int(round(z_ideal)) + span
    cands = []
    for z in range(lo, hi + 1):
        if math.gcd(z, imp_z) > 2:
            continue
        s = _diffuser_solidity(z, r3, r4, a3, a4)
        score = abs(z - z_ideal) + 4.0 * max(0.0, 1.0 - s) + 4.0 * max(0.0, s - 1.7)
        cands.append((score, z))
    if cands:
        return min(cands)[1]
    return int(round(z_ideal))


def _diffuser_solidity(z: int, r3: float, r4: float, a3: float, a4: float) -> float:
    am = math.radians(0.5 * (a3 + a4))
    L = (r4 - r3) / max(math.cos(am), 0.2)
    rm = 0.5 * (r3 + r4)
    return z * L / (2.0 * math.pi * rm)


# ============================================================
# 蜗壳
# ============================================================
def design_volute(c: Case, imp: Impeller, d: Diffuser) -> Volute:
    g = c.gas
    v = Volute()
    v.A_in = 2.0 * math.pi * d.r4 * d.b4
    v.T5 = imp.T02                       # 无功耗，总温不变
    p05 = d.p4 * (1.0 + 0.0)             # 占位：下面用总压近似
    # 用出口静压 + 出口速度反推
    T5 = v.T5
    a5 = g.a(T5)
    c5 = c.M5_target * a5
    # 迭代：rho5 依赖 p5 依赖 c5（弱依赖），两轮足够
    p5 = d.p4
    for _ in range(20):
        rho5 = g.rho(p5, T5)
        A_out = c.mdot / (rho5 * c5)
        v.A_out = A_out
        v.r5 = math.sqrt(A_out / math.pi)
        # 出口动压回收：p5 = p05 - 0.5*rho*c5^2*Cp_loss
        p5_new = d.p4 * 0.995 - 0.5 * rho5 * c5 ** 2 * 0.15
        if abs(p5_new - p5) < 1.0:
            p5 = p5_new
            break
        p5 = p5_new
    v.c5, v.M5, v.p5 = c5, c5 / a5, p5
    v.area_ratio = v.A_out / v.A_in
    return v


# ============================================================
# 主入口
# ============================================================
def design(c: Case) -> Result:
    imp = design_impeller(c)
    dif = design_diffuser(c, imp)
    vol = design_volute(c, imp, dif)
    return Result(case=c, imp=imp, dif=dif, vol=vol)


def summary(r: Result) -> str:
    """人读的摘要"""
    c, i, d, v = r.case, r.imp, r.dif, r.vol
    L = []
    L.append("=" * 72)
    L.append("一维设计结果")
    L.append("=" * 72)
    L.append(f"工况: mdot={c.mdot:.4f} kg/s, n={c.n_rpm:.0f} r/min, "
             f"PR_tt={c.PR_tt:.3f}, T0in={c.T0_in:.2f} K, p0in={c.p0_in/1000:.2f} kPa")
    L.append(f"选择: psi={c.psi:.3f}, phi2={c.phi2:.3f}, "
             f"z={c.z_main}+{c.z_split}, Kd={c.Kd:.2f}, eta_tt={c.eta_tt:.3f}")
    L.append("")
    L.append("-- 叶轮 " + "-" * 62)
    L.append(f"  U2        = {i.U2:9.2f} m/s      r2      = {i.r2*1e3:8.2f} mm")
    L.append(f"  b2        = {i.b2*1e3:9.2f} mm       b2/r2   = {i.b2/i.r2:8.4f}")
    L.append(f"  r1t       = {i.r1t*1e3:9.2f} mm       r1h     = {i.r1h*1e3:8.2f} mm")
    L.append(f"  beta1A    = {i.beta1A_deg:9.2f} deg      beta2A  = {i.beta2A_deg:8.2f} deg")
    L.append(f"  beta1_flow= {i.beta1_flow_deg:9.2f} deg      beta2_flow={i.beta2_flow_deg:7.2f} deg")
    L.append(f"  c_u2      = {i.c_u2:9.2f} m/s      c_r2    = {i.c_r2:8.2f} m/s")
    L.append(f"  M_w1      = {i.M_w1:9.4f}          M2_abs  = {i.M2_abs:8.4f}")
    L.append(f"  w2/w1     = {i.w2_over_w1:9.4f}          slip    = {i.slip_wiesner:8.4f}")
    L.append(f"  dh        = {i.dh/1000:9.2f} kJ/kg     P       = {i.power/1000:8.2f} kW")
    L.append(f"  solidity  = {i.solidity:9.3f} (主叶片)")
    L.append("")
    L.append("-- 扩压器 " + "-" * 61)
    L.append(f"  r3/r2     = {d.r3/i.r2:9.4f}          r4/r3   = {d.r4/d.r3:8.4f}")
    L.append(f"  r3        = {d.r3*1e3:9.2f} mm       r4      = {d.r4*1e3:8.2f} mm")
    L.append(f"  b3=b4     = {d.b3*1e3:9.2f} mm")
    L.append(f"  alpha3    = {d.alpha3_deg:9.2f} deg      alpha4  = {d.alpha4_deg:8.2f} deg")
    L.append(f"  beta3A    = {d.beta3A_deg:9.2f} deg      beta4A  = {d.beta4A_deg:8.2f} deg")
    L.append(f"  M3        = {d.M3:9.4f}          M4      = {d.M4:8.4f}")
    L.append(f"  z_diff    = {d.z_diff:9d}          theta   = {d.theta_deg:8.2f} deg")
    L.append(f"  solidity  = {d.solidity:9.3f}          L_blade = {d.blade_length*1e3:8.2f} mm")
    L.append(f"  o3        = {d.o3*1e3:9.2f} mm       o4      = {d.o4*1e3:8.2f} mm")
    L.append("")
    L.append("-- 蜗壳 " + "-" * 63)
    L.append(f"  A_in      = {v.A_in*1e4:9.3f} cm2     A_out   = {v.A_out*1e4:8.3f} cm2")
    L.append(f"  r5        = {v.r5*1e3:9.2f} mm       A_out/A_in = {v.area_ratio:7.3f}")
    L.append(f"  M5        = {v.M5:9.4f}          c5      = {v.c5:8.2f} m/s")
    return "\n".join(L)
