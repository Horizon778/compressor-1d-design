"""
子午流道构造：椭圆弧（默认）+ 五点四次贝塞尔（可选）。

为什么默认不用贝塞尔
--------------------
崔晓杰(2025) 2.2.2 节用五点四次贝塞尔，端点位置由 r1/r2/轴向长度定，
端点斜率由 dz/du 定，中间控制点由端点曲率反解。方法本身没问题，
但用在"进口大、出口窄、90 度转弯"的离心叶轮上有两个结构性缺陷：

1) 曲率约束 k0=k4=0 时，解出来的 P2 必然落在 **进口半径** 上
   （推导：两端切线正交 -> 行列式只含 dz/du，P2r 退化为 P0r）。
   于是前三个控制点 r 相同，曲线前半段 r 几乎不动、后半段猛涨，
   结果是"通道宽度在中段不收缩"，流通面积先鼓后收。
   本算例（r1t=49.6, r2=99.4, b2=7.6, z=30）面积从 65 涨到 78 再掉到 52 cm²，
   而正确设计应该 65 -> 47 单调递减。

2) 想靠调 lam（切向量长度）修好也没用：P2r 只依赖 P3r/P4r 和端点切线，
   与 lam 无关，所以调 lam 只改 z 分布不改 r 分布。

结论：贝塞尔适合"形状自由、靠人工调"的场合（论文里就是这么用的，
他后面还要靠内切圆图反复检查）。要做"面积可控"的设计，用椭圆弧更直接：
进口直段 + 椭圆弧转弯（起点切向轴向、终点切向径向）+ 出口，
hub 和 shroud 用不同的直段长度，就能精确控制面积单调性。

椭圆弧构造
----------
进口直段长 lead*L_z（lead 为轴向占比），
转弯段为椭圆：z = z1 + a*sin(t), r = r_in + b*(1-cos(t)), t: 0 -> pi/2
其中 a = (1-lead)*L_z, b = r_out - r_in。
t=0 处切向 (a, 0) 纯轴向，t=pi/2 处切向 (0, b) 纯径向，天然满足进出口方向要求，
且曲率沿程连续（椭圆段内）。

hub 取 lead 小（早转），shroud 取 lead 大（晚转），
两者错开后通道宽度在前段就开始收缩，面积才能单调。

坐标：(z, r)，单位 m。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize_scalar


# ============================================================
# 贝塞尔曲线求值（保留给 build_stage 用）
# ============================================================
def bernstein5(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return np.stack(
        [(1 - u) ** 4, 4 * (1 - u) ** 3 * u, 6 * (1 - u) ** 2 * u ** 2,
         4 * (1 - u) * u ** 3, u ** 4],
        axis=-1,
    )


def curve(P, u) -> np.ndarray:
    """五点四次贝塞尔求值。P:(5,2), u:(n,) -> (n,2)"""
    return bernstein5(u) @ np.asarray(P, dtype=float)


def derivative(P, u) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    u = np.asarray(u, dtype=float)[..., None]
    return (
        4 * (1 - u) ** 3 * (P[1] - P[0])
        + 12 * (1 - u) ** 2 * u * (P[2] - P[1])
        + 12 * (1 - u) * u ** 2 * (P[3] - P[2])
        + 4 * u ** 3 * (P[4] - P[3])
    )


def second_derivative(P, u) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    u = np.asarray(u, dtype=float)[..., None]
    return 12 * (
        (1 - u) ** 2 * (P[2] - 2 * P[1] + P[0])
        + 2 * (1 - u) * u * (P[3] - 2 * P[2] + P[1])
        + u ** 2 * (P[4] - 2 * P[3] + P[2])
    )


def curvature(P, u) -> np.ndarray:
    """带符号曲率 kappa = (z'r'' - r'z'') / (z'^2+r'^2)^1.5"""
    d1 = derivative(P, u)
    d2 = second_derivative(P, u)
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    return num / np.maximum(den, 1e-30)


def curvature_from_points(p: np.ndarray) -> np.ndarray:
    """由点集数值求曲率（椭圆弧/任意离散曲线用）"""
    z = p[:, 0]
    r = p[:, 1]
    dz = np.gradient(z)
    dr = np.gradient(r)
    ddz = np.gradient(dz)
    ddr = np.gradient(dr)
    num = dz * ddr - dr * ddz
    den = (dz ** 2 + dr ** 2) ** 1.5
    return num / np.maximum(den, 1e-30)


# ============================================================
# 贝塞尔控制点求解
# ============================================================
def solve_control_points(P0, P4, t0, t4, k0=0.0, k4=0.0, lam0=None, lam4=None) -> np.ndarray:
    """
    由两端点、端点切向、端点带符号曲率反解 5 个控制点。

    解的是这个线性方程组（P2 为唯一未知）：
        A0*P2r - B0*P2z = k0*L0^3/12 - [A0*(P0r-2P1r) - B0*(P0z-2P1z)]
        A1*P2r - B1*P2z = k4*L1^3/12 - [A1*(P4r-2P3r) - B1*(P4z-2P3z)]
    其中 Ai=z'(i), Bi=r'(i), Li=|dP/du|_i。

    注意 Cramer 的列顺序：系数矩阵是 [[-B0, A0], [-B1, A1]]，
    P2z 对应第一列、P2r 对应第二列，写反了会得到 r 为负的解。

    退化：两端切向平行 -> det=0 -> P2 取 P1P3 中点。
    """
    P0 = np.asarray(P0, dtype=float)
    P4 = np.asarray(P4, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    t4 = np.asarray(t4, dtype=float)
    t0 = t0 / np.linalg.norm(t0)
    t4 = t4 / np.linalg.norm(t4)

    L = float(np.linalg.norm(P4 - P0))
    lam0 = 0.25 * L if lam0 is None else float(lam0)
    lam4 = 0.25 * L if lam4 is None else float(lam4)

    P1 = P0 + lam0 * t0
    P3 = P4 - lam4 * t4

    A0, B0 = 4.0 * lam0 * t0
    A1, B1 = 4.0 * lam4 * t4
    L0 = math.hypot(A0, B0)
    L1 = math.hypot(A1, B1)

    R0 = (k0 * L0 ** 3 / 12.0) - (A0 * (P0[1] - 2 * P1[1]) - B0 * (P0[0] - 2 * P1[0]))
    R1 = (k4 * L1 ** 3 / 12.0) - (A1 * (P4[1] - 2 * P3[1]) - B1 * (P4[0] - 2 * P3[0]))

    det = A0 * B1 - A1 * B0
    scale = max(abs(A0 * B1), abs(A1 * B0), 1e-30)
    if abs(det) < 1e-9 * scale:
        P2 = 0.5 * (P1 + P3)
    else:
        P2z = (R0 * A1 - A0 * R1) / det
        P2r = (B1 * R0 - B0 * R1) / det
        P2 = np.array([P2z, P2r])

    return np.vstack([P0, P1, P2, P3, P4])


def tangent_from_angle(deg: float) -> np.ndarray:
    """切向单位向量：0=轴向(+z)，90=径向(+r)"""
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


# ============================================================
# 内切圆工具（可选，诊断用）
# ============================================================
def _dist_to_points(P, pts) -> float:
    d2 = ((pts - np.asarray(P, dtype=float)) ** 2).sum(axis=1)
    return float(np.sqrt(d2.min()))


def inscribed_circle(pts_hub, pts_shroud, i_hub) -> tuple[np.ndarray, float]:
    """
    在 pts_hub[i_hub] 处求内切圆（离散版）。
    仅作诊断：通道宽度用准正交线算更稳，这个函数在急转弯处会失效。
    """
    M = pts_hub[i_hub]
    n = min(i_hub + 1, len(pts_hub) - 1)
    m = max(i_hub - 1, 0)
    dM = pts_hub[n] - pts_hub[m]
    dM = dM / max(np.linalg.norm(dM), 1e-30)

    proj = (pts_shroud - M) @ dM
    idx = np.where(np.diff(np.sign(proj)) != 0)[0]
    if len(idx) == 0:
        return M, 0.0
    K = pts_shroud[idx[0]]
    return 0.5 * (M + K), 0.5 * float(np.linalg.norm(K - M))


# ============================================================
# 结果容器
# ============================================================
@dataclass
class Meridian:
    hub: np.ndarray                    # (n,2) 点集
    shroud: np.ndarray                 # (n,2) 点集
    hub_ctrl: np.ndarray | None = None      # 贝塞尔控制点（若有）
    shroud_ctrl: np.ndarray | None = None
    s: np.ndarray | None = None             # 归一化进度
    centers: np.ndarray | None = None       # 准正交线中点 (n,2)
    widths: np.ndarray | None = None        # 通道宽度 b
    area: np.ndarray | None = None          # 流通面积 m2
    kind: str = "elliptic"                  # elliptic / bezier

    # ---- 便捷 ----
    def hub_pts(self):
        return self.hub

    def shroud_pts(self):
        return self.shroud

    def kappa_hub(self):
        return curvature_from_points(self.hub)

    def kappa_shroud(self):
        return curvature_from_points(self.shroud)

    def meridional_length(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.hub, axis=0), axis=1)))

    @property
    def radii(self):
        """半通道宽度，兼容旧接口"""
        return None if self.widths is None else 0.5 * self.widths

    @property
    def u(self):
        return self.s

    # ---- 判据 ----
    def check(self, tol=0.10) -> dict:
        """
        三条判据：
          1) 流通面积不得出现"比进出口都大"的鼓包（超出 tol 即不合格）
          2) 通道宽度沿程单调
          3) hub/shroud 曲率不反号
        tol 是允许的相对过冲，默认 10%。进口段轻微扩张（<=10%）是允许的，
        用于控制进口相对马赫；超过就是流道设计问题。
        """
        items = []
        ok_all = True

        if self.area is not None and len(self.area) > 2:
            a = np.asarray(self.area)
            ref = max(a[0], a[-1])
            overshoot = a.max() / ref - 1.0
            ok = overshoot <= tol
            ok_all &= ok
            direction = "递减" if a[-1] < a[0] else "递增"
            items.append((
                "流通面积无鼓包",
                ok,
                f"A_in={a[0]*1e4:.2f} cm2, A_out={a[-1]*1e4:.2f} cm2, "
                f"A_max={a.max()*1e4:.2f} cm2 ({direction}), 过冲={overshoot*100:.1f}%",
            ))

        if self.widths is not None and len(self.widths) > 2:
            b = np.asarray(self.widths)
            d = np.diff(b)
            n_pos = int(np.sum(d > 0.02 * abs(b[0])))
            n_neg = int(np.sum(d < -0.02 * abs(b[0])))
            n_bad = min(n_pos, n_neg)
            ok = n_bad <= max(1, int(0.05 * len(d)))
            ok_all &= ok
            items.append((
                "通道宽度沿程单调",
                ok,
                f"b_in={b[0]*1e3:.2f} mm, b_out={b[-1]*1e3:.2f} mm, 反向点数={n_bad}",
            ))

        for name, k in (("hub", self.kappa_hub()), ("shroud", self.kappa_shroud())):
            # 直段的曲率理论上是 0，数值上会在 0 附近抖，
            # 判"反号"前先把 |k| < 5% max 的点剔掉，否则噪声会被当成反曲。
            thr = 0.05 * float(np.max(np.abs(k))) if len(k) else 0.0
            kk = k[np.abs(k) > max(thr, 1e-9)]
            sgn = np.sign(kk)
            ok = len(np.unique(sgn)) <= 1
            ok_all &= ok
            items.append((
                f"{name} 曲率不反号",
                ok,
                f"kappa 范围 [{k.min():.2f}, {k.max():.2f}] 1/m, "
                f"有效点数={len(kk)}",
            ))

        return {"ok": bool(ok_all), "items": items}


# ============================================================
# 椭圆弧构造（默认）
# ============================================================
def _quad_point(s, r_in, r_out, z_in, z_out) -> tuple[float, float]:
    """
    二次贝塞尔：P0=(z_in,r_in), P1=(z_out,r_in), P2=(z_out,r_out)。
    展开后
        z = z_in + L_z*(2s - s^2)
        r = r_in + L_r*s^2
    性质：s=0 切向 (2L_z,0) 纯轴向，s=1 切向 (0,2L_r) 纯径向，
    两端导数连续、曲率连续，不存在"直段拼圆弧"的折角。
    r 在进口变化慢（s^2），正好是叶轮进口段要的形状。
    """
    L_z = z_out - z_in
    L_r = r_out - r_in
    return z_in + L_z * (2.0 * s - s * s), r_in + L_r * s * s


def _elliptic_point(s, r_in, r_out, z_in, z_out, lead) -> tuple[float, float]:
    """
    归一化进度 s in [0,1] 处的一点（进口直段 + 椭圆弧）。
    保留作对照；build_by_area 用的是 _quad_point，更光滑。
    """
    L_z = z_out - z_in
    z1 = z_in + lead * L_z
    if s <= lead:
        z = z_in + (s / lead) * (z1 - z_in) if lead > 1e-9 else z1
        return z, r_in
    t = (s - lead) / (1.0 - lead)
    th = 0.5 * math.pi * t
    a = L_z - lead * L_z
    b = r_out - r_in
    return z1 + a * math.sin(th), r_in + b * (1.0 - math.cos(th))


def build_by_area(
    r_hub_in: float,
    r_shroud_in: float,
    r_hub_out: float,
    r_shroud_out: float,
    z_in: float,
    z_out: float,
    lead_shroud: float = 0.55,
    blend: float = 1.0,
    n: int = 201,
) -> Meridian:
    """
    由 **目标面积分布** 反求 hub 型线（推荐用法）。

    顺序反过来：
      1) shroud 用椭圆弧（进口直段 + 90 度转弯），形状由 lead_shroud 控制；
      2) 面积按 A(s) 从 A_in 线性过渡到 A_out；
      3) hub 半径由 π*(r_s^2 - r_h^2) = A 直接解出：
             r_h(s) = sqrt( r_s(s)^2 - A(s)/pi )

    这个式子就是环形面积的定义，所以面积 **精确** 按目标分布走，
    不存在"先画几何再祈祷面积单调"的问题。A(s) 线性 -> 面积严格单调；
    想要前段面积降得慢（压低进口马赫），把 blend 调大（t = s**blend）。

    代价：hub 的形状不再是解析曲线，而是由面积分布驱动 —— 但这正是
    设计该有的顺序：面积（也就是 c_m 沿程）才是气动上要控的量，
    几何是它的结果。
    """
    s = np.linspace(0.0, 1.0, n)
    shroud = np.array([_quad_point(x, r_shroud_in, r_shroud_out, z_in, z_out)
                       for x in s])

    A_in = math.pi * (r_shroud_in ** 2 - r_hub_in ** 2)
    A_out = math.pi * (r_shroud_out ** 2 - r_hub_out ** 2)
    t = np.power(s, blend)
    A = A_in + (A_out - A_in) * t

    rs2 = shroud[:, 1] ** 2
    rh = np.sqrt(np.maximum(rs2 - A / math.pi, 1e-12))
    hub = np.column_stack([shroud[:, 0], rh])

    m = Meridian(hub=hub, shroud=shroud, s=s, kind="area-driven")
    _populate(m)
    return m


def build_elliptic(
    r_hub_in: float,
    r_shroud_in: float,
    r_hub_out: float,
    r_shroud_out: float,
    z_in: float,
    z_out: float,
    lead_hub: float = 0.30,
    lead_shroud: float = 0.70,
    n: int = 201,
    n_circ: int | None = None,
) -> Meridian:
    """
    进口直段 + 椭圆弧 + 径向出口。

    lead_hub < lead_shroud 是关键：hub 早转，shroud 晚转，
    两者错开后通道宽度从进口就开始收缩，面积才能单调递减。
    lead 越大，该侧直段越长、转弯越靠后。
    """
    s = np.linspace(0.0, 1.0, n)
    hub = np.array([_elliptic_point(x, r_hub_in, r_hub_out, z_in, z_out, lead_hub)
                    for x in s])
    shroud = np.array([_elliptic_point(x, r_shroud_in, r_shroud_out, z_in, z_out, lead_shroud)
                       for x in s])
    m = Meridian(hub=hub, shroud=shroud, s=s, kind="elliptic")
    _populate(m, n_circ)
    return m


def _populate(m: Meridian, n_circ: int | None = None) -> None:
    """
    用等参数准正交线量通道宽度：
        b(s) = |P_shroud(s) - P_hub(s)|,  A = 2*pi*r_m*b, r_m 取中点半径。
    这正是 TurboGrid 建立准正交线时用的对应关系，跟求解器的面积定义一致。
    默认对全部点计算（与 m.s 等长），n_circ 给定时才抽样。
    """
    n = m.hub.shape[0]
    idx = np.arange(n) if n_circ is None else np.linspace(0, n - 1, n_circ).astype(int)
    H = m.hub[idx]
    S = m.shroud[idx]
    b = np.linalg.norm(S - H, axis=1)
    r_m = 0.5 * (H[:, 1] + S[:, 1])
    m.centers = 0.5 * (H + S)
    m.widths = b
    m.area = 2.0 * math.pi * r_m * b


# ============================================================
# 贝塞尔构造（可选）
# ============================================================
def build_stage(
    r_shroud_in: float,
    r_hub_in: float,
    r_shroud_out: float,
    r_hub_out: float,
    z_in: float,
    z_out: float,
    theta_in_deg: float = 0.0,
    theta_out_deg: float = 90.0,
    kappa_in: float = 0.0,
    kappa_out: float = 0.0,
    lam_in: float | None = None,
    lam_out: float | None = None,
    n: int = 201,
    n_circ: int | None = None,
) -> Meridian:
    """
    五点四次贝塞尔版（论文方法）。保留是为了对照，
    默认不建议用于"面积要单调"的设计，原因见模块文档。
    """
    t_in = tangent_from_angle(theta_in_deg)
    t_out = tangent_from_angle(theta_out_deg)

    if lam_in is None:
        lam_in_hub = 0.32 * abs(z_out - z_in)
        lam_in_shroud = 0.68 * abs(z_out - z_in)
    else:
        lam_in_hub = lam_in_shroud = lam_in
    lam_out_hub = 0.50 * abs(r_hub_out - r_hub_in) if lam_out is None else lam_out
    lam_out_shroud = 0.50 * abs(r_shroud_out - r_shroud_in) if lam_out is None else lam_out

    c_shroud = solve_control_points(
        (z_in, r_shroud_in), (z_out, r_shroud_out),
        t_in, t_out, kappa_in, kappa_out, lam_in_shroud, lam_out_shroud)
    c_hub = solve_control_points(
        (z_in, r_hub_in), (z_out, r_hub_out),
        t_in, t_out, kappa_in, kappa_out, lam_in_hub, lam_out_hub)

    s = np.linspace(0.0, 1.0, n)
    m = Meridian(hub=curve(c_hub, s), shroud=curve(c_shroud, s), s=s,
                 hub_ctrl=c_hub, shroud_ctrl=c_shroud, kind="bezier")
    _populate(m, n_circ)
    return m


def from_control_points(hub: np.ndarray, shroud: np.ndarray,
                        n: int = 201, n_circ: int = 41) -> Meridian:
    """直接给定两组贝塞尔控制点（CFturbo 里手工拖出来的那种）"""
    s = np.linspace(0.0, 1.0, n)
    m = Meridian(hub=curve(hub, s), shroud=curve(shroud, s), s=s,
                 hub_ctrl=np.asarray(hub, float),
                 shroud_ctrl=np.asarray(shroud, float), kind="bezier")
    _populate(m, n_circ)
    return m


def from_points(hub_pts, shroud_pts, n_circ: int = 41) -> Meridian:
    """
    直接给两组点集（从 CFturbo 导出 / 从 .gtm 量出来的子午线）。
    要求两组点数量相同、且按同一进度排列。
    """
    hub = np.asarray(hub_pts, float)
    shroud = np.asarray(shroud_pts, float)
    if hub.shape != shroud.shape:
        raise ValueError(f"hub/shroud 点数不一致: {hub.shape} vs {shroud.shape}")
    s = np.linspace(0.0, 1.0, hub.shape[0])
    m = Meridian(hub=hub, shroud=shroud, s=s, kind="points")
    _populate(m, n_circ)
    return m
