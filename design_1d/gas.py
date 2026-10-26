"""
理想气体物性与一维热力关系。

单位约定：全程 SI —— Pa, K, kg/s, m, rad/s。
角度只在接口层用 degree，进入计算立刻转 rad。

比热用常值：空气在 300~450 K 区间 cp 从 1005 变到 1015（+1%），
对几何尺寸的影响 < 0.3%，远小于经验系数的散布，因此不值得引入变物性。
如果做高温或工质不是空气，把 cp/R 改掉即可。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IdealGas:
    """常比热理想气体。默认空气。"""

    R: float = 287.1      # J/(kg K)
    cp: float = 1004.5    # J/(kg K)

    @property
    def gamma(self) -> float:
        return self.cp / (self.cp - self.R)

    @property
    def g_exp(self) -> float:
        """gamma/(gamma-1)，等熵关系的指数。空气约 3.5。"""
        return self.gamma / (self.gamma - 1.0)

    # ---- 基本量 -------------------------------------------------
    def a(self, T: float) -> float:
        """当地音速 [m/s]"""
        return math.sqrt(self.gamma * self.R * T)

    def rho(self, p: float, T: float) -> float:
        """密度 [kg/m3]"""
        return p / (self.R * T)

    # ---- 总-静关系 ----------------------------------------------
    def T_from_T0_M(self, T0: float, M: float) -> float:
        return T0 / (1.0 + 0.5 * (self.gamma - 1.0) * M * M)

    def M_from_T(self, T: float, T0: float) -> float:
        """由 T0/T 反算马赫数"""
        return math.sqrt(2.0 * (T0 / T - 1.0) / (self.gamma - 1.0))

    def p_static_from_T(self, p0: float, T0: float, T: float, eta_p: float) -> float:
        """
        多变过程的静压：p/p0 = (T/T0)^(eta_p * gamma/(gamma-1))。

        eta_p = 1 退化为等熵关系。这是审计"总压/静压/总温"是否自洽的核心公式：
        反过来用可以解出实际多变效率，若 eta_p > 1 说明这组数不可能是同一个截面。
        """
        return p0 * (T / T0) ** (eta_p * self.g_exp)

    def eta_p_from_states(self, p0: float, T0: float, p: float, T: float) -> float:
        """由总/静状态反算多变效率。>1 或 <0.4 都说明数据有问题。"""
        if not (0.0 < p / p0 < 1.0) or T >= T0:
            return float("nan")
        return math.log(p / p0) / math.log(T / T0) / self.g_exp

    # ---- 焓 ------------------------------------------------------
    def dh_s(self, T0_in: float, PR: float) -> float:
        """等熵焓升 [J/kg]"""
        return self.cp * T0_in * (PR ** ((self.gamma - 1.0) / self.gamma) - 1.0)

    def PR_from_dh_s(self, T0_in: float, dh_s: float) -> float:
        return (1.0 + dh_s / (self.cp * T0_in)) ** self.g_exp

    def T0_from_dh(self, T0: float, dh: float) -> float:
        return T0 + dh / self.cp

    def dh_from_T0(self, T0_a: float, T0_b: float) -> float:
        return self.cp * (T0_b - T0_a)


def mass_flow(p: float, T: float, A: float, V: float) -> float:
    """连续性辅助：m_dot = rho * A * V"""
    return p / (287.1 * T) * A * V


def area_annulus(r_tip: float, r_hub: float) -> float:
    return math.pi * (r_tip ** 2 - r_hub ** 2)


def area_cyl(r: float, b: float) -> float:
    """轴向/径向环形流通面积 A = 2*pi*r*b"""
    return 2.0 * math.pi * r * b
