"""
判据表：把"这个几何合不合理"变成可量化的 PASS / WARN / FAIL。

每条判据都带来源，区分三类：
  hard —— 有明确物理或文献依据，超了基本就是设计错误
  soft —— 经验范围，超了要解释，不一定是错
  info —— 只报告数值，不做判定（依赖具体机型，没有普适范围）

来源缩写：
  [C25]  崔晓杰(2025) 第二章，单级离心压缩机一维设计
  [A]    Aungier, Centrifugal Compressors: A Strategy for Aerodynamic
         Design and Analysis, ASME Press, 2000
  [C]    Cumpsty, Compressor Aerodynamics
  [W]    Wiesner (1967) 滑移系数
  [DH]   de Haller 判据（相对速度沿程减速）
  [J]    工程经验值，本工具作者标注，超出范围需验证
"""

from __future__ import annotations

from dataclasses import dataclass

PASS, WARN, FAIL, INFO, NA = "PASS", "WARN", "FAIL", "INFO", "N/A"


@dataclass
class Check:
    key: str
    name: str
    value: float | None
    lo: float | None
    hi: float | None
    unit: str = ""
    source: str = ""
    note: str = ""
    tol: float = 0.05
    kind: str = "hard"          # hard / soft / info

    @property
    def status(self) -> str:
        if self.value is None:
            return NA
        if self.kind == "info":
            return INFO
        v = self.value
        if self.lo is not None and v < self.lo:
            return FAIL if v < self.lo * (1 - self.tol) else WARN
        if self.hi is not None and v > self.hi:
            return FAIL if v > self.hi * (1 + self.tol) else WARN
        return PASS

    @property
    def ref(self) -> str:
        if self.lo is not None and self.hi is not None:
            return f"{self.lo:g} ~ {self.hi:g}"
        if self.lo is not None:
            return f">= {self.lo:g}"
        if self.hi is not None:
            return f"<= {self.hi:g}"
        return "-"

    def fmt_value(self) -> str:
        if self.value is None:
            return "-"
        v = self.value
        if abs(v) >= 1000 or (v != 0 and abs(v) < 0.01):
            return f"{v:.3e}"
        return f"{v:.4g}"


# ============================================================
# 叶轮
# ============================================================
def impeller_checks(
    M_w1: float | None = None,
    w2_over_w1: float | None = None,
    psi: float | None = None,
    phi2: float | None = None,
    b2_over_r2: float | None = None,
    beta2A_deg: float | None = None,
    beta2_flow_deg: float | None = None,
    slip_wiesner: float | None = None,
    slip_cfd: float | None = None,
    hub_tip_ratio: float | None = None,
    blockage_inlet: float | None = None,
    M2_rel: float | None = None,
) -> list[Check]:
    out: list[Check] = []

    out.append(Check(
        "M_w1", "叶尖进口相对马赫 M_w1", M_w1, None, 0.90, "-",
        "[C25][A]", "亚音速设计上限；跨音速叶轮可到 1.0 但需激波管理",
    ))
    out.append(Check(
        "w2/w1", "相对速度减速比 w2/w1", w2_over_w1, 0.72, 1.0, "-",
        "[DH]", "de Haller 判据，<0.72 意味着扩压度过大、易分离",
    ))
    out.append(Check(
        "psi", "负荷系数 psi = dh/U2^2", psi, 0.50, 0.68, "-",
        "[A]", "低于 0.5 叶轮没吃饱，高于 0.68 后弯不足、出口易分离",
        kind="soft",
    ))
    out.append(Check(
        "phi2", "出口流量系数 phi2 = c_r2/U2", phi2, 0.20, 0.35, "-",
        "[A]", "偏低 -> b2 太宽、轮盘摩擦与二次流损失大；偏高 -> 出口堵塞",
        kind="soft",
    ))
    out.append(Check(
        "b2/r2", "出口宽度比 b2/r2", b2_over_r2, 0.03, 0.09, "-",
        "[C25]", "常规 0.03~0.06，低压比大流量机型可到 0.09。"
                 "注意 b2/r2 不是独立选择，它由 phi2、mdot、rho2、r2 唯一决定，"
                 "真正的自变量是 phi2",
        kind="soft",
    ))
    out.append(Check(
        "beta2A", "叶片出口安装角 beta2A", beta2A_deg, 20.0, 50.0, "deg",
        "[C25]", "论文取 40~45；后弯角越大效率越高、压比越低",
        kind="soft",
    ))
    if beta2A_deg is not None and beta2_flow_deg is not None:
        out.append(Check(
            "deviation", "落后角 delta = beta2A - beta2_flow",
            beta2A_deg - beta2_flow_deg, 2.0, 12.0, "deg",
            "[C]", "偏离 2~12 度以外说明叶片数或后弯角与滑移不自洽",
            kind="soft",
        ))
    if slip_wiesner is not None and slip_cfd is not None:
        out.append(Check(
            "slip_diff", "滑移系数偏差 |mu_Wiesner - mu_CFD|",
            abs(slip_wiesner - slip_cfd), None, 0.05, "-",
            "[W]", ">0.05 说明实际做功与一维预测不符，先查叶片数 z 和 beta2A",
        ))
    out.append(Check(
        "hub_tip", "轮毂比 r1h/r1t", hub_tip_ratio, 0.30, 0.55, "-",
        "[A]", "过小 -> 轮毂处叶片稠密、堵塞；过大 -> 进口面积不够",
        kind="soft",
    ))
    out.append(Check(
        "blockage", "进口有效面积比 (1 - 叶片堵塞)", blockage_inlet, 0.85, 1.0, "-",
        "[A]", "<0.85 说明叶片太厚或叶片数太多，进口被堵",
    ))
    out.append(Check(
        "M2_rel", "出口相对马赫 M_w2", M2_rel, None, 1.10, "-",
        "[A]", "半开式叶轮出口相对马赫超过 1.1 会在叶顶间隙产生强激波",
        kind="soft",
    ))
    return out


# ============================================================
# 扩压器
# ============================================================
def diffuser_checks(
    k3: float | None = None,
    ky: float | None = None,
    M3: float | None = None,
    theta_deg: float | None = None,
    ak: float | None = None,
    z_diff: int | None = None,
    z_imp: int | None = None,
    solidity: float | None = None,
    alpha3_deg: float | None = None,
    alpha4_deg: float | None = None,
    theta_is_estimate: bool = False,
) -> list[Check]:
    out: list[Check] = []

    out.append(Check(
        "k3", "无叶段长度比 k3 = r3/r2", k3, 1.08, 1.20, "-",
        "[C25]", "这是最容易被忽略又最要命的一条：k3≈1.0 意味着叶片扩压器"
                 "直接贴在叶轮出口，射流尾迹没机会掺混，前缘必然冲击",
    ))
    out.append(Check(
        "ky", "扩压器半径比 ky = r4/r3", ky, 1.20, 1.40, "-",
        "[C25]", "决定扩压器承担的压升份额；>1.4 面积增长过快易分离",
    ))
    out.append(Check(
        "M3", "扩压器进口马赫 M3", M3, None, 0.85, "-",
        "[C25][A]", "注意这里应该是**质量平均**值；局部峰值可以更高，"
                    "但平均值超过 0.85 说明叶轮出口速度没降下来",
    ))
    out.append(Check(
        "theta", "当量扩张角 theta", theta_deg, 8.0, 10.0, "deg",
        "[C25]",
        ">10 度边界层易分离，<8 度叶片过长、摩擦损失大"
        + ("。⚠ 叶片角未提供，此值按 beta≈alpha 估算，只作量级参考"
           if theta_is_estimate else ""),
        kind="info" if theta_is_estimate else "soft",
    ))
    out.append(Check(
        "ak", "流道深度比 a_out/a_in", ak, None, 2.5, "-",
        "[C25]", "等宽扩压器 a_out/a_in = 1，永远满足；变宽时要查",
        kind="soft",
    ))
    out.append(Check(
        "solidity", "叶片稠度 sigma = z*L/(2*pi*r_m)", solidity, 1.0, 1.7, "-",
        "[A]", "<1.0 叶片太少、气流不跟随；>1.7 摩擦损失大",
        kind="soft",
    ))
    if z_diff is not None and z_imp is not None:
        g = _gcd(z_diff, z_imp)
        out.append(Check(
            "z_coprime", "扩压器/叶轮叶片数最大公约数", float(g), None, 2.0, "-",
            "[C25]", f"z_diff={z_diff}, z_imp={z_imp}；公约数 >2 会激发强声学共振",
        ))
    if alpha3_deg is not None and alpha4_deg is not None:
        out.append(Check(
            "dalpha", "气流角变化 alpha4 - alpha3",
            alpha4_deg - alpha3_deg, -20.0, 0.0, "deg",
            "[A]", "从径向测量；扩压器内气流角应减小（更趋切向）",
            kind="soft",
        ))
    return out


# ============================================================
# 蜗壳
# ============================================================
def volute_checks(
    M5: float | None = None,
    A_out: float | None = None,
    A_in: float | None = None,
    area_ratio: float | None = None,
    tongue_gap_ratio: float | None = None,
) -> list[Check]:
    out: list[Check] = []
    out.append(Check(
        "M5", "蜗壳出口马赫 M5", M5, None, 0.15, "-",
        "[A]", "出口动压占静压的比例；>0.15 说明出口管太细、动能没回收",
    ))
    out.append(Check(
        "A_ratio", "蜗壳出口/进口面积比", area_ratio, 1.2, 3.5, "-",
        "[J]", "经验范围，与 M5 目标强相关：M5 从 0.15 降到 0.05，"
               "这个比值会翻一倍。只作参考",
        kind="soft",
    ))
    out.append(Check(
        "tongue", "蜗舌间隙比 r_tongue/r2", tongue_gap_ratio, 1.05, 1.20, "-",
        "[A]", "太小 -> 蜗舌处强非定常激励、噪声与振动；太大 -> 泄漏流回流",
        kind="soft",
    ))
    return out


# ============================================================
# 数值 / 后处理自洽性
# ============================================================
def consistency_checks(
    mdot_in: float | None = None,
    mdot_out: float | None = None,
    eta_p_imp: float | None = None,
    T0_drift_diffuser: float | None = None,
    T0_drift_volute: float | None = None,
    cm_continuity: float | None = None,
    cm_cfd: float | None = None,
) -> list[Check]:
    out: list[Check] = []
    if mdot_in is not None and mdot_out is not None and mdot_in != 0:
        out.append(Check(
            "mass_balance", "全机质量不平衡 |m_in-m_out|/m_in",
            abs(mdot_in - mdot_out) / mdot_in * 100.0, None, 0.5, "%",
            "[J]", ">0.5% 说明进出口边界过约束，先别谈效率",
        ))
    out.append(Check(
        "eta_p_imp", "叶轮多变效率（由 p0/T0/p/T 反算）",
        eta_p_imp, 0.80, 1.0, "-",
        "[J]", ">1.0 或 <0.8 说明这组总/静参数不是同一个截面、或取自不同迭代步",
        tol=0.001,   # 多变效率超过 1 在物理上不可能，不留容差
    ))
    if T0_drift_diffuser is not None:
        out.append(Check(
            "T0_diff", "扩压器总温漂移（无功耗部件应为 0）",
            abs(T0_drift_diffuser), None, 0.5, "K",
            "[J]", "无做功部件 T0 必须沿程不变；涨了就是质量不平衡的能量投影",
        ))
    if T0_drift_volute is not None:
        out.append(Check(
            "T0_vol", "蜗壳总温漂移",
            abs(T0_drift_volute), None, 0.5, "K",
            "[J]", "同上",
        ))
    if cm_continuity is not None and cm_cfd is not None and cm_continuity != 0:
        out.append(Check(
            "cm_check", "子午速度：连续性 vs CFD 偏差",
            abs(cm_cfd - cm_continuity) / cm_continuity * 100.0, None, 10.0, "%",
            "[J]", "偏差大说明 A 或 rho 用错（例如把直径当半径），"
                   "或者 CFD 取的是局部值而非面积平均",
        ))
    return out


# ============================================================
# 渲染
# ============================================================
def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


_ICON = {PASS: " OK ", WARN: "WARN", FAIL: "FAIL", INFO: "info", NA: " -- "}


def render(checks: list[Check], title: str = "") -> str:
    lines = []
    if title:
        lines.append("")
        lines.append(title)
    w_name = max((len(c.name) for c in checks), default=10)
    w_name = min(max(w_name, 14), 42)
    head = (f"  {'判据':<{w_name}} {'当前值':>10} {'推荐':>12} {'单位':<5} "
            f"{'状态':<5}  依据")
    lines.append("-" * (len(head) + 40))
    lines.append(head)
    lines.append("-" * (len(head) + 40))
    for c in checks:
        st = c.status
        name = c.name if len(c.name) <= w_name else c.name[: w_name - 1] + "…"
        lines.append(
            f"  {name:<{w_name}} {c.fmt_value():>10} {c.ref:>12} "
            f"{c.unit:<5} {_ICON[st]:<5}  {c.source}"
        )
        if c.note and st in (FAIL, WARN):
            lines.append(f"  {'':<{w_name}} └─ {c.note}")
    lines.append("-" * (len(head) + 40))
    n_fail = sum(1 for c in checks if c.status == FAIL)
    n_warn = sum(1 for c in checks if c.status == WARN)
    n_pass = sum(1 for c in checks if c.status == PASS)
    lines.append(f"  合计：PASS {n_pass} / WARN {n_warn} / FAIL {n_fail}")
    return "\n".join(lines)


def to_dict(checks: list[Check]) -> list[dict]:
    return [
        {
            "key": c.key, "name": c.name, "value": c.value,
            "lo": c.lo, "hi": c.hi, "unit": c.unit,
            "status": c.status, "source": c.source,
        }
        for c in checks
    ]
