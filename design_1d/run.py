"""
命令行入口。

    python -m design_1d.run design   --case y7v2
    python -m design_1d.run audit    --case y7v2
    python -m design_1d.run meridian --case y7v2 --out meridian.png
    python -m design_1d.run all      --case y7v2

也可以在自己的脚本里直接 import 用：
    from design_1d import Case, design
    from design_1d.cases.y7v2 import DESIGN
    r = design(DESIGN)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from . import criteria as cr


# ============================================================
# 设计结果的判据
# ============================================================
def design_checks(r) -> list:
    i, d, v = r.imp, r.dif, r.vol
    out = []
    out += cr.impeller_checks(
        M_w1=i.M_w1,
        w2_over_w1=i.w2_over_w1,
        psi=r.case.psi,
        phi2=r.case.phi2,
        b2_over_r2=i.b2 / i.r2,
        beta2A_deg=i.beta2A_deg,
        beta2_flow_deg=i.beta2_flow_deg,
        slip_wiesner=i.slip_wiesner,
        slip_cfd=i.slip_wiesner,
        hub_tip_ratio=i.r1h / i.r1t,
        blockage_inlet=i.blockage_inlet,
        M2_rel=i.M2_rel,
    )
    out += cr.diffuser_checks(
        k3=d.r3 / i.r2,
        ky=d.r4 / d.r3,
        M3=d.M3,
        theta_deg=d.theta_deg,
        ak=d.b4 / d.b3,
        z_diff=d.z_diff,
        z_imp=r.case.z_total,
        solidity=d.solidity,
        alpha3_deg=d.alpha3_deg,
        alpha4_deg=d.alpha4_deg,
    )
    out += cr.volute_checks(M5=v.M5, area_ratio=v.area_ratio)
    return out


# ============================================================
# 画图
# ============================================================
def plot_meridian(imp, out_path: str, title: str = "Meridional channel"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    m = imp.meridian
    if m is None:
        print("  [warn] 没有子午线数据，跳过画图")
        return

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) 子午面
    hp = m.hub_pts()
    sp = m.shroud_pts()
    ax[0].plot(hp[:, 0] * 1e3, hp[:, 1] * 1e3, "b-", lw=2, label="Hub")
    ax[0].plot(sp[:, 0] * 1e3, sp[:, 1] * 1e3, "r-", lw=2, label="Shroud")
    if m.centers is not None:
        # 准正交线：每 5 条画一根
        n = len(m.centers)
        step = max(1, n // 8)
        for i in range(0, n, step):
            j = int(round(i / max(n - 1, 1) * (len(hp) - 1)))
            ax[0].plot([hp[j, 0] * 1e3, sp[j, 0] * 1e3],
                       [hp[j, 1] * 1e3, sp[j, 1] * 1e3],
                       "k-", lw=0.5, alpha=0.45)
        ax[0].plot(m.centers[:, 0] * 1e3, m.centers[:, 1] * 1e3,
                   "g.", ms=3, label="准正交线中点")
    ax[0].set_xlabel("z [mm]")
    ax[0].set_ylabel("r [mm]")
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title(f"{title} ({m.kind})", fontsize=10)
    ax[0].grid(alpha=0.3)

    # (2) 面积分布
    if m.area is not None:
        s = m.s if m.s is not None else np.linspace(0, 1, len(m.area))
        ax[1].plot(s, m.area * 1e4, "b-o", ms=3, lw=1.5)
        ax[1].set_xlabel("归一化进度")
        ax[1].set_ylabel("流通面积 A [cm²]")
        ax[1].set_title("流通面积沿程分布（应单调）", fontsize=10)
        ax[1].grid(alpha=0.3)
        ref = max(m.area[0], m.area[-1])
        overshoot = m.area.max() / ref - 1.0
        mono = overshoot <= 0.10
        direction = "递减" if m.area[-1] < m.area[0] else "递增"
        ax[1].text(0.05, 0.92,
                   f"单调性: {'OK' if mono else 'FAIL'} ({direction}, 过冲 {overshoot*100:.1f}%)",
                   transform=ax[1].transAxes,
                   color="green" if mono else "red", fontsize=9)

    # (3) 曲率
    u = m.s if m.s is not None else np.linspace(0, 1, len(m.hub))
    ax[2].plot(u, m.kappa_hub(), "b-", lw=1.5, label="Hub")
    ax[2].plot(u, m.kappa_shroud(), "r-", lw=1.5, label="Shroud")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlabel("归一化进度")
    ax[2].set_ylabel("曲率 κ [1/m]")
    ax[2].set_title("曲率（不应反号）", fontsize=10)
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  图已保存: {out_path}")


# ============================================================
# 子命令
# ============================================================
def cmd_design(case_name: str, as_json: bool = False, out_png: str | None = None):
    from .design import design, summary
    from .cases import y7v2  # noqa: F401  确保模块被导入

    mod = importlib.import_module(f"design_1d.cases.{case_name}")
    case = mod.DESIGN

    r = design(case)
    print(summary(r))
    checks = design_checks(r)
    print(cr.render(checks, title="-- 判据 --"))

    if out_png:
        plot_meridian(r.imp, out_png, title=f"{case_name} 一维设计子午线")

    if as_json:
        payload = {
            "case": {
                "mdot": case.mdot, "n_rpm": case.n_rpm,
                "PR_tt": case.PR_tt, "T0_in": case.T0_in, "p0_in": case.p0_in,
            },
            "impeller": {k: v for k, v in r.imp.__dict__.items()
                         if k != "meridian"},
            "diffuser": r.dif.__dict__,
            "volute": r.vol.__dict__,
            "checks": cr.to_dict(checks),
        }
        print("\n-- JSON --")
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_audit(case_name: str):
    from .audit import audit, summary
    mod = importlib.import_module(f"design_1d.cases.{case_name}")
    m = mod.MACHINE

    d, checks = audit(m)
    print(summary(m, d))
    print(cr.render(checks, title="-- 判据 --"))


def cmd_meridian(case_name: str, out_png: str):
    from .design import design
    mod = importlib.import_module(f"design_1d.cases.{case_name}")
    r = design(mod.DESIGN)
    plot_meridian(r.imp, out_png, title=f"{case_name} 子午线")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="design_1d",
        description="单级离心压缩机一维设计与逆向体检",
    )
    p.add_argument("cmd", choices=["design", "audit", "meridian", "all"],
                   help="design=正设计  audit=体检现有机器  meridian=画子午线  all=两者都跑")
    p.add_argument("--case", default="y7v2", help="算例名（cases/ 下的模块名）")
    p.add_argument("--json", action="store_true", help="design 时同时输出 JSON")
    p.add_argument("--out", default=None, help="图片输出路径")
    a = p.parse_args(argv)

    if a.cmd == "design":
        cmd_design(a.case, a.json, a.out)
    elif a.cmd == "audit":
        cmd_audit(a.case)
    elif a.cmd == "meridian":
        cmd_meridian(a.case, a.out or f"{a.case}_meridian.png")
    else:
        cmd_design(a.case, a.json, a.out)
        print("\n")
        cmd_audit(a.case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
