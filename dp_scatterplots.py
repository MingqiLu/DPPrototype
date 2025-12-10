# dp_scatterplots.py
"""
封装 AHP / DAWA / Laplace 等 DP 散点图（2D 直方图）生成逻辑，
供 Streamlit 页面调用。

本文件假设：
- 已经 git clone 并安装好 dpcomp_core
- 已经安装 diffprivlib
"""

from __future__ import division, print_function

import numpy as np
import pandas as pd

from dpcomp_core.algorithm import dawa, ahp
from dpcomp_core import workload

from diffprivlib.tools import histogram2d

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========  单个机制的运行函数  =========


def run_dawa(points: pd.DataFrame, bins, epsilon: float, seed: int = 0) -> np.ndarray:
    """
    运行 2D DAWA（使用 dpcomp_core.algorithm.dawa.dawa2D_engine）：
    - points: DataFrame，必须已经有 'x','y' 两列（数值型）
    - bins:   可以是整数；如果传的是 bin 边界数组，则用 len(bins)-1 作为 bin 数
    - epsilon: 隐私参数
    - seed:    随机种子
    """
    # ---------- 1. 统一 bins，拿到一个 Python int ----------
    if np.isscalar(bins):
        num_bins = int(bins)
    else:
        bins = np.asarray(bins)
        if bins.ndim != 1:
            raise ValueError("bins 只能是一维数组或标量")
        num_bins = int(len(bins) - 1)

    if num_bins <= 0:
        raise ValueError(f"非法的 bins 值: {num_bins}")

    # ---------- 2. 在原始网格上做一个 2D 直方图 ----------
    x = points["x"].to_numpy(dtype=float)
    y = points["y"].to_numpy(dtype=float)

    H, xedges, yedges = np.histogram2d(x, y, bins=num_bins)
    H = H.astype(float, copy=False)   # H.shape = (n1, n2)

    # ---------- 3. 构造 DAWA 所需的 workload Q ----------
    # 根据 dawa2D_engine.Run 的实现，它内部会：
    #   n1, n2 = x.shape
    #   d = 2**ceil(log2(max(n1, n2)))
    #   query.asArray([d, d])
    #
    # 为了和内部逻辑兼容，我们也用同样的 d 来设置 domain_shape。
    n1, n2 = H.shape
    d = 2 ** int(np.ceil(np.log2(max(n1, n2))))   # >= max(n1, n2)，且是 2 的幂
    domain_shape = (d, d)

    # 和 AHP 一致的 workload 结构，只是 domain_shape 换成 (d,d)
    shape_list = [(5, 5), (10, 10)]
    size = 5000

    Q = workload.RandomRange(
        shape_list=shape_list,
        domain_shape=domain_shape,
        size=size,
        seed=int(seed),
    )

    # ---------- 4. 跑 DAWA 2D 引擎 ----------
    engine = dawa.dawa2D_engine()     # ratio 默认 0.25

    # 严格按照源码签名传参：Run(self, Q, x, epsilon, seed)
    x_hat = engine.Run(Q, H, float(epsilon), int(seed))

    # ---------- 5. 返回 numpy 数组 ----------
    return np.asarray(x_hat, dtype=float)


def run_ahp(points: pd.DataFrame, bins: int, epsilon: float) -> np.ndarray:
    """
    AHP 机制（使用 dpcomp_core.algorithm.ahp.ahpND_engine）。
    """
    domain = (bins, bins)
    seed = 1
    shape_list = [(5, 5), (10, 10)]
    size = 5000

    w = workload.RandomRange(
        shape_list=shape_list,
        domain_shape=domain,
        size=size,
        seed=seed,
    )

    engine = ahp.ahpND_engine(ratio=0.85, eta=0.35)
    H, xedges, yedges = np.histogram2d(points["x"], points["y"], bins=bins)
    H = H.T
    x_hat = engine.Run(w, H, epsilon, seed)
    return x_hat


def run_laplace(points: pd.DataFrame, bins: int, epsilon: float) -> np.ndarray:
    """
    使用 diffprivlib 的 histogram2d 做 Laplace 直方图。
    """
    dp_hist, xedges, yedges = histogram2d(
        points["x"],
        points["y"],
        epsilon=epsilon,
        bins=bins,
        density=False,
    )
    return dp_hist.T


def run_original(points: pd.DataFrame, bins: int) -> np.ndarray:
    """
    不加噪的原始直方图。
    """
    H, xedges, yedges = np.histogram2d(points["x"], points["y"], bins=bins)
    return H.T


# =========  统一的调度函数  =========

_MECH_FUNC_MAP = {
    "Original": run_original,
    "AHP": run_ahp,
    "DAWA": run_dawa,
    "Laplace": run_laplace,
}


def compute_histogram(points, mechanism: str, bins: int, epsilon: float):
    """
    根据机制名字调用对应的函数。
    mechanism 支持: "Original", "AHP", "DAWA", "Laplace"
    """
    if mechanism not in _MECH_FUNC_MAP:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    if mechanism == "Original":
        # Original 不需要 epsilon
        return _MECH_FUNC_MAP[mechanism](points, bins)
    else:
        return _MECH_FUNC_MAP[mechanism](points, bins, epsilon)


# =========  返回「一系列图片」的主函数  =========

def generate_dp_scatter_figs(
    points: pd.DataFrame,
    epsilons=(0.5, 0.1, 0.05, 0.01),
    mechanisms=("AHP", "DAWA", "Laplace"),
    bins=(32, 64),
    consistent_colorscale: bool = True,
    progress_callback=None,
):
    """
    生成一系列 Plotly Figure，每个 figure 对应一个 (mechanism, epsilon, bin)。

    返回：
        一个 list，其中每个元素是
        {"mechanism": <str>, "epsilon": <float>, "figure": <go.Figure>}
    """
    charts = {}
    all_vals = []

    epsilons = list(epsilons)
    mechanisms = list(mechanisms)

    total = len(mechanisms) * len(epsilons)
    done = 0

    # 先算出所有直方图（这一步最耗时），顺便可以更新进度条
    for mech in mechanisms:
        for eps in epsilons:
            for bin in bins:
                hist = compute_histogram(points, mech, bin, eps)
                charts[(mech, eps, bin)] = hist
                all_vals.append(hist)

                done += 1
                if progress_callback is not None:
                    progress_callback(done / total)

    # 统一颜色范围，方便比较
    if consistent_colorscale and all_vals:
        global_min = min(np.min(h) for h in all_vals)
        global_max = max(np.max(h) for h in all_vals)
        center = max(abs(global_min), abs(global_max))
    else:
        global_min = global_max = center = None

    teal_colorscale = [
        [0.0, "rgb(255,255,255)"],
        [1.0, "rgb(3, 86, 94)"],
    ]

    fig_items = []

    for mech in mechanisms:
        for eps in epsilons:
            for bin in bins:
                z = charts[(mech, eps, bin)]

                fig = go.Figure(
                    data=[
                        go.Heatmap(
                            z=z,
                            colorscale=teal_colorscale,
                            zmin=-center if center is not None else None,
                            zmax=center if center is not None else None,
                            showscale=False,
                        )
                    ]
                )

                fig.update_xaxes(
                    showline=True,
                    linewidth=1,
                    linecolor="darkgrey",
                    mirror=True,
                    showticklabels=False,
                )
                fig.update_yaxes(
                    showline=True,
                    linewidth=1,
                    linecolor="darkgrey",
                    mirror=True,
                    showticklabels=False,
                )

                fig.update_layout(
                    height=250,
                    width=250,
                    margin=dict(l=20, r=10, t=10, b=20),
                    plot_bgcolor="rgb(255,255,255)",
                    paper_bgcolor="rgb(255,255,255)",
                )

                fig_items.append(
                    {"mechanism": mech, "epsilon": eps, "bin": bin, "figure": fig}
                )

    return fig_items