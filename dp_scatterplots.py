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

from dpcomp_core.algorithm import dawa, ahp   # 只需要算法模块

from diffprivlib.tools import histogram2d

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========  单个机制的运行函数  =========


def run_dawa(points: pd.DataFrame, bins, epsilon: float, seed: int = 0) -> np.ndarray:
    """
    运行 2D DAWA：
    - points: 必须已经有 'x','y' 两列，连续型
    - bins:   可以是整数；如果传的是边界数组，会用 len(bins) - 1 当作 bin 数
    - epsilon: 隐私参数
    """
    # --- 1. 统一 bins，确保是 Python int ---
    if np.isscalar(bins):
        num_bins = int(bins)
    else:
        bins = np.asarray(bins)
        if bins.ndim != 1:
            raise ValueError("bins 只能是一维数组或标量")
        num_bins = int(len(bins) - 1)

    if num_bins <= 0:
        raise ValueError(f"非法的 bins 值: {num_bins}")

    # --- 2. 做一个干净的 2D 直方图 ---
    x = points["x"].to_numpy(dtype=float)
    y = points["y"].to_numpy(dtype=float)

    H, xedges, yedges = np.histogram2d(x, y, bins=num_bins)
    H = H.astype(float, copy=False)

    # --- 3. 调 DAWA 的 2D 引擎 ---
    engine = dawa.dawa2D_engine()
    # dpcomp 的 2D DAWA 接口是：Run(hist, epsilon, seed=0)
    # （不需要 workload，由内部自己构造）
    x_hat = engine.Run(H, float(epsilon), int(seed))

    # 保证是 numpy array
    return np.asarray(x_hat, dtype=float)


def run_ahp(points: pd.DataFrame, bins: int, epsilon: float) -> np.ndarray:
    """
    AHP 机制。
    """
    domain = (bins, bins)
    seed = 1
    shape_list = [(5, 5), (10, 10)]
    size = 5000

    w = workload.RandomRange(   # 这一块还是沿用你原来 notebook 的 AHP 用法
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


# =========  生成网格图的主函数  =========

def generate_dp_scatter_grid(
    points: pd.DataFrame,
    epsilons=(0.5, 0.1, 0.05, 0.01),
    mechanisms=("AHP", "DAWA", "Laplace"),
    bins: int = 64,
    consistent_colorscale: bool = True,
) -> go.Figure:
    """
    生成一个 rows = len(mechanisms), cols = len(epsilons) 的 Heatmap 网格，
    每个 cell 是对应机制 + epsilon 下的 2D 直方图。

    返回一个 Plotly Figure，供 Streamlit 直接 plotly_chart。
    points: 需要包含 'x', 'y' 两列
    """
    charts = {}
    all_vals = []

    epsilons = list(epsilons)
    mechanisms = list(mechanisms)

    for mech in mechanisms:
        for eps in epsilons:
            hist = compute_histogram(points, mech, bins, eps)
            charts[(mech, eps)] = hist
            all_vals.append(hist)

    # 统一颜色范围，方便比较
    if consistent_colorscale and all_vals:
        global_min = min(np.min(h) for h in all_vals)
        global_max = max(np.max(h) for h in all_vals)
        center = max(abs(global_min), abs(global_max))
    else:
        global_min = global_max = center = None

    fig = make_subplots(
        rows=len(mechanisms),
        cols=len(epsilons),
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )

    teal_colorscale = [
        [0.0, "rgb(255,255,255)"],
        [1.0, "rgb(3, 86, 94)"],
    ]

    for row, mech in enumerate(mechanisms, start=1):
        for col, eps in enumerate(epsilons, start=1):
            z = charts[(mech, eps)]

            fig.add_trace(
                go.Heatmap(
                    z=z,
                    colorscale=teal_colorscale,
                    zmin=-center if center is not None else None,
                    zmax=center if center is not None else None,
                    showscale=False,
                ),
                row=row,
                col=col,
            )

    # 顶部 ε 标签
    for col, eps in enumerate(epsilons, start=1):
        fig.add_annotation(
            xref="x domain",
            yref="paper",
            x=(col - 0.5) / len(epsilons),
            y=1.05,
            showarrow=False,
            text=f"ε={eps}",
            font=dict(size=18),
        )

    # 左侧机制标签
    for row, mech in enumerate(mechanisms, start=1):
        fig.add_annotation(
            xref="paper",
            yref="y domain",
            x=-0.03,
            y=(row - 0.5) / len(mechanisms),
            showarrow=False,
            text=mech,
            font=dict(size=18),
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
        height=600,
        width=1200,
        margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="rgb(255,255,255)",
        paper_bgcolor="rgb(255,255,255)",
    )

    return fig
