# ---------- helpers for sigma guides ----------
def add_sigma_guides(ax, mu, std, one_sigma_alpha=0.10, line_alpha=0.35, mu_label=None, label=None, mu_color='red', color='blue'):
    """
    在圖上加上 μ±1σ 淡色區間、以及 μ、μ±1σ、μ±2σ 參考線。
    """
    ymin, ymax = ax.get_ylim()

    # ±1σ淡色區間
    ax.axhspan(mu - std, mu + std, alpha=one_sigma_alpha, color=color)

    # μ與σ參考線
    ax.axhline(mu, linestyle='--', linewidth=1.5, label=mu_label, color=mu_color)
    for k in (1, 2):
        ax.axhline(mu + k*std, linestyle=':', alpha=line_alpha, linewidth=1.2, 
                   color=color, label=label if k == 1 else None)
        ax.axhline(mu - k*std, linestyle=':', alpha=line_alpha, linewidth=1.2, 
                   color=color)

    # 保持原本y範圍（避免被新元素改動）
    ax.set_ylim(ymin, ymax)

def add_zscore_right_axis(ax, mu, std):
    """
    在右側加上 z-score 軸（標準化差異）。
    """
    def y_to_z(y): return (y - mu) / std
    def z_to_y(z): return z * std + mu
    secax = ax.secondary_yaxis('right', functions=(y_to_z, z_to_y))
    secax.set_ylabel('Standardized difference (σ)')
    secax.set_yticks([-2, -1, 0, 1, 2])
    return secax