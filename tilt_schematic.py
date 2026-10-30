"""Срез через ось линзы при наклонной съёмке.

Две параболические полости, лучи пучка под углом θ к плоскости диска,
точки касания (видимые вершины) и их проекции на детектор.
Ползунки: θ, R1, R2, истинный зазор Δz, сдвиг осей Δb в плоскости наклона.
Выводятся истинный Δz и видимый зазор G.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

Z_MAX = 105.0     # полуразмер области по оси, мкм
Y_RANGE = 130.0   # полуширина области поперёк оси, мкм
L_DET = 170.0     # расстояние до детектора вдоль пучка, мкм

COL_APEX = "#b2336d"
COL_TAN = "#0b7a86"
COL_DET = "#b86e00"
COL_MAT = "#dfe5ea"


def geometry(theta_deg, r1, r2, dz, db):
    th = np.radians(theta_deg)
    s, c, t = np.sin(th), np.cos(th), np.tan(th)
    h1, h2, b1, b2 = dz / 2, -dz / 2, db / 2, -db / 2
    k = np.array([c, s])      # направление пучка в плоскости (y, z)
    ev = np.array([-s, c])    # направление v на детекторе
    a1, a2 = np.array([b1, h1]), np.array([b2, h2])
    t1 = np.array([b1 + r1 * t, h1 + r1 * t * t / 2])   # касание верхней полости
    t2 = np.array([b2 - r2 * t, h2 - r2 * t * t / 2])   # касание нижней полости
    g_visible = (t1 - t2) @ ev
    return dict(k=k, ev=ev, a1=a1, a2=a2, t1=t1, t2=t2, b1=b1, b2=b2, h1=h1, h2=h2,
                G=g_visible, apex_gap=(a1 - a2) @ ev)


def on_detector(p, k, ev):
    return L_DET * k + (p @ ev) * ev


def draw(ax, theta_deg, r1, r2, dz, db):
    ax.clear()
    g = geometry(theta_deg, r1, r2, dz, db)
    k, ev = g["k"], g["ev"]

    ax.fill_between([-Y_RANGE, Y_RANGE], -Z_MAX, Z_MAX, color=COL_MAT, zorder=0)
    y = np.linspace(-Y_RANGE, Y_RANGE, 600)
    for b, h, r, sign in ((g["b1"], g["h1"], r1, 1), (g["b2"], g["h2"], r2, -1)):
        z = h + sign * (y - b) ** 2 / (2 * r)
        z = np.clip(z, -Z_MAX, Z_MAX)
        ax.fill_between(y, z, sign * Z_MAX, color="white", zorder=1)
        ax.plot(y, z, color="black", lw=1.6, zorder=2)
        ax.axvline(b, color="0.5", lw=1, ls="-.", zorder=2)

    def ray(p, col, ls, lw):
        start, end = p - 230 * k, on_detector(p, k, ev)
        ax.plot([start[0], end[0]], [start[1], end[1]], color=col, ls=ls, lw=lw, zorder=3)

    for p in (g["a1"], g["a2"]):
        ray(p, COL_APEX, "--", 1.2)
        ax.plot(*p, "o", color=COL_APEX, ms=7, zorder=5)
    for p in (g["t1"], g["t2"]):
        ray(p, COL_TAN, "-", 1.8)
        ax.plot(*p, "x", color=COL_TAN, ms=11, mew=2.5, zorder=5)

    d1, d2 = L_DET * k - 120 * ev, L_DET * k + 120 * ev
    ax.plot([d1[0], d2[0]], [d1[1], d2[1]], color=COL_DET, lw=5, zorder=4)

    ax.set_xlim(-Y_RANGE - 10, L_DET * np.cos(np.radians(theta_deg)) + 60)
    ax.set_ylim(-Z_MAX - 30, Z_MAX + 30)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return g


def main():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.03, 0.30, 0.94, 0.66])
    info = fig.text(0.5, 0.25, "", ha="center", va="center", fontsize=15, family="monospace")

    specs = [("θ, °", 0, 35, 20, 0.5), ("R1, мкм", 20, 150, 50, 1), ("R2, мкм", 20, 150, 50, 1),
             ("Δz, мкм", 0, 60, 30, 0.5), ("Δb, мкм", -15, 15, 0, 0.5)]
    sliders = []
    for i, (name, lo, hi, val, step) in enumerate(specs):
        sax = fig.add_axes([0.15, 0.18 - i * 0.035, 0.7, 0.022])
        sliders.append(Slider(sax, name, lo, hi, valinit=val, valstep=step))

    def update(_=None):
        g = draw(ax, *(s.val for s in sliders))
        info.set_text(f"Δz истинный = {sliders[3].val:6.2f} мкм     G видимый = {g['G']:6.2f} мкм")
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)
    update()
    plt.show()


if __name__ == "__main__":
    main()
