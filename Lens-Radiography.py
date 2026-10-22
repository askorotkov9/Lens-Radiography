import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import tkinter as tk
from tkinter import filedialog
from PIL import Image

PIXEL_SIZE = 1.95  # 2.02
gray_level_min = 4210 
gray_level_max = 6683

SCALE_BAR_POSITIONS = {
    "Lower Right": "lower right",
    "Lower Left": "lower left",
    "Upper Right": "upper right",
    "Upper Left": "upper left",
}
SCALE_BAR_COLORS = ["White", "Black", "Yellow", "Red", "Cyan"]
TILT_ANGLE = 20.0  # угол между пучком и плоскостью диска линзы, град


def reconstruct_3d(p0, p90, theta_deg, sign=None):
    """3D-геометрия пары параболоидов по проекциям 0° и 90°.

    p0, p90: dict(u, G, R_up, R_down) — измерения на снимке:
      u      — смещение вершин поперёк оси (мкм, со знаком);
      G      — видимый зазор между вершинами вдоль оси (мкм);
      R_up, R_down — видимые радиусы параболоидов (мкм).
    theta_deg: наклон пучка относительно плоскости, перпендикулярной оси.
    sign: ±1 — знак связи продольного смещения с наклоном; None = подбор
      по согласованности двух проекций.

    Модель (параллельный пучок, оси параболоидов параллельны):
      u(0°) = Δx,  u(90°) = ±Δy,  R_вид = R / cosθ,
      G = Δz·cosθ − t·sinθ − (R1 + R2)·sinθ·tgθ / 2,
    где t — смещение осей вдоль направления наклона (t0 = −s·u90, t90 = s·u0).
    """
    th = np.radians(theta_deg)
    c, sn, tg = np.cos(th), np.sin(th), np.tan(th)
    u0, u90 = p0['u'], p90['u']

    def corrected(s):
        # A = G + t·sinθ = Δz·cosθ − ΣR·sinθ·tgθ/2, должно совпасть в обеих проекциях
        return p0['G'] - s * sn * u90, p90['G'] + s * sn * u0

    if sign is None:
        sign = min((1, -1), key=lambda s: abs(np.subtract(*corrected(s))))
    A0, A90 = corrected(sign)

    R_up = c * (p0['R_up'] + p90['R_up']) / 2
    R_down = c * (p0['R_down'] + p90['R_down']) / 2
    sum_R = R_up + R_down

    dz0 = (A0 + sum_R * sn * tg / 2) / c
    dz90 = (A90 + sum_R * sn * tg / 2) / c
    dz = (dz0 + dz90) / 2
    e = np.hypot(u0, u90)
    return dict(
        sign=sign, dx=u0, dy=u90, e=e, dz=dz, dz0=dz0, dz90=dz90,
        D=np.hypot(e, dz), d_min=dz + e ** 2 / (2 * sum_R),
        R_up=R_up, R_down=R_down,
        R_up_0=c * p0['R_up'], R_up_90=c * p90['R_up'],
        R_down_0=c * p0['R_down'], R_down_90=c * p90['R_down'],
    )


class ParabolaAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("X-Ray Lens Profile Analyzer")
        self.root.geometry("1300x900")
        self.root.configure(bg="#2c3e50")

        self.raw_image = None
        self.pixel_size = tk.DoubleVar(value=PIXEL_SIZE)
        self.vmin = tk.DoubleVar(value=gray_level_min)
        self.vmax = tk.DoubleVar(value=gray_level_max)

        self.clicked_points = []
        self.apexes = []
        self.fitted_plots = []
        self.fits = []  # dict(x0, y0, angle, a, R) для каждого фита

        # 3D-реконструкция по двум проекциям
        self.theta = tk.DoubleVar(value=TILT_ANGLE)
        self.sign_mode = tk.StringVar(value="Auto")
        self.projections = {0: None, 90: None}
        self.win_3d = None

        # Scale bar (как Analyze > Tools > Scale Bar в ImageJ)
        self.sb_show = tk.BooleanVar(value=True)
        self.sb_length = tk.DoubleVar(value=0)  # 0 = auto
        self.sb_position = tk.StringVar(value="Lower Right")
        self.sb_color = tk.StringVar(value="White")
        self.sb_thickness = tk.DoubleVar(value=4)  # px на экране
        self.sb_font_size = tk.DoubleVar(value=12)
        self.sb_hide_text = tk.BooleanVar(value=False)
        self.sb_background = tk.BooleanVar(value=False)
        self.scale_bar = None

        self.first_open = True
        self.setup_ui()

    def setup_ui(self):
        self.main_frame = tk.Frame(self.root, bg="#2c3e50")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(self.main_frame, bg="black")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.left_panel)
        self.toolbar.update()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.root.bind('<Return>', lambda e: self.fit_parabola() if (e.state & 0x1) else None)

        self.side_panel = tk.Frame(self.main_frame, width=320, bg="#34495e", padx=20, pady=10)
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(self.side_panel, text="ANALYSIS", fg="white", bg="#34495e", font=("Arial", 14, "bold")).pack(pady=(0, 10))

        self.create_input("Pixel Size (um):", self.pixel_size)
        self.create_input("Vmin:", self.vmin)
        self.create_input("Vmax:", self.vmax)

        tk.Button(self.side_panel, text="Load TIFF File", command=self.load_tiff, bg="#ecf0f1", height=2).pack(fill=tk.X, pady=10)
        tk.Button(self.side_panel, text="Update Display", command=self.update_plot, bg="#bdc3c7").pack(fill=tk.X, pady=5)

        tk.Label(self.side_panel, text="Controls", fg="#bdc3c7", bg="#34495e").pack(pady=(20, 5))
        tk.Label(self.side_panel, text="Shift + Click to set point\nShift + Enter to Fit", fg="#f1c40f", bg="#34495e", font=("Arial", 9, "italic")).pack()

        tk.Button(self.side_panel, text="Clear Current Points", command=self.clear_current_selection, bg="#95a5a6").pack(fill=tk.X, pady=2)
        tk.Button(self.side_panel, text="Fit Parabola", command=self.fit_parabola, bg="#5dade2", fg="white", font=("Arial", 11, "bold"), height=2).pack(fill=tk.X, pady=10)
        tk.Button(self.side_panel, text="Reset All", command=self.reset_all, bg="#e74c3c", fg="white").pack(fill=tk.X, pady=5)

        tk.Button(self.side_panel, text="3D Reconstruction (0° / 90°)...", command=self.open_3d_window,
                  bg="#8e44ad", fg="white").pack(fill=tk.X, pady=(10, 0))

        self.create_scale_bar_controls()

        tk.Frame(self.side_panel, height=2, bg="#5dade2").pack(fill=tk.X, pady=15)
        tk.Label(self.side_panel, text="Calculated Delta X (um):", fg="white", bg="#34495e", font=("Arial", 11)).pack()
        self.delta_label = tk.Label(self.side_panel, text="---", fg="#40e0d0", bg="#34495e", font=("Arial", 30, "bold"))
        self.delta_label.pack()
        tk.Label(self.side_panel, text="Apex gap along axis (um):", fg="white", bg="#34495e", font=("Arial", 10)).pack()
        self.gap_label = tk.Label(self.side_panel, text="---", fg="#40e0d0", bg="#34495e", font=("Arial", 16, "bold"))
        self.gap_label.pack()

    def create_input(self, label_text, var):
        frame = tk.Frame(self.side_panel, bg="#34495e")
        frame.pack(fill=tk.X, pady=5)
        tk.Label(frame, text=label_text, fg="white", bg="#34495e", width=15, anchor="w").pack(side=tk.LEFT)
        tk.Entry(frame, textvariable=var, width=10, justify='center').pack(side=tk.RIGHT)

    def create_scale_bar_controls(self):
        box = tk.LabelFrame(self.side_panel, text="Scale Bar", fg="white", bg="#34495e", padx=8, pady=5)
        box.pack(fill=tk.X, pady=(15, 0))
        opts = dict(bg="#34495e", fg="white", selectcolor="#2c3e50",
                    activebackground="#34495e", activeforeground="white",
                    command=self.update_plot)

        tk.Checkbutton(box, text="Show", variable=self.sb_show, **opts).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(box, text="Background", variable=self.sb_background, **opts).grid(row=0, column=1, sticky="w")
        tk.Checkbutton(box, text="Hide text", variable=self.sb_hide_text, **opts).grid(row=1, column=0, sticky="w")

        rows = [("Length (um, 0=auto):", self.sb_length),
                ("Thickness (px):", self.sb_thickness),
                ("Font size:", self.sb_font_size)]
        for i, (text, var) in enumerate(rows, start=2):
            tk.Label(box, text=text, fg="white", bg="#34495e", anchor="w").grid(row=i, column=0, sticky="w")
            entry = tk.Entry(box, textvariable=var, width=8, justify='center')
            entry.grid(row=i, column=1, sticky="e", pady=1)
            entry.bind('<Return>', lambda e: self.update_plot())
            entry.bind('<FocusOut>', lambda e: self.update_plot())

        for i, (text, var, values) in enumerate([("Location:", self.sb_position, list(SCALE_BAR_POSITIONS)),
                                                  ("Color:", self.sb_color, SCALE_BAR_COLORS)], start=5):
            tk.Label(box, text=text, fg="white", bg="#34495e", anchor="w").grid(row=i, column=0, sticky="w")
            menu = tk.OptionMenu(box, var, *values, command=lambda _: self.update_plot())
            menu.config(width=10, bg="#ecf0f1", highlightthickness=0)
            menu.grid(row=i, column=1, sticky="e", pady=1)
        box.columnconfigure(1, weight=1)

    @staticmethod
    def nice_length(target):
        """Ближайшая «круглая» длина 1/2/5 x 10^n, не превышающая target."""
        if target <= 0:
            return 1.0
        exp = np.floor(np.log10(target))
        for m in (5, 2, 1):
            if m * 10 ** exp <= target:
                return m * 10 ** exp
        return 10 ** exp

    def get_float(self, var, default):
        try:
            return float(var.get())
        except (tk.TclError, ValueError):
            return default

    def draw_scale_bar(self):
        if self.scale_bar is not None:
            try:
                self.scale_bar.remove()
            except (ValueError, NotImplementedError):
                pass
            self.scale_bar = None
        if self.raw_image is None or not self.sb_show.get():
            return

        # координаты осей уже в мкм (extent учитывает pixel size)
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        length = self.get_float(self.sb_length, 0)
        if length <= 0:
            length = self.nice_length(abs(x1 - x0) * 0.2)

        # толщина задаётся в экранных пикселях -> переводим в мкм по оси Y
        ax_h_px = max(self.ax.get_window_extent().height, 1)
        thickness = abs(y1 - y0) * self.get_float(self.sb_thickness, 4) / ax_h_px

        label = "" if self.sb_hide_text.get() else f"{length:g} µm"
        font = FontProperties(size=self.get_float(self.sb_font_size, 12), weight='bold')
        bar = AnchoredSizeBar(self.ax.transData, length, label,
                              loc=SCALE_BAR_POSITIONS.get(self.sb_position.get(), "lower right"),
                              pad=0.4, borderpad=0.8, sep=4,
                              color=self.sb_color.get().lower(),
                              frameon=self.sb_background.get(),
                              size_vertical=thickness, fontproperties=font)
        if self.sb_background.get():
            bar.patch.set_facecolor("black")
            bar.patch.set_alpha(0.6)
            bar.patch.set_edgecolor("none")
        bar.set_zorder(10)
        self.scale_bar = self.ax.add_artist(bar)

    def on_view_change(self, _ax):
        # при зуме/панорамировании пересчитываем авто-длину и толщину
        self.draw_scale_bar()
        self.canvas.draw_idle()

    def load_tiff(self):
        file_path = filedialog.askopenfilename(filetypes=[("TIFF images", "*.tif *.tiff")])
        if file_path:
            with Image.open(file_path) as img:
                self.raw_image = np.transpose(np.array(img), (1, 0))
            self.first_open = True
            self.reset_all()

    def rotate_points(self, x, y, angle_deg, center):
        """center обязателен и должен быть одним и тем же для прямого
        и обратного поворота, иначе обратное преобразование съезжает."""
        angle = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        cx, cy = center
        x_c, y_c = x - cx, y - cy
        x_rot = x_c * cos_a - y_c * sin_a
        y_rot = x_c * sin_a + y_c * cos_a
        return x_rot + cx, y_rot + cy

    def update_plot(self):
        if self.raw_image is None:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        self.ax.clear()
        self.scale_bar = None
        ps = self.pixel_size.get()
        h, w = self.raw_image.shape
        extent = [-w / 2 * ps, w / 2 * ps, -h / 2 * ps, h / 2 * ps]

        self.ax.imshow(self.raw_image, extent=extent, cmap='gray',
                        vmin=self.vmin.get(), vmax=self.vmax.get(), origin='lower')

        for lx, ly, color, label_txt, tx, ty, a_coeff in self.fitted_plots:
            self.ax.plot(lx, ly, color=color, lw=2)
            # маленькая точка в апексе, чтобы визуально видеть вершину
            self.ax.plot(tx, ty, marker='o', markersize=5,
                         markerfacecolor=color, markeredgecolor='white', zorder=6)
            alignment = 'left' if a_coeff > 0 else 'right'
            offset = 100 if a_coeff > 0 else -100
            self.ax.text(tx + offset, ty, label_txt, color=color, fontsize=10,
                         fontweight='bold', horizontalalignment=alignment,
                         bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

        if self.clicked_points:
            px, py = zip(*self.clicked_points)
            self.ax.scatter(px, py, color='yellow', s=100, edgecolors='black', zorder=5)

        if self.first_open:
            self.ax.set_xlim(extent[0], extent[1])
            self.ax.set_ylim(extent[2], extent[3])
            self.toolbar.update()
            self.first_open = False
        else:
            if cur_xlim != (0.0, 1.0):
                self.ax.set_xlim(cur_xlim)
                self.ax.set_ylim(cur_ylim)

        self.draw_scale_bar()
        # ax.clear() сбрасывает колбэки осей, поэтому подключаем заново
        self.ax.callbacks.connect('xlim_changed', self.on_view_change)
        self.ax.callbacks.connect('ylim_changed', self.on_view_change)
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax or self.raw_image is None:
            return
        if event.key == 'shift':
            if len(self.clicked_points) < 4:
                self.clicked_points.append((event.xdata, event.ydata))
                self.update_plot()

    def fit_parabola(self):
        if len(self.clicked_points) < 4:
            return

        x_raw = np.array([p[0] for p in self.clicked_points])
        y_raw = np.array([p[1] for p in self.clicked_points])

        # Единый центр вращения для всех прямых и обратных поворотов
        center = (np.mean(x_raw), np.mean(y_raw))

        best_angle = 0
        best_r2 = -np.inf

        for angle in np.linspace(-45, 45, 91):
            xr, yr = self.rotate_points(x_raw, y_raw, angle, center)
            p = np.polyfit(xr, yr, 2)
            y_fit = np.polyval(p, xr)

            # R2 считаем в ТОЙ ЖЕ (повёрнутой) системе координат,
            # а не смешивая с исходным y_raw
            residuals = yr - y_fit
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((yr - np.mean(yr)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf

            if r2 > best_r2:
                best_r2 = r2
                best_angle = angle

        # Точный подгон угла вокруг грубого оптимума (шаг сетки был 1°)
        def neg_r2(angle):
            xr, yr = self.rotate_points(x_raw, y_raw, angle, center)
            p = np.polyfit(xr, yr, 2)
            y_fit = np.polyval(p, xr)
            residuals = yr - y_fit
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((yr - np.mean(yr)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
            return -r2

        res = minimize_scalar(neg_r2, bounds=(best_angle - 1, best_angle + 1),
                              method='bounded', options={'xatol': 1e-6})
        best_angle = res.x

        xr, yr = self.rotate_points(x_raw, y_raw, best_angle, center)
        p = np.polyfit(xr, yr, 2)

        x_range = np.linspace(min(xr) - 100, max(xr) + 100, 200)
        y_line = np.polyval(p, x_range)

        # обратный поворот линии вокруг ТОГО ЖЕ центра
        line_x, line_y = self.rotate_points(x_range, y_line, -best_angle, center)

        a, b, c = p
        x0_rot = -b / (2 * a)
        y0_rot = np.polyval(p, x0_rot)

        # обратный поворот вершины вокруг ТОГО ЖЕ центра
        x0, y0 = self.rotate_points(np.array([x0_rot]), np.array([y0_rot]), -best_angle, center)

        R = 1.0 / abs(2 * a)

        self.apexes.append((x0[0], y0[0]))
        self.fits.append(dict(x0=x0[0], y0=y0[0], angle=best_angle, a=a, R=R))

        color = "#ff3333" if a < 0 else "#33ff33"
        info_text = f"R: {R:.2f} um\nAngle: {best_angle:.3f}°"
        self.fitted_plots.append((line_x, line_y, color, info_text, x0[0], y0[0], a))

        pair = self.current_pair()
        if pair is not None:
            self.delta_label.config(text=f"{abs(pair['u']):.3f}")
            self.gap_label.config(text=f"{pair['G']:.3f}")
        elif len(self.apexes) >= 2:
            x0_prev, y0_prev = self.apexes[-2]
            x0_curr, y0_curr = self.apexes[-1]
            dx = abs(x0_curr - x0_prev)
            self.delta_label.config(text=f"{dx:.3f}")
            self.gap_label.config(text="---")

        self.clicked_points = []
        self.update_plot()

    def current_pair(self):
        """Последние два фита с противоположным направлением ветвей.

        Смещение u и зазор G считаются в системе координат оси (повёрнутой
        на средний угол фитов), а не в осях картинки: иначе при наклоне оси
        на угол α зазор подмешивается в Δx как G·sinα.
        u = x(ветви вверх) − x(ветви вниз), G = y(ветви вверх) − y(ветви вниз).
        """
        if len(self.fits) < 2:
            return None
        f1, f2 = self.fits[-2:]
        if np.sign(f1['a']) == np.sign(f2['a']):
            return None
        up, down = (f1, f2) if f1['a'] > 0 else (f2, f1)
        angle = (f1['angle'] + f2['angle']) / 2
        xu, yu = self.rotate_points(up['x0'], up['y0'], angle, (0.0, 0.0))
        xd, yd = self.rotate_points(down['x0'], down['y0'], angle, (0.0, 0.0))
        return dict(u=xu - xd, G=yu - yd, R_up=up['R'], R_down=down['R'], angle=angle)

    def open_3d_window(self):
        if self.win_3d is not None and self.win_3d.winfo_exists():
            self.win_3d.lift()
            return
        win = self.win_3d = tk.Toplevel(self.root)
        win.title("3D Reconstruction")
        win.configure(bg="#34495e", padx=15, pady=15)
        lbl = dict(fg="white", bg="#34495e")

        row = tk.Frame(win, bg="#34495e")
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, text="Tilt θ (beam vs. disk plane), °:", **lbl).pack(side=tk.LEFT)
        entry = tk.Entry(row, textvariable=self.theta, width=8, justify='center')
        entry.pack(side=tk.RIGHT)
        entry.bind('<Return>', lambda e: self.update_3d())
        entry.bind('<FocusOut>', lambda e: self.update_3d())

        row = tk.Frame(win, bg="#34495e")
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, text="Tilt sign s:", **lbl).pack(side=tk.LEFT)
        tk.OptionMenu(row, self.sign_mode, "Auto", "+1", "-1",
                      command=lambda _: self.update_3d()).pack(side=tk.RIGHT)

        row = tk.Frame(win, bg="#34495e")
        row.pack(fill=tk.X, pady=8)
        tk.Button(row, text="Store pair as 0°", bg="#5dade2", fg="white",
                  command=lambda: self.store_projection(0)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(row, text="Store pair as 90°", bg="#5dade2", fg="white",
                  command=lambda: self.store_projection(90)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(row, text="Clear", bg="#e74c3c", fg="white",
                  command=self.clear_projections).pack(side=tk.LEFT, padx=2)

        tk.Label(win, text="Fit both parabolas on a projection (Load TIFF -> fit up & down),\n"
                           "then store it. Load the 90° image and repeat.",
                 fg="#f1c40f", bg="#34495e", font=("Arial", 9, "italic"), justify=tk.LEFT).pack(anchor="w")

        self.result_text = tk.Text(win, width=72, height=22, font=("Consolas", 10),
                                   bg="#2c3e50", fg="#ecf0f1", relief=tk.FLAT)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.update_3d()

    def store_projection(self, angle):
        pair = self.current_pair()
        if pair is None:
            self.show_3d_text("Нужны два последних фита с ветвями в разные стороны\n"
                              "(верхний и нижний параболоид).")
            return
        self.projections[angle] = pair
        self.update_3d()

    def clear_projections(self):
        self.projections = {0: None, 90: None}
        self.update_3d()

    def show_3d_text(self, text):
        if self.win_3d is None or not self.win_3d.winfo_exists():
            return
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

    def update_3d(self):
        lines = []
        for ang in (0, 90):
            p = self.projections[ang]
            if p is None:
                lines.append(f"{ang:>3}°: not stored")
            else:
                lines.append(f"{ang:>3}°: u = {p['u']:+.3f}  G = {p['G']:.3f}  "
                             f"R'up = {p['R_up']:.2f}  R'down = {p['R_down']:.2f}")
        lines.append("")

        p0, p90 = self.projections[0], self.projections[90]
        if p0 is None or p90 is None:
            self.show_3d_text("\n".join(lines))
            return

        theta = self.get_float(self.theta, TILT_ANGLE)
        sign = {"+1": 1, "-1": -1}.get(self.sign_mode.get())
        r = reconstruct_3d(p0, p90, theta, sign)
        lines += [
            f"θ = {theta:g}°,  s = {r['sign']:+d}" + ("  (auto)" if sign is None else ""),
            "",
            "Lateral axis offset (um)",
            f"  Δx (0°)            = {r['dx']:+.3f}",
            f"  Δy (90°)           = {r['dy']:+.3f}",
            f"  e = √(Δx²+Δy²)     = {r['e']:.3f}",
            "",
            "Along the axis (um)",
            f"  Δz apex gap        = {r['dz']:.3f}",
            f"    from 0° / 90°    = {r['dz0']:.3f} / {r['dz90']:.3f}",
            f"    mismatch         = {r['dz0'] - r['dz90']:+.3f}",
            f"  d_min web          = {r['d_min']:.3f}",
            f"  D apex-apex (3D)   = {r['D']:.3f}",
            "",
            "Radii, corrected R = R'·cosθ (um)",
            f"  R up    = {r['R_up']:.3f}   (0°: {r['R_up_0']:.3f}, 90°: {r['R_up_90']:.3f})",
            f"  R down  = {r['R_down']:.3f}   (0°: {r['R_down_0']:.3f}, 90°: {r['R_down_90']:.3f})",
        ]
        self.show_3d_text("\n".join(lines))

    def clear_current_selection(self):
        self.clicked_points = []
        self.update_plot()

    def reset_all(self):
        self.clicked_points = []
        self.apexes = []
        self.fitted_plots = []
        self.fits = []
        self.delta_label.config(text="---")
        self.gap_label.config(text="---")
        self.first_open = True
        self.update_plot()


if __name__ == "__main__":
    root = tk.Tk()
    app = ParabolaAnalyzer(root)
    root.mainloop()