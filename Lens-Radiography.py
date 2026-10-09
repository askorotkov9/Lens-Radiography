import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import filedialog
from PIL import Image

PIXEL_SIZE = 1.95  # 2.02
gray_level_min = 4210
gray_level_max = 6683


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

        self.side_panel = tk.Frame(self.main_frame, width=320, bg="#34495e", padx=20, pady=20)
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(self.side_panel, text="ANALYSIS", fg="white", bg="#34495e", font=("Arial", 14, "bold")).pack(pady=(0, 20))

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

        tk.Frame(self.side_panel, height=2, bg="#5dade2").pack(fill=tk.X, pady=20)
        tk.Label(self.side_panel, text="Calculated Delta X (um):", fg="white", bg="#34495e", font=("Arial", 11)).pack()
        self.delta_label = tk.Label(self.side_panel, text="---", fg="#40e0d0", bg="#34495e", font=("Arial", 36, "bold"))
        self.delta_label.pack()

    def create_input(self, label_text, var):
        frame = tk.Frame(self.side_panel, bg="#34495e")
        frame.pack(fill=tk.X, pady=5)
        tk.Label(frame, text=label_text, fg="white", bg="#34495e", width=15, anchor="w").pack(side=tk.LEFT)
        tk.Entry(frame, textvariable=var, width=10, justify='center').pack(side=tk.RIGHT)

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

        color = "#ff3333" if a < 0 else "#33ff33"
        info_text = f"R: {R:.2f} um\nAngle: {best_angle:.3f}°"
        self.fitted_plots.append((line_x, line_y, color, info_text, x0[0], y0[0], a))

        if len(self.apexes) >= 2:
            x0_prev, y0_prev = self.apexes[-2]
            x0_curr, y0_curr = self.apexes[-1]
            dx = abs(x0_curr - x0_prev)
            self.delta_label.config(text=f"{dx:.3f}")

        self.clicked_points = []
        self.update_plot()

    def clear_current_selection(self):
        self.clicked_points = []
        self.update_plot()

    def reset_all(self):
        self.clicked_points = []
        self.apexes = []
        self.fitted_plots = []
        self.delta_label.config(text="---")
        self.first_open = True
        self.update_plot()


if __name__ == "__main__":
    root = tk.Tk()
    app = ParabolaAnalyzer(root)
    root.mainloop()