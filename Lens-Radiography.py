import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image

class ParabolaAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("X-Ray Lens Parabola Analyzer (Pro)")
        self.root.geometry("1300x900")
        self.root.configure(bg="#2c3e50")

        # Данные
        self.raw_image = None
        self.pixel_size = tk.DoubleVar(value=2.02)
        self.vmin = tk.DoubleVar(value=4210)
        self.vmax = tk.DoubleVar(value=6683)
        
        # Интерактив
        self.clicked_points = []  
        self.apexes = []          
        self.fitted_plots = []    
        
        self.first_open = True  # Флаг для отслеживания первого открытия
        self.setup_ui()

    def setup_ui(self):
        self.main_frame = tk.Frame(self.root, bg="#2c3e50")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Левая часть: График и тулбар
        self.left_panel = tk.Frame(self.main_frame, bg="black")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Добавляем стандартную панель навигации (Zoom, Pan, Home)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.left_panel)
        self.toolbar.update()

        # Привязка событий
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.root.bind('<Return>', lambda e: self.fit_parabola() if (e.state & 0x1) else None) # Shift + Enter

        # Правая часть: Панель управленияs
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
                self.raw_image = np.rot90(np.array(img))
            self.first_open = True
            self.reset_all()

    def update_plot(self):
        if self.raw_image is None: return
        
        # Сохраняем текущие границы осей (zoom)
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        self.ax.clear()
        ps = self.pixel_size.get()
        h, w = self.raw_image.shape
        extent = [-w/2*ps, w/2*ps, -h/2*ps, h/2*ps]
        
        self.ax.imshow(self.raw_image, extent=extent, cmap='gray', 
                       vmin=self.vmin.get(), vmax=self.vmax.get(), origin='lower')
        
        for lx, ly, color, label_txt, tx, ty, a_coeff in self.fitted_plots:
            self.ax.plot(lx, ly, color=color, lw=2)
            alignment = 'left' if a_coeff > 0 else 'right'
            offset = 100 if a_coeff > 0 else -100
            self.ax.text(tx + offset, ty, label_txt, color=color, fontsize=10, 
                         fontweight='bold', horizontalalignment=alignment,
                         bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

        if self.clicked_points:
            px, py = zip(*self.clicked_points)
            self.ax.scatter(px, py, color='yellow', s=100, edgecolors='black', zorder=5)

        if self.first_open:
            # Если файл только что загружен — показываем всё изображение целиком
            self.ax.set_xlim(extent[0], extent[1])
            self.ax.set_ylim(extent[2], extent[3])
            self.first_open = False # Сбрасываем флаг
        else:
            # Если мы просто обновляем контраст или ставим точки — сохраняем текущий Zoom пользователя
            if cur_xlim != (0.0, 1.0): 
                self.ax.set_xlim(cur_xlim)
                self.ax.set_ylim(cur_ylim)

        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax or self.raw_image is None: return
        
        # В matplotlib event.key содержит имя нажатой клавиши во время клика
        if event.key == 'shift':
            if len(self.clicked_points) < 4:
                self.clicked_points.append((event.xdata, event.ydata))
                self.update_plot()

    def fit_parabola(self):
        if len(self.clicked_points) < 4:
            return

        x_pts, y_pts = zip(*self.clicked_points)
        p = np.polyfit(x_pts, y_pts, 2)
        a, b, c = p
        
        x0 = -b / (2 * a)
        y0 = np.polyval(p, x0)
        R = 1.0 / abs(2 * a)
        angle = np.degrees(np.arctan(b)) 

        self.apexes.append(x0)

        margin = (max(x_pts) - min(x_pts)) * 0.5
        line_x = np.linspace(min(x_pts) - margin, max(x_pts) + margin, 100)
        line_y = np.polyval(p, line_x)
        
        color = "#ff3333" if a < 0 else "#33ff33"
        info_text = (f"R: {R:.2f} um\n"
                     f"Apex X: {x0:.2f} um\n"
                     f"Angle: {angle:.2f}°")
        
        self.fitted_plots.append((line_x, line_y, color, info_text, x0, y0, a))

        if len(self.apexes) >= 2:
            dx = abs(self.apexes[-1] - self.apexes[-2])
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
        self.update_plot()
        # Сброс Zoom
        if self.raw_image is not None:
            ps = self.pixel_size.get()
            h, w = self.raw_image.shape
            self.ax.set_xlim(-w/2, w/2)
            self.ax.set_ylim(-h/2, h/2)
            self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ParabolaAnalyzer(root)
    root.mainloop()