"""
gui.py
──────
Tkinter GUI for the trained ExoplanetCNN. Reuses inference.py's model class,
predict_tic() pipeline, TIC stellar lookup, and physical-parameter formulas —
same code paths as the CLI.

Run:
  python gui.py
"""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

matplotlib.use("TkAgg")

from inference import (
    DEVICE, ExoplanetCNN, fetch_stellar_params, planet_physical_params, predict_tic,
)

MODEL_PATH = "exoplanet_cnn.pt"
CSV_PATH = "labeled_tess_dataset.csv"
CACHE_DIR = "lc_cache"


def _star_color(teff_k: float) -> str:
    if teff_k >= 10000:
        return "#9bb0ff"
    if teff_k >= 7500:
        return "#bbccff"
    if teff_k >= 6000:
        return "#f5f3ff"
    if teff_k >= 5200:
        return "#fff2a1"
    if teff_k >= 3700:
        return "#ffcc6f"
    return "#ff8c5a"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Exoplanet Finder")
        self.geometry("1020x620")

        self.model = None

        left = ttk.Frame(self, padding=12)
        left.pack(side="left", fill="y")

        form = ttk.Frame(left)
        form.pack(fill="x")
        ttk.Label(form, text="TIC ID").grid(row=0, column=0, sticky="w")
        self.tic_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.tic_var, width=16).grid(row=0, column=1, sticky="ew", padx=6)

        self.force_var = tk.BooleanVar()
        ttk.Checkbutton(form, text="Force fresh download",
                        variable=self.force_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.run_btn = ttk.Button(left, text="Classify", command=self.on_classify)
        self.run_btn.pack(pady=8, fill="x")

        self.status_var = tk.StringVar(value=f"Loading model on {DEVICE}...")
        ttk.Label(left, textvariable=self.status_var, foreground="gray", wraplength=220).pack(fill="x")

        ttk.Label(left, text="BLS transit fit", font=("", 9, "bold")).pack(anchor="w", pady=(14, 0))
        self.result = tk.Text(left, height=12, width=30, state="disabled")
        self.result.pack(pady=4, fill="both")

        right = ttk.Frame(self, padding=(0, 12, 12, 12))
        right.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(8.0, 5.6), dpi=100)
        gs = self.fig.add_gridspec(2, 3, width_ratios=[3, 3, 1.3], height_ratios=[1, 1])
        self.ax_main = self.fig.add_subplot(gs[:, :2])
        self.ax_global = self.fig.add_subplot(gs[0, 2])
        self.ax_local = self.fig.add_subplot(gs[1, 2])
        self._reset_plot()

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.after(100, self.load_model)

    # ── Model loading ───────────────────────────────────────────────────────

    def load_model(self):
        try:
            model = ExoplanetCNN().to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            self.model = model
            self.status_var.set(f"Model loaded ({DEVICE}). Ready.")
        except Exception as e:
            self.status_var.set(f"Failed to load model: {e}")

    # ── Classify button ─────────────────────────────────────────────────────

    def on_classify(self):
        tic_id = self.tic_var.get().strip()
        if not tic_id:
            self.status_var.set("Enter a TIC ID first.")
            return
        if self.model is None:
            self.status_var.set("Model not loaded yet.")
            return

        self.run_btn.state(["disabled"])
        self.status_var.set(f"Processing TIC {tic_id}...")
        self._write("")
        self._reset_plot()
        self.canvas.draw()
        threading.Thread(target=self._classify, args=(tic_id,), daemon=True).start()

    def _classify(self, tic_id: str):
        prob, bls_info, vec = predict_tic(
            tic_id, self.model, csv_path=CSV_PATH, cache_dir=CACHE_DIR,
            force_download=self.force_var.get(),
        )
        if prob is None:
            self.after(0, self._show_result, tic_id, None, None, None, None, None)
            return

        self.after(0, self.status_var.set, f"Looking up stellar parameters for TIC {tic_id}...")
        stellar = fetch_stellar_params(tic_id)
        phys = planet_physical_params(bls_info, stellar) if stellar else None

        self.after(0, self._show_result, tic_id, prob, bls_info, vec, phys, stellar)

    # ── Results ──────────────────────────────────────────────────────────────

    def _show_result(self, tic_id, prob, bls_info, vec, phys, stellar):
        self.run_btn.state(["!disabled"])
        if prob is None:
            self.status_var.set(f"Could not process TIC {tic_id}.")
            return

        label = "Planet candidate" if prob >= 0.5 else "False positive"
        self.status_var.set(f"Done — {label} ({prob:.1%})")

        self._write(
            f"Period    : {bls_info['period']:.4f} d\n"
            f"Duration  : {bls_info['duration_hr']:.2f} hr\n"
            f"Depth     : {bls_info['depth_ppm']:.0f} ppm\n"
            f"Sec. ecl. : {bls_info['sec_depth']:.6f}\n"
            f"Even/odd  : {bls_info['even_odd_diff']:.6f}\n"
            f"Sectors   : {bls_info['sectors_used']}\n"
        )

        self._update_plot(tic_id, prob, bls_info, vec, phys, stellar)

    # ── Plotting ─────────────────────────────────────────────────────────────

    def _reset_plot(self):
        for ax, title in [(self.ax_main, "Planet vs Earth"),
                          (self.ax_global, "Folded light curve"),
                          (self.ax_local, "Transit zoom")]:
            ax.clear()
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig.tight_layout()

    def _update_plot(self, tic_id, prob, bls_info, vec: np.ndarray, phys, stellar):
        global_view = vec[:201]
        local_view = vec[201:262]

        self.ax_global.clear()
        self.ax_global.plot(np.linspace(-0.5, 0.5, len(global_view)), global_view, lw=0.7)
        self.ax_global.set_title("Folded light curve", fontsize=9)
        self.ax_global.tick_params(labelsize=6)

        self.ax_local.clear()
        self.ax_local.plot(np.linspace(-1, 1, len(local_view)), local_view, lw=0.7, color="tab:orange")
        self.ax_local.set_title("Transit zoom", fontsize=9)
        self.ax_local.tick_params(labelsize=6)

        ax = self.ax_main
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

        label = "Planet candidate" if prob >= 0.5 else "False positive"
        title = f"TIC {tic_id}"
        if phys:
            title += f"  —  {phys['type_label']}"
        ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=13, weight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.90, f"CNN result: {label}  ({prob:.1%} planet probability)",
                ha="center", va="top", fontsize=9, color="dimgray", transform=ax.transAxes)

        if not phys:
            ax.text(0.5, 0.5, "Stellar parameters unavailable\n(no radius/mass/temperature estimate)",
                    ha="center", va="center", fontsize=10, transform=ax.transAxes)
            self.fig.tight_layout()
            self.canvas.draw()
            return

        # Host star backdrop (visually compressed — not to scale)
        star_color = _star_color(stellar["teff_k"])
        ax.add_patch(Circle((0.28, 0.62), 0.22, color=star_color, alpha=0.5, zorder=1))
        ax.text(0.28, 0.35, "host star (not to scale)", ha="center", fontsize=7, color="dimgray")
        ax.text(0.28, 0.62,
                f"Teff {stellar['teff_k']:.0f} K\n{stellar['rad_rsun']:.2f} R☉\n{stellar['mass_msun']:.2f} M☉",
                ha="center", va="center", fontsize=8)

        # Earth vs planet, to scale relative to each other
        earth_r, planet_r = 1.0, phys["rp_earth"]
        max_r = max(earth_r, planet_r)
        cmap = matplotlib.colormaps["coolwarm"]
        teq_norm = np.clip((phys["teq_k"] - 150) / (1500 - 150), 0, 1)
        planet_color = cmap(teq_norm)

        cx_earth, cx_planet, cy = 0.62, 0.85, 0.58
        ax.add_patch(Circle((cx_earth, cy), 0.10 * earth_r / max_r, color="tab:blue", zorder=2))
        ax.add_patch(Circle((cx_planet, cy), 0.10 * planet_r / max_r, color=planet_color, zorder=2))
        ax.text(cx_earth, cy - 0.16, "Earth", ha="center", fontsize=8)
        ax.text(cx_planet, cy - 0.16, f"{planet_r:.2f} R⊕", ha="center", fontsize=8)

        mass_str = f"{phys['mass_earth']:.2f} M⊕" if phys["mass_earth"] is not None else "unconstrained"
        density_str = f"{phys['density_gcm3']:.2f} g/cm³" if phys["density_gcm3"] is not None else "n/a"

        info = (
            f"Radius        : {phys['rp_earth']:.2f} R⊕  ({phys['rp_jupiter']:.3f} R♃)\n"
            f"Mass (est.)   : {mass_str}\n"
            f"Density (est.): {density_str}\n"
            f"Semi-major axis: {phys['a_au']:.4f} AU\n"
            f"Orbital period : {bls_info['period']:.3f} days\n"
            f"Eq. temperature: {phys['teq_k']:.0f} K\n"
            f"Insolation     : {phys['insolation_earth']:.2f}× Earth"
        )
        ax.text(0.02, 0.20, info, ha="left", va="top", fontsize=9, family="monospace",
                transform=ax.transAxes)

        self.fig.tight_layout()
        self.canvas.draw()

    def _write(self, text: str):
        self.result.configure(state="normal")
        self.result.delete("1.0", "end")
        self.result.insert("1.0", text)
        self.result.configure(state="disabled")


if __name__ == "__main__":
    if not Path(MODEL_PATH).exists():
        raise SystemExit(f"Model not found at '{MODEL_PATH}'. Run train_classifier.py first.")
    App().mainloop()
