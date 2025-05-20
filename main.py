"""Моделирование точек Лагранжа и их устойчивости

Запуск GUI: python main.py
Запуск CLI: python main.py -cli

Параметры:
  - m1(кг) - масса первого тела
  - m2(кг) - масса второго тела
  - R(м) - расстояние между ними
  - Ω(c^-1) - угловая скорость вращения системы
  - t_max(с) - продолжительность моделирования по времени
  - Смещение стартового положения отн. т.Лагранжа - δx
  - Смещение стартового положения отн. т.Лагранжа - δy
  - Точки(номера) - какие точки рассчитывать(цифры 1-5 через ',')
  - Интервал кадра(мс) - интервал между кадрами

Дефолтные значения:
defaults={'m1':'1','m2':'0.02','R':'1','Omega':'1',
              't_max':'100','dx0':'0.001', 'dy0': 0.001, 'interval':'20','pts':'1,4'}

Графики:
  - Траектория за время t_max
  - Фазовый портрет (x,vx)
  - Расстояние от тела до точки Лагранжа r(t)
  - Анимация движения тел вокруг точек Лагранжа
"""

import sys, math
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

pal = {
    "L1": "tab:red",
    "L2": "tab:orange",
    "L3": "tab:pink",
    "L4": "tab:green",
    "L5": "tab:blue",
}

# Process V, A projections
def rhs(t, s, mu):
    x, y, vx, vy = s
    r1x, r2x = x + mu, x - 1 + mu
    r1 = math.hypot(r1x, y)
    r2 = math.hypot(r2x, y)
    dUdx = x - (1 - mu) * r1x / r1**3 - mu * r2x / r2**3
    dUdy = y - (1 - mu) * y / r1**3 - mu * y / r2**3
    ax = 2 * vy + dUdx
    ay = -2 * vx + dUdy
    return np.array([vx, vy, ax, ay])

# dU/dx in two objects case
def dUdx(x, mu):
    r1 = abs(x + mu)
    r2 = abs(x - 1 + mu)
    return x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3

# Support for L_points
def root(f, a, b, mu):
    try:
        return brentq(lambda z: f(z, mu), a, b, maxiter=600)
    except ValueError:
        return None

# Process positions of Lagrange Points
# For L1,L2,L3 - counting using dUdx - numerical method
# For L4,L5 - formula
def L_points(mu):
    L1 = root(dUdx, -mu + 1e-6, 1 - mu - 1e-6, mu)
    L2 = root(dUdx, 1 - mu + 1e-6, 2.0, mu)
    L3 = root(dUdx, -2.0, -mu - 1e-6, mu)
    L4 = (0.5 - mu, math.sqrt(3) / 2)
    L5 = (0.5 - mu, -math.sqrt(3) / 2)
    return {"L1": (L1, 0), "L2": (L2, 0), "L3": (L3, 0), "L4": L4, "L5": L5}


def U_eff(x, y, mu):
    r1 = np.hypot(x + mu, y)
    r2 = np.hypot(x - 1 + mu, y)
    return -(1 - mu)/r1 - mu/r2 - 0.5*(x**2 + y**2)

# Integrating equation of motion using DOP853
# Getting arrays of t,x,y,vx,rL on the interval [0, t_max] with dx0,dy0 init shift 
def solve_point(p, mu, R, Om, t_max, dx0, dy0, steps=2000):
    x0, y0 = L_points(mu)[p]
    if x0 is None:
        raise ValueError("нет корня")
    st = np.array([x0 + dx0, y0 + dy0, 0, 0])
    sol = solve_ivp(
        partial(rhs, mu=mu),
        (0, t_max * Om),
        st,
        t_eval=np.linspace(0, t_max * Om, steps),
        rtol=1e-10,
        atol=1e-11,
        method="DOP853",
    )
    x = R * sol.y[0]
    y = R * sol.y[1]
    vx = R * Om * sol.y[2]
    t = sol.t / Om
    c = -2*U_eff(sol.y[0], sol.y[1], mu) - (sol.y[2]**2 + sol.y[3]**2)
    return {"t": t, "x": x, "y": y, "vx": vx, "rL": np.array([R * x0, R * y0]), "c": c}


# Рендеринг графиков
def make_plots(sel, mu, R, Om, t_max, dx0, dy0):
    k = len(sel)
    figT, axT = plt.subplots(1, k, figsize=(4 * k, 4))
    figP, axP = plt.subplots(1, k, figsize=(4 * k, 4))
    figD, axD = plt.subplots(1, k, figsize=(4 * k, 4))
    figC, axC = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axT = [axT]
        axP = [axP]
        axD = [axD]
        axC = [axC]
    data = {}
    for aT, aP, aD, aC, pt in zip(axT, axP, axD, axC, sel):
        d = solve_point(pt, mu, R, Om, t_max, dx0, dy0)
        data[pt] = d
        
        aT.plot(d["x"], d["y"], color=pal[pt])
        aT.scatter(*d["rL"], color=pal[pt], marker="*")
        aT.set(title=f"Траектория {pt} ", xlabel="x (м)", ylabel="y (м)")
        aT.set_aspect("equal")
        aT.grid()
        
        aP.plot(d["x"], d["vx"], color=pal[pt])
        aP.set(title=f"Фазовый портрет {pt} ", xlabel="x (м)", ylabel="vx (м/с)")
        aP.grid()
        
        dist = np.hypot(d["x"] - d["rL"][0], d["y"] - d["rL"][1])
        aD.semilogy(d["t"], dist, color=pal[pt])
        aD.set(title=f"Расстояние тела до {pt} |δr|(t)", xlabel="t (с)", ylabel="|δr| (м)")
        aD.grid()

        aC.plot(d["t"], d["c"], color=pal[pt])
        aC.set(title=f"Интеграл Якоби {pt} ", xlabel="t (с)", ylabel="C")
        aC.grid()

    for f in (figT, figP, figD, figC):
        f.tight_layout()
    return data


# Animation of Lagrange Points
def animate(sel, data, R, mu, interval=40, trail=300):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set(aspect="equal", xlim=(-2 * R, 2 * R), ylim=(-2 * R, 2 * R), title="Анимация")
    ax.add_patch(Circle((-mu * R, 0), 0.03 * R, color="black"))
    ax.add_patch(
        Circle(((1 - mu) * R, 0), 0.02 * R, edgecolor="black", facecolor="none")
    )
    handles = {}
    for p in sel:
        (ln,) = ax.plot([], [], marker="o", color=pal[p], label=p)
        (tr,) = ax.plot([], [], color=pal[p], lw=0.5)
        handles[p] = (ln, tr)
    ax.legend()
    maxf = max(len(data[p]["t"]) for p in sel)

    def init():
        for ln, tr in handles.values():
            ln.set_data([], [])
            tr.set_data([], [])
        return sum(handles.values(), ())

    def upd(f):
        for p in sel:
            d = data[p]
            if f < len(d["x"]):
                ln, tr = handles[p]
                ln.set_data(d["x"][f], d["y"][f])
                st = max(0, f - trail)
                tr.set_data(d["x"][st:f], d["y"][st:f])
        return sum(handles.values(), ())

    ani = FuncAnimation(
        fig, upd, frames=maxf, init_func=init, interval=interval, blit=True
    )
    plt.show()
    return ani  # сохранить ссылку


# Текстовое поле
def summary_text(mu, sel, R, Om, t_max, dx0, dy0):
    txt = [
        f"μ = {mu:.5f}  (устойчивы L4,L5 если μ<0.03852)",
        f"R = {R} м,  Ω = {Om} с⁻¹,  t_max = {t_max} с,  Начальное смещение отн. т.Лагранжа (δx м,δy м) = ({dx0},{dy0})",
        "Выбранные точки: " + ", ".join(sel),
    ]
    for p, (x, y) in L_points(mu).items():
        if x is not None:
            txt.append(f"  {p}: ({x*R:.3f}, {y*R:.3f}) м")
    return "\n".join(txt)


def run(params):
    mu = params["m2"] / (params["m1"] + params["m2"])
    data = make_plots(
        params["pts"], mu, params["R"], params["Omega"], params["t_max"], params["dx0"], params["dy0"]
    )
    _ = animate(params["pts"], data, params["R"], mu, params["interval"])


def gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("Точки Лагранжа")
    defaults = {
        "m1": "1",
        "m2": "0.02",
        "R": "1",
        "Omega": "1",
        "t_max": "100",
        "dx0": "0.001",
        "dy0": "0.001",
        "interval": "20",
        "pts": "1,4",
    }
    entries = {}
    labels = {
        "m1": "m₁",
        "m2": "m₂",
        "R": "R (м)",
        "Omega": "Ω (с⁻¹)",
        "t_max": "Продолжительность (с)",
        "dx0": "Начальное смещение δx",
        "dy0": "Начальное смещение δy",
        "interval": "Интервал между кадрами (мс)",
        "pts": "Точки(1-5)",
    }
    for i, (k, v) in enumerate(defaults.items()):
        ttk.Label(root, text=labels[k]).grid(row=i, column=0)
        e = tk.Entry(root)
        e.insert(0, v)
        e.grid(row=i, column=1)
        entries[k] = e
    summary = tk.Text(root, width=65, height=15)
    summary.grid(row=len(defaults), columnspan=2)

    def start():
        try:
            m1 = float(entries["m1"].get())
            m2 = float(entries["m2"].get())
            R = float(entries["R"].get())
            Om = float(entries["Omega"].get())
            tmax = float(entries["t_max"].get())
            dx0 = float(entries["dx0"].get())
            dy0 = float(entries["dy0"].get())
            inter = int(float(entries["interval"].get()))
            mapping = {"1": "L1", "2": "L2", "3": "L3", "4": "L4", "5": "L5"}
            pts = [
                mapping[p.strip()]
                for p in entries["pts"].get().split(",")
                if p.strip() in mapping
            ]
            if not pts:
                raise ValueError
        except Exception:
            messagebox.showerror("Ошибка", "Неверный ввод")
            return
        summary.delete("1.0", tk.END)
        mu = m2 / (m1 + m2)
        summary.insert(tk.END, summary_text(mu, pts, R, Om, tmax, dx0, dy0))
        run(
            {
                "m1": m1,
                "m2": m2,
                "R": R,
                "Omega": Om,
                "t_max": tmax,
                "dx0": dx0,
                "dy0": dy0,
                "pts": pts,
                "interval": inter,
            }
        )

    ttk.Button(root, text="Старт", command=start).grid(
        row=len(defaults) + 1, columnspan=2
    )
    root.mainloop()


def cli():
    mp = {"1": "L1", "2": "L2", "3": "L3", "4": "L4", "5": "L5"}
    m1 = float(input("m1: "))
    m2 = float(input("m2: "))
    R = float(input("R (м): "))
    Om = float(input("Ω (с^-1): "))
    t = float(input("t_max (с): "))
    dx0 = float(input("δx: "))
    dy0 = float(input("δy: "))
    pts = [mp[p.strip()] for p in input("точки 1‑5: ").split(",") if p.strip() in mp]
    inter = int(float(input("интервал кадра (мс): ")))
    run(
        {
            "m1": m1,
            "m2": m2,
            "R": R,
            "Omega": Om,
            "t_max": t,
            "dx0": dx0,
            "dy0": dy0,
            "pts": pts,
            "interval": inter,
        }
    )


if __name__ == "__main__":
    if "--cli" in sys.argv:
        cli()
    else:
        try:
            gui()
        except Exception as e:
            print("GUI не запустился, CLI", e)
            cli()
