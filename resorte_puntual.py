#!/usr/bin/env python3
"""
Resorte con masa puntual y anclaje fijo: limite de aro puntual (R -> 0).

En el limite R -> 0, el anclaje del resorte queda fijo en el origen: A=(0,0).
La coordenada theta desaparece y q=(l, phi), donde phi se mide desde la vertical
descendente. La masa se encuentra en P=(l sin(phi), -l cos(phi)).

Uso:
    python resorte_aro.py            # ventana interactiva
    python resorte_aro.py --test     # comprobacion numerica + PNG sin ventana
"""
import os
import sys
import numpy as np
from scipy.integrate import solve_ivp

TMAX = 20.0
FPS = 50
L_MIN = 1e-3


# ----------------------------------------------------------------------------- fisica
def mass_matrix(q, p):
    l, ph = q
    m = p["m"]
    return np.array([[m, 0.0], [0.0, m * l**2]])


def rhs(t, y, p):
    l, ph, dl, dph = y
    m, k, l0, g = p["m"], p["k"], p["l0"], p["g"]
    ddl = l * dph**2 + g * np.cos(ph) - (k / m) * (l - l0)
    ddph = -2 * dl * dph / l - g * np.sin(ph) / l
    return np.array([dl, dph, ddl, ddph])


def energy(y, p):
    l, ph, dl, dph = y
    T = 0.5 * p["m"] * (dl**2 + l**2 * dph**2)
    V = -p["m"] * p["g"] * l * np.cos(ph) + 0.5 * p["k"] * (l - p["l0"]) ** 2
    return T + V


def simulate(p, y0, tmax=TMAX, fps=FPS):
    def colapso(t, y, p):          # el resorte no puede tener longitud <= 0
        return y[0] - L_MIN
    colapso.terminal = True
    colapso.direction = -1

    y0 = list(y0)
    y0[0] = max(y0[0], 2 * L_MIN)  # evita arrancar ya sobre el evento / l=0

    t_eval = np.arange(0.0, tmax, 1.0 / fps)
    sol = solve_ivp(rhs, (0, tmax), y0, method="DOP853", t_eval=t_eval,
                    rtol=1e-7, atol=1e-9, args=(p,), events=colapso)
    t, Y = sol.t, sol.y
    if len(t) == 0:                # integracion terminada antes del primer paso
        t, Y = np.array([0.0]), np.array(y0, dtype=float).reshape(4, 1)
    E = np.array([energy(Y[:, i], p) for i in range(Y.shape[1])])
    return t, Y, E


def positions(Y, p):
    l, ph = Y[0], Y[1]
    A = np.zeros((2,) + np.shape(l))
    P = np.array([l * np.sin(ph), -l * np.cos(ph)])
    return A, P


def spring_xy(A, P, w, n=10):
    d = P - A
    L = np.hypot(*d)
    if L < 1e-9:
        return np.array([A[0], P[0]]), np.array([A[1], P[1]])
    u = d / L
    nv = np.array([-u[1], u[0]])
    ts = np.linspace(0.12, 0.88, 2 * n)
    off = w * (-1.0) ** np.arange(2 * n)
    pts = [A] + [A + u * L * ti + nv * oi for ti, oi in zip(ts, off)] + [P]
    pts = np.array(pts)
    return pts[:, 0], pts[:, 1]


# ----------------------------------------------------------------------------- GUI
def build_gui():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(14, 9.6))
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is not None and hasattr(mgr, "set_window_title"):
        mgr.set_window_title("Resorte con anclaje fijo - Lagrange")

    ax = fig.add_axes([0.04, 0.40, 0.46, 0.57])
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax_ang = fig.add_axes([0.58, 0.80, 0.39, 0.16])
    ax_l = fig.add_axes([0.58, 0.61, 0.39, 0.14], sharex=ax_ang)
    ax_E = fig.add_axes([0.58, 0.42, 0.39, 0.14], sharex=ax_ang)
    for a in (ax_ang, ax_l):
        a.tick_params(labelbottom=False)

    # (nombre, etiqueta, min, max, valor, paso)
    fis = [("m", "m (kg)", 0.1, 5.0, 1.0, 0.05),
           ("k", "k (N/m)", 1.0, 200.0, 30.0, 1.0),
           ("l0", "l0 (m)", 0.1, 2.5, 1.0, 0.05),
           ("g", "g (m/s2)", 0.0, 20.0, 9.81, 0.01)]
    ini = [("l_i", "l(0) (m)", 0.1, 3.5, 1.5, 0.05),
           ("ph0", "phi0 (deg)", -180, 180, 20.0, 1),
           ("dl0", "l' (m/s)", -5, 5, 0.0, 0.1),
           ("dph0", "phi' (rad/s)", -10, 10, 0.0, 0.1)]
    sliders = {}
    fig.text(0.10, 0.325, "Parametros fisicos", fontweight="bold")
    fig.text(0.68, 0.325, "Condiciones iniciales", fontweight="bold")
    for col, lista in enumerate((fis, ini)):
        x0 = 0.10 if col == 0 else 0.68
        for r, (nm, lab, lo, hi, v, step) in enumerate(lista):
            a = fig.add_axes([x0, 0.285 - r * 0.042, 0.28, 0.022])
            sliders[nm] = Slider(a, lab, lo, hi, valinit=v, valstep=step)

    st = {"i": 0, "play": True, "data": None}
    spring, = ax.plot([], [], "-", color="tab:blue", lw=1.4)
    rod, = ax.plot([], [], "o", color="k", ms=4)
    mass, = ax.plot([], [], "o", color="tab:red", ms=13, zorder=5)
    trail, = ax.plot([], [], "-", color="tab:red", lw=0.8, alpha=0.5)
    ax.plot([0], [0], "k+")
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                   family="monospace", fontsize=9)

    lph, = ax_ang.plot([], [], label="phi", color="tab:orange")
    ax_ang.set_ylabel("phi (deg)"); ax_ang.legend(loc="upper right"); ax_ang.grid(alpha=0.25)
    ll, = ax_l.plot([], [], color="tab:green"); ax_l.set_ylabel("l (m)"); ax_l.grid(alpha=0.25)
    lE, = ax_E.plot([], [], color="tab:purple"); ax_E.set_ylabel("E - E0 (J)"); ax_E.grid(alpha=0.25)
    ax_E.set_xlabel("t (s)")
    cur = [a.axvline(0, color="gray", lw=0.8) for a in (ax_ang, ax_l, ax_E)]

    def read():
        v = {k_: s.val for k_, s in sliders.items()}
        p = {k_: v[k_] for k_ in ("m", "k", "l0", "g")}
        y0 = [v["l_i"], np.radians(v["ph0"]), v["dl0"], v["dph0"]]
        return p, y0

    def recompute(_=None):
        p, y0 = read()
        t, Y, E = simulate(p, y0)
        st.update(data=(t, Y, E, p), i=0)
        A, P = positions(Y, p)
        lim = 1.1 * max(np.abs(P).max(), 0.2)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        lph.set_data(t, np.degrees(Y[1]))
        ll.set_data(t, Y[0]); lE.set_data(t, E - E[0])
        for a in (ax_ang, ax_l, ax_E):
            a.relim(); a.autoscale_view()
        ax_ang.set_xlim(0, TMAX)
        fig.canvas.draw_idle()

    def frame(_):
        if st["data"] is None:
            return []
        t, Y, E, p = st["data"]
        if st["play"]:
            st["i"] = (st["i"] + 1) % len(t)
        i = min(st["i"], len(t) - 1)
        A, P = positions(Y[:, i:i + 1], p)
        A, P = A[:, 0], P[:, 0]
        sx, sy = spring_xy(A, P, 0.05)
        spring.set_data(sx, sy); rod.set_data([0], [0]); mass.set_data([P[0]], [P[1]])
        Ak, Pk = positions(Y[:, max(0, i - 120):i + 1], p)
        trail.set_data(Pk[0], Pk[1])
        for c_ in cur:
            c_.set_xdata([t[i], t[i]])
        info.set_text(f"t = {t[i]:6.2f} s\nl     = {Y[0, i]:7.3f} m\n"
                      f"phi   = {np.degrees(Y[1, i]):7.1f} deg\n"
                      f"E     = {E[i]:7.3f} J")
        return []

    # antirrebote: al arrastrar un slider solo se recalcula cuando se deja de mover (~0.25 s)
    timer = fig.canvas.new_timer(interval=250)
    timer.single_shot = True
    timer.add_callback(recompute)

    def pedir_recalculo(_):
        timer.stop()
        timer.start()

    for s in sliders.values():
        s.on_changed(pedir_recalculo)

    b1 = Button(fig.add_axes([0.10, 0.02, 0.12, 0.04]), "Play / Pausa")
    b2 = Button(fig.add_axes([0.25, 0.02, 0.12, 0.04]), "Reiniciar t")
    b1.on_clicked(lambda _: st.update(play=not st["play"]))
    b2.on_clicked(lambda _: st.update(i=0))

    recompute()
    anim = FuncAnimation(fig, frame, interval=1000 / FPS, blit=False, cache_frame_data=False)
    # evita que el recolector de basura elimine estos objetos
    fig._anim, fig._widgets, fig._timer = anim, (sliders, b1, b2), timer
    return fig, frame, st


# ----------------------------------------------------------------------------- test
def self_test():
    import matplotlib
    matplotlib.use("Agg")
    p = dict(m=1.0, k=30.0, l0=1.0, g=9.81)
    y0 = [1.5, np.radians(20), 0, 0]
    t, Y, E = simulate(p, y0)
    print(f"pasos: {len(t)}  t_final={t[-1]:.2f}  deriva de energia max = "
          f"{np.abs(E - E[0]).max():.2e} J (E0={E[0]:.3f})")
    print("l min/max:", Y[0].min().round(3), Y[0].max().round(3))
    fig, frame, st = build_gui()
    st["i"] = 400
    frame(0)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prueba.png")
    fig.savefig(out, dpi=80)
    print("PNG guardado en", out)


if __name__ == "__main__":
    if "--test" in sys.argv:
        self_test()
    else:
        import matplotlib.pyplot as plt
        build_gui()
        plt.show()
