#!/usr/bin/env python3
""" 
Resorte con lastre sobre un aro vertical - simulacion lagrangiana interactiva.

Coordenadas generalizadas q = (theta, l, phi)
    theta : posicion angular del extremo del resorte sobre el aro (medida desde +y hacia +x)
    l     : longitud del resorte
    phi   : inclinacion del resorte respecto a la vertical descendente

    A = (R sin th,  R cos th)                 (anclaje en el aro)
    P = A + l (sin ph, -cos ph)               (masa puntual m)

Lagrangiana:
    L = 1/2 m [R^2 th'^2 + l'^2 + l^2 ph'^2 + 2 R l' th' sin(th+ph) + 2 R l th' ph' cos(th+ph)]
        - m g (R cos th - l cos ph) - 1/2 k (l - l0)^2

NOTA: con anclaje sin masa la matriz de masas es singular (det M = 0): el anclaje libre y sin
masa obliga a que el resorte sea radial (sin(th+ph)=0) salvo que no tenga tension. Para poder
integrar las 3 ecuaciones se anade un anclaje de masa pequena m_a (con su gravedad).
Con m_a -> 0 se recupera el limite del enunciado.

Uso:
    python resorte_aro.py            # ventana interactiva
    python resorte_aro.py --test     # comprobacion numerica + PNG sin ventana
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp

TMAX = 20.0
FPS = 50


# ----------------------------------------------------------------------------- fisica
def mass_matrix(q, p):
    th, l, ph = q
    s, c = np.sin(th + ph), np.cos(th + ph)
    m, ma, R = p["m"], p["ma"], p["R"]
    return np.array([[(m + ma) * R**2, m * R * s, m * R * l * c],
                     [m * R * s,       m,         0.0],
                     [m * R * l * c,   0.0,       m * l**2]])


def rhs(t, y, p):
    th, l, ph, dth, dl, dph = y
    m, ma, R, k, l0, g = p["m"], p["ma"], p["R"], p["k"], p["l0"], p["g"]
    s, c = np.sin(th + ph), np.cos(th + ph)
    M = mass_matrix(y[:3], p)
    F = np.array([
        m * (-2 * R * dl * dph * c + R * l * dph**2 * s) + (m + ma) * g * R * np.sin(th),
        m * (-R * dth**2 * c + l * dph**2) - k * (l - l0) + m * g * np.cos(ph),
        m * (-2 * l * dl * dph + R * l * dth**2 * s) - m * g * l * np.sin(ph),
    ])
    acc = np.linalg.solve(M, F)
    return np.concatenate(([dth, dl, dph], acc))


def energy(y, p):
    q, dq = y[:3], y[3:]
    th, l, ph = q
    T = 0.5 * dq @ mass_matrix(q, p) @ dq
    V = (p["m"] * p["g"] * (p["R"] * np.cos(th) - l * np.cos(ph))
         + p["ma"] * p["g"] * p["R"] * np.cos(th) + 0.5 * p["k"] * (l - p["l0"]) ** 2)
    return T + V


def simulate(p, y0, tmax=TMAX, fps=FPS):
    def colapso(t, y, p):          # el resorte no puede tener longitud <= 0
        return y[1] - 1e-3
    colapso.terminal = True
    t_eval = np.arange(0.0, tmax, 1.0 / fps)
    sol = solve_ivp(rhs, (0, tmax), y0, method="DOP853", t_eval=t_eval,
                    rtol=1e-7, atol=1e-9, args=(p,), events=colapso)
    E = np.array([energy(sol.y[:, i], p) for i in range(sol.y.shape[1])])
    return sol.t, sol.y, E


def positions(Y, p):
    th, l, ph = Y[0], Y[1], Y[2]
    A = np.array([p["R"] * np.sin(th), p["R"] * np.cos(th)])
    P = A + np.array([l * np.sin(ph), -l * np.cos(ph)])
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
    fig.canvas.manager.set_window_title("Resorte sobre aro vertical - Lagrange") if hasattr(fig.canvas, "manager") and fig.canvas.manager else None

    ax = fig.add_axes([0.04, 0.40, 0.46, 0.57])
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax_ang = fig.add_axes([0.58, 0.80, 0.39, 0.16])
    ax_l = fig.add_axes([0.58, 0.61, 0.39, 0.14], sharex=ax_ang)
    ax_E = fig.add_axes([0.58, 0.42, 0.39, 0.14], sharex=ax_ang)

    # (nombre, etiqueta, min, max, valor, paso)
    fis = [("R", "R (m)", 0.3, 3.0, 1.0, 0.05),
           ("m", "m (kg)", 0.1, 5.0, 1.0, 0.05),
           ("k", "k (N/m)", 1.0, 200.0, 30.0, 1.0),
           ("l0", "l0 (m)", 0.1, 2.5, 1.0, 0.05),
           ("g", "g (m/s2)", 0.0, 20.0, 9.81, 0.01),
           ("ma", "m_ancla (kg)", 0.005, 0.5, 0.05, 0.005)]
    ini = [("th0", "theta0 (deg)", -180, 180, 60.0, 1),
           ("l_i", "l(0) (m)", 0.1, 3.5, 1.5, 0.05),
           ("ph0", "phi0 (deg)", -180, 180, 20.0, 1),
           ("dth0", "theta' (rad/s)", -10, 10, 0.0, 0.1),
           ("dl0", "l' (m/s)", -5, 5, 0.0, 0.1),
           ("dph0", "phi' (rad/s)", -10, 10, 0.0, 0.1)]
    sliders = {}
    fig.text(0.10, 0.325, "Parametros fisicos", fontweight="bold")
    fig.text(0.58, 0.325, "Condiciones iniciales", fontweight="bold")
    for col, lista in enumerate((fis, ini)):
        x0 = 0.10 if col == 0 else 0.62
        for r, (nm, lab, lo, hi, v, st) in enumerate(lista):
            a = fig.add_axes([x0 + 0.04 * col * 0, 0.285 - r * 0.042, 0.28, 0.022])
            if col == 1:
                a.set_position([0.68, 0.285 - r * 0.042, 0.28, 0.022])
            sliders[nm] = Slider(a, lab, lo, hi, valinit=v, valstep=st)

    st = {"i": 0, "play": True, "data": None}
    ring_t = np.linspace(0, 2 * np.pi, 400)
    ring, = ax.plot([], [], "k-", lw=1.5)
    spring, = ax.plot([], [], "-", color="tab:blue", lw=1.4)
    rod, = ax.plot([], [], "o", color="k", ms=4)
    mass, = ax.plot([], [], "o", color="tab:red", ms=13, zorder=5)
    trail, = ax.plot([], [], "-", color="tab:red", lw=0.8, alpha=0.5)
    ax.plot([0], [0], "k+")
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9)

    lth, = ax_ang.plot([], [], label="theta"); lph, = ax_ang.plot([], [], label="phi")
    ax_ang.set_ylabel("angulos (deg)"); ax_ang.legend(loc="upper right"); ax_ang.grid(alpha=0.25)
    ll, = ax_l.plot([], [], color="tab:green"); ax_l.set_ylabel("l (m)"); ax_l.grid(alpha=0.25)
    lE, = ax_E.plot([], [], color="tab:purple"); ax_E.set_ylabel("E - E0 (J)"); ax_E.grid(alpha=0.25)
    ax_E.set_xlabel("t (s)")
    cur = [a.axvline(0, color="gray", lw=0.8) for a in (ax_ang, ax_l, ax_E)]

    def read():
        v = {k_: s.val for k_, s in sliders.items()}
        p = {k_: v[k_] for k_ in ("R", "m", "k", "l0", "g", "ma")}
        y0 = [np.radians(v["th0"]), v["l_i"], np.radians(v["ph0"]), v["dth0"], v["dl0"], v["dph0"]]
        return p, y0

    def recompute(_=None):
        p, y0 = read()
        t, Y, E = simulate(p, y0)
        st.update(data=(t, Y, E, p), i=0)
        ring.set_data(p["R"] * np.sin(ring_t), p["R"] * np.cos(ring_t))
        A, P = positions(Y, p)
        lim = 1.1 * max(np.abs(P).max(), p["R"])
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        lth.set_data(t, np.degrees(Y[0])); lph.set_data(t, np.degrees(Y[2]))
        ll.set_data(t, Y[1]); lE.set_data(t, E - E[0])
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
        i = st["i"]
        A, P = positions(Y[:, i:i + 1], p)
        A, P = A[:, 0], P[:, 0]
        sx, sy = spring_xy(A, P, 0.05 * p["R"])
        spring.set_data(sx, sy); rod.set_data([A[0]], [A[1]]); mass.set_data([P[0]], [P[1]])
        Ak, Pk = positions(Y[:, max(0, i - 120):i + 1], p)
        trail.set_data(Pk[0], Pk[1])
        for c_ in cur:
            c_.set_xdata([t[i], t[i]])
        info.set_text(f"t = {t[i]:6.2f} s\ntheta = {np.degrees(Y[0, i]):7.1f} deg\n"
                      f"l     = {Y[1, i]:7.3f} m\nphi   = {np.degrees(Y[2, i]):7.1f} deg\n"
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
    fig._anim, fig._widgets = anim, (sliders, b1, b2)   # evita que el recolector los elimine
    return fig, frame, st


# ----------------------------------------------------------------------------- test
def self_test():
    import matplotlib
    matplotlib.use("Agg")
    p = dict(R=1.0, m=1.0, k=30.0, l0=1.0, g=9.81, ma=0.05)
    y0 = [np.radians(60), 1.5, np.radians(20), 0, 0, 0]
    t, Y, E = simulate(p, y0)
    print(f"pasos: {len(t)}  t_final={t[-1]:.2f}  deriva de energia max = {np.abs(E - E[0]).max():.2e} J (E0={E[0]:.3f})")
    print("l min/max:", Y[1].min().round(3), Y[1].max().round(3))
    fig, frame, st = build_gui()
    st["i"] = 400
    frame(0)
    fig.savefig("/tmp/work/prueba.png", dpi=80)
    print("PNG guardado")


if __name__ == "__main__":
    if "--test" in sys.argv:
        self_test()
    else:
        import matplotlib.pyplot as plt
        build_gui()
        plt.show()
