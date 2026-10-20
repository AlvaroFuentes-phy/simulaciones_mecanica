# Resorte con Lastre sobre un Aro Vertical (Simulación Lagrangiana)

Este proyecto ofrece una simulación interactiva en Python del comportamiento dinámico de una masa puntual (lastre) unida a un resorte, cuyo extremo superior está anclado a un punto deslizante sobre un aro vertical fijo.

La formulación física utiliza las **Ecuaciones de Euler-Lagrange** para modelar el sistema y resuelve numéricamente la evolución temporal mediante integradores adaptativos.

---

## 📌 Formulación Física y Sistema de Coordenadas

El sistema se describe mediante $3$ coordenadas generalizadas $q = (\theta, l, \phi)$:

1. $\theta$: Posición angular del anclaje del resorte sobre el aro (medida en sentido horario desde el eje $+y$).
2. $l$: Longitud instantánea del resorte.
3. $\phi$: Inclinación angular del resorte respecto a la vertical descendente.

### Posición de las Masas
* **Anclaje $A$ (sobre el aro):**
  $$A = (R \sin\theta, R \cos\theta)$$
* **Masa puntual $P$ (lastre):**
  $$P = A + l \begin{pmatrix} \sin\phi \\ -\cos\phi \end{pmatrix} = \begin{pmatrix} R \sin\theta + l \sin\phi \\ R \cos\theta - l \cos\phi \end{pmatrix}$$

### Lagrangiana ($L = T - V$)
La energía cinética $T$ y la energía potencial $V$ del sistema se definen como:

$$T = \frac{1}{2} m \left[ R^2 \dot{\theta}^2 + \dot{l}^2 + l^2 \dot{\phi}^2 + 2 R \dot{l} \dot{\theta} \sin(\theta + \phi) + 2 R l \dot{\theta} \dot{\phi} \cos(\theta + \phi) \right] + \frac{1}{2} m_a R^2 \dot{\theta}^2$$

$$V = m g (R \cos\theta - l \cos\phi) + m_a g R \cos\theta + \frac{1}{2} k (l - l_0)^2$$

> **Nota técnica (Matriz de Masas y Anclaje):** 
> En el caso límite ideal de un anclaje sin masa ($m_a = 0$), el determinante de la matriz de masas se vuelve singular ($\det \mathbf{M} = 0$), lo que fuerza la alineación radial estricta del resorte ($\sin(\theta+\phi)=0$) salvo que no exista tensión. Para resolver numéricamente las ecuaciones diferenciales sin singularidades algebraicas, el modelo introduce una pequeña masa auxiliar en el anclaje $m_a \ll m$. Al hacer $m_a \to 0$, se recupera progresivamente el comportamiento exacto del modelo teórico.

---

## 🚀 Requisitos e Instalación

Para ejecutar la simulación se requiere **Python 3.8+** y las siguientes bibliotecas de cálculo científico e interfaz gráfica:

```bash
pip install numpy scipy matplotlib
```

---

## 💻 Uso de la Aplicación

### 1. Modo Interactivo (GUI)
Para abrir la interfaz gráfica con controles deslizantes (sliders) y animación en tiempo real:

```bash
python resorte_aro.py
```

#### Características de la GUI:
* **Simulación en Tiempo Real:** Visualización gráfica de la masa, el resorte y la trayectoria (estela) del lastre.
* **Gráficos Dinámicos:** Seguimiento continuo de los ángulos ($\theta, \phi$), la longitud $l(t)$ y la conservación/deriva de energía total ($E - E_0$).
* **Control de Parámetros:** Modificación interactiva de parámetros físicos ($R$, $m$, $k$, $l_0$, $g$, $m_a$) y condiciones iniciales ($\theta_0, l(0), \phi_0, \dot{\theta}_0, \dot{l}_0, \dot{\phi}_0$).
* **Optimización (Debounce):** Los cálculos se posponen brevemente al arrastrar los sliders para garantizar fluidez en la interfaz.

### 2. Modo de Prueba / Ejecución en Chasis Sin Pantalla (--test)
Para validar el integrador numérico sin desplegar la ventana interactiva (útil en servidores o entornos CI/CD):

```bash
python resorte_aro.py --test
```
Este comando ejecuta una integración usando el algoritmo `DOP853` (Runge-Kutta explícito de orden 8/5), comprueba la precisión en la conservación de la energía y genera un renderizado estático del estado actual en `/tmp/work/prueba.png`.

---

## ⚙️ Estructura del Código

| Función | Descripción |
| :--- | :--- |
| `mass_matrix(q, p)` | Calcula la matriz de masas $\mathbf{M}(q)$ dependiente de la posición. |
| `rhs(t, y, p)` | Define el sistema de ODEs $\mathbf{M} \cdot \ddot{q} = \mathbf{F}$ y retorna las derivadas de primer orden. |
| `energy(y, p)` | Evalúa la energía mecánica total ($E = T + V$) del sistema. |
| `simulate(p, y0)` | Integra numéricamente las ecuaciones mediante `scipy.integrate.solve_ivp` con control de eventos de colapso ($l \le 0$). |
| `build_gui()` | Construye la interfaz gráfica basada en `matplotlib.widgets` y gestiona las animaciones. |
