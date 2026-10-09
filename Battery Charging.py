"""
Math 3 Project 3 - Battery Charging (RC model)
Part 1: Mathematical model
Part 2: Numerical solution (Euler, Heun, RK4)
Part 3: Programming in Python (every step is printed)
Part 4: Experiment (step-size study + changing R and C)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
# ----------------------------------------------------------------------
# PART 1 - MATHEMATICAL MODEL
# ----------------------------------------------------------------------
Vs, R, C, V0 = 5.0, 10.0, 0.5, 0.0      # source V, ohm, farad, initial V
tau = R * C
T_END = 20.0
 
 
def f(t, V, R=R, C=C, Vs=Vs):
    """dV/dt = (Vs - V) / (R*C)"""
    return (Vs - V) / (R * C)
 
 
def exact(t, R=R, C=C, Vs=Vs, V0=V0):
    """V(t) = Vs - (Vs - V0) * exp(-t / RC)"""
    return Vs - (Vs - V0) * np.exp(-t / (R * C))
 
 
print("=" * 70)
print("PART 1: MATHEMATICAL MODEL")
print("=" * 70)
print("Battery modelled as a capacitor charged through a resistor.")
print("Kirchhoff voltage law:  Vs = R*i + V,   i = C*dV/dt")
print("=> ODE:  dV/dt = (Vs - V) / (R*C),   V(0) = V0")
print(f"Parameters: Vs={Vs} V, R={R} ohm, C={C} F, V0={V0} V")
print(f"Time constant tau = R*C = {tau} s")
print("\nAnalytical solution (separation of variables):")
print("  dV/(Vs - V) = dt/(RC)")
print("  -ln|Vs - V| = t/(RC) + K")
print("  V(t) = Vs - (Vs - V0) * exp(-t/RC)")
print(f"  V(t) = {Vs} - {Vs - V0} * exp(-t/{tau})")
 
# ----------------------------------------------------------------------
# PART 2 - NUMERICAL METHODS (formulas)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 2: NUMERICAL METHODS")
print("=" * 70)
print("Euler : V_{n+1} = V_n + h*f(t_n, V_n)                    (order 1)")
print("Heun  : k1=f(t_n,V_n); k2=f(t_n+h, V_n+h*k1)")
print("        V_{n+1} = V_n + h/2*(k1+k2)                      (order 2)")
print("RK4   : k1=f(t,V); k2=f(t+h/2,V+h/2*k1); k3=f(t+h/2,V+h/2*k2)")
print("        k4=f(t+h,V+h*k3); V_{n+1}=V_n+h/6*(k1+2k2+2k3+k4) (order 4)")
 
 
# ----------------------------------------------------------------------
# PART 3 - PROGRAMMING (every step printed when verbose=True)
# ----------------------------------------------------------------------
def euler(h, verbose=False, R=R, C=C, Vs=Vs, V0=V0, T=T_END):
    n = int(round(T / h))
    t, V = np.zeros(n + 1), np.zeros(n + 1)
    V[0] = V0
    if verbose:
        print(f"{'n':>2} {'t_n':>6} {'V_n':>10} {'f(t_n,V_n)':>12} "
              f"{'V_n+1':>10} {'exact':>10} {'error':>10}")
    for i in range(n):
        slope = f(t[i], V[i], R, C, Vs)
        t[i + 1] = t[i] + h
        V[i + 1] = V[i] + h * slope
        if verbose:
            ex = exact(t[i + 1], R, C, Vs, V0)
            print(f"{i:>2} {t[i]:>6.2f} {V[i]:>10.5f} {slope:>12.5f} "
                  f"{V[i+1]:>10.5f} {ex:>10.5f} {abs(ex-V[i+1]):>10.2e}")
    return t, V
 
 
def heun(h, verbose=False, R=R, C=C, Vs=Vs, V0=V0, T=T_END):
    n = int(round(T / h))
    t, V = np.zeros(n + 1), np.zeros(n + 1)
    V[0] = V0
    if verbose:
        print(f"{'n':>2} {'t_n':>6} {'V_n':>10} {'k1':>10} {'k2':>10} "
              f"{'V_n+1':>10} {'exact':>10} {'error':>10}")
    for i in range(n):
        k1 = f(t[i], V[i], R, C, Vs)
        k2 = f(t[i] + h, V[i] + h * k1, R, C, Vs)
        t[i + 1] = t[i] + h
        V[i + 1] = V[i] + h / 2 * (k1 + k2)
        if verbose:
            ex = exact(t[i + 1], R, C, Vs, V0)
            print(f"{i:>2} {t[i]:>6.2f} {V[i]:>10.5f} {k1:>10.5f} {k2:>10.5f} "
                  f"{V[i+1]:>10.5f} {ex:>10.5f} {abs(ex-V[i+1]):>10.2e}")
    return t, V
 
 
def rk4(h, verbose=False, R=R, C=C, Vs=Vs, V0=V0, T=T_END):
    n = int(round(T / h))
    t, V = np.zeros(n + 1), np.zeros(n + 1)
    V[0] = V0
    if verbose:
        print(f"{'n':>2} {'t_n':>6} {'V_n':>9} {'k1':>9} {'k2':>9} {'k3':>9} "
              f"{'k4':>9} {'V_n+1':>9} {'exact':>9} {'error':>9}")
    for i in range(n):
        k1 = f(t[i], V[i], R, C, Vs)
        k2 = f(t[i] + h / 2, V[i] + h / 2 * k1, R, C, Vs)
        k3 = f(t[i] + h / 2, V[i] + h / 2 * k2, R, C, Vs)
        k4 = f(t[i] + h, V[i] + h * k3, R, C, Vs)
        t[i + 1] = t[i] + h
        V[i + 1] = V[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if verbose:
            ex = exact(t[i + 1], R, C, Vs, V0)
            print(f"{i:>2} {t[i]:>6.2f} {V[i]:>9.5f} {k1:>9.5f} {k2:>9.5f} "
                  f"{k3:>9.5f} {k4:>9.5f} {V[i+1]:>9.5f} {ex:>9.5f} "
                  f"{abs(ex-V[i+1]):>9.2e}")
    return t, V
 
 
print("\n" + "=" * 70)
print("PART 3: STEP-BY-STEP SOLUTION (h = 2 s)")
print("=" * 70)
H = 2.0
print("\n--- Euler ---")
t_e, V_e = euler(H, verbose=True)
print("\n--- Heun (improved Euler) ---")
t_h, V_h = heun(H, verbose=True)
print("\n--- Runge-Kutta 4 ---")
t_r, V_r = rk4(H, verbose=True)
 
# ----------------------------------------------------------------------
# PART 4 - EXPERIMENT
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 4: EXPERIMENTS")
print("=" * 70)
 
# Experiment A: effect of step size on the error at t = T_END
print("\nExperiment A: max error vs step size h")
print(f"{'h':>8} {'Euler':>12} {'Heun':>12} {'RK4':>12}")
steps = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]
errs = {"Euler": [], "Heun": [], "RK4": []}
for h in steps:
    row = []
    for name, method in (("Euler", euler), ("Heun", heun), ("RK4", rk4)):
        t, V = method(h)
        e = np.max(np.abs(V - exact(t)))
        errs[name].append(e)
        row.append(e)
    print(f"{h:>8.3f} {row[0]:>12.3e} {row[1]:>12.3e} {row[2]:>12.3e}")
 
print("\nObserved order of convergence (error ratio when h is halved):")
for name in errs:
    ratios = [errs[name][i] / errs[name][i + 1] for i in range(len(steps) - 1)]
    p = np.log2(ratios[-1])
    print(f"  {name:6s}: last ratio = {ratios[-1]:.2f}  =>  order ~ {p:.2f}")
 
# Experiment B: change R and C -> charging time
print("\nExperiment B: effect of R and C (time to reach 95% of Vs = 3*tau)")
print(f"{'R':>6} {'C':>6} {'tau':>8} {'t_95% (s)':>11} {'V(T) RK4':>10}")
cases = [(5, 0.5), (10, 0.5), (20, 0.5), (10, 1.0), (10, 0.25)]
curves = {}
for Rv, Cv in cases:
    t, V = rk4(0.1, R=Rv, C=Cv, T=T_END)
    curves[(Rv, Cv)] = (t, V)
    print(f"{Rv:>6} {Cv:>6} {Rv*Cv:>8.2f} {3*Rv*Cv:>11.2f} {V[-1]:>10.4f}")
 
# Plots
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
 
tt = np.linspace(0, T_END, 400)
ax[0].plot(tt, exact(tt), "k", lw=2, label="Exact")
ax[0].plot(t_e, V_e, "o--", label="Euler h=2")
ax[0].plot(t_h, V_h, "s--", label="Heun h=2")
ax[0].plot(t_r, V_r, "^--", label="RK4 h=2")
ax[0].set(title="Battery voltage vs time", xlabel="t (s)", ylabel="V (volt)")
ax[0].legend(); ax[0].grid(alpha=0.3)
 
for name, e in errs.items():
    ax[1].loglog(steps, e, "o-", label=name)
ax[1].set(title="Max error vs step size", xlabel="h", ylabel="max |error|")
ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
 
for (Rv, Cv), (t, V) in curves.items():
    ax[2].plot(t, V, label=f"R={Rv}, C={Cv}")
ax[2].set(title="Effect of R and C", xlabel="t (s)", ylabel="V (volt)")
ax[2].legend(); ax[2].grid(alpha=0.3)
 
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/battery_charging_results.png", dpi=130)
print("\nPlot saved: battery_charging_results.png")
 