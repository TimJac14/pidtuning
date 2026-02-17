import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st

# =========================================================
# Streamlit Setup
# =========================================================
st.set_page_config(layout="wide", page_title="PID Tuning Simulator")

# =========================================================
# Time helper (Berlin time)
# =========================================================
def now_berlin():
    return datetime.now(ZoneInfo("Europe/Berlin"))

# =========================================================
# Helpers
# =========================================================
def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def eu_to_pct(x_eu, lrv, urv):
    span = (urv - lrv)
    if abs(span) < 1e-12:
        span = 1.0
    return 100.0 * (x_eu - lrv) / span

def fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

# =========================================================
# Metrics
# =========================================================
def metrics_servo(t_min, sp_eu, pv_eu, op_pct, t_step_min, band=0.02):
    t = np.asarray(t_min); sp = np.asarray(sp_eu); pv = np.asarray(pv_eu); op = np.asarray(op_pct)
    k0 = int(np.searchsorted(t, t_step_min, side="left"))
    k0 = min(max(k0, 0), len(t)-1)

    sp0 = sp[k0-1] if k0 > 0 else sp[0]
    sp1 = sp[-1]
    step = sp1 - sp0
    eps = 1e-12
    step_abs = abs(step) if abs(step) > eps else eps

    n_tail = max(int(0.1 * len(t)), 5)
    pv_final = float(np.mean(pv[-n_tail:]))

    if step >= 0:
        pv_peak = float(np.max(pv[k0:])) if k0 < len(pv) else float(np.max(pv))
        overshoot = max(0.0, (pv_peak - sp1) / step_abs * 100.0)
    else:
        pv_min = float(np.min(pv[k0:])) if k0 < len(pv) else float(np.min(pv))
        overshoot = max(0.0, (sp1 - pv_min) / step_abs * 100.0)

    target10 = sp0 + 0.1 * step
    target90 = sp0 + 0.9 * step
    rising = step >= 0

    def first_cross(y, target, start, rising=True):
        if rising:
            for k in range(start, len(y)):
                if y[k] >= target:
                    return k
        else:
            for k in range(start, len(y)):
                if y[k] <= target:
                    return k
        return None

    k10 = first_cross(pv, target10, k0, rising=rising)
    k90 = first_cross(pv, target90, k0, rising=rising)
    rise = None
    if k10 is not None and k90 is not None and k90 >= k10:
        rise = float(t[k90] - t[k10])

    tol = band * step_abs
    settling = None
    in_band = np.abs(pv - sp1) <= tol
    for k in range(k0, len(t)):
        if np.all(in_band[k:]):
            settling = float(t[k] - t_step_min)
            break

    dt = np.diff(t)
    dt = np.append(dt, dt[-1] if len(dt) else 0.0)
    e = sp - pv
    e_post = e[k0:]
    t_post = t[k0:] - t_step_min
    dt_post = dt[k0:]
    iae = float(np.sum(np.abs(e_post) * dt_post))
    ise = float(np.sum((e_post**2) * dt_post))
    itae = float(np.sum((np.abs(e_post) * t_post) * dt_post))

    sse = float(sp1 - pv_final)

    return {
        "Overshoot [%]": overshoot,
        "Rise time [min]": rise,
        "Settling time [min]": settling,
        "Steady-state error [EU]": sse,
        "IAE [EU·min]": iae,
        "ISE [EU²·min]": ise,
        "ITAE [EU·min²]": itae,
        "Max OP [%]": float(np.max(op)),
        "Min OP [%]": float(np.min(op)),
    }

def metrics_disturbance(t_min, sp_eu, pv_eu, op_pct, t_step_min, band=0.02):
    t = np.asarray(t_min); sp = np.asarray(sp_eu); pv = np.asarray(pv_eu); op = np.asarray(op_pct)
    k0 = int(np.searchsorted(t, t_step_min, side="left"))
    k0 = min(max(k0, 0), len(t)-1)

    sp1 = sp[-1]
    dev = pv - sp1
    max_dev = float(np.max(np.abs(dev[k0:])) if k0 < len(dev) else np.max(np.abs(dev)))

    tol = band * max(1e-6, max_dev)
    settling = None
    in_band = np.abs(dev) <= tol
    for k in range(k0, len(t)):
        if np.all(in_band[k:]):
            settling = float(t[k] - t_step_min)
            break

    dt = np.diff(t)
    dt = np.append(dt, dt[-1] if len(dt) else 0.0)
    e = sp - pv
    e_post = e[k0:]
    t_post = t[k0:] - t_step_min
    dt_post = dt[k0:]
    iae = float(np.sum(np.abs(e_post) * dt_post))
    ise = float(np.sum((e_post**2) * dt_post))
    itae = float(np.sum((np.abs(e_post) * t_post) * dt_post))

    return {
        "Max deviation [EU]": max_dev,
        "Recovery time [min]": settling,
        "IAE [EU·min]": iae,
        "ISE [EU²·min]": ise,
        "ITAE [EU·min²]": itae,
        "Max OP [%]": float(np.max(op)),
        "Min OP [%]": float(np.min(op)),
    }

# =========================================================
# Deadtime (minutes)
# =========================================================
class DeadTimeMin:
    def __init__(self, dead_min: float, dt_min: float, init: float = 0.0):
        dt_min = max(dt_min, 1e-12)
        self.n = int(round(max(dead_min, 0.0) / dt_min))
        self.buf = deque([init] * self.n, maxlen=self.n) if self.n > 0 else None

    def step(self, u: float) -> float:
        if self.n <= 0:
            return u
        self.buf.append(u)
        return self.buf[0]

# =========================================================
# Aspen/dmcplus models in MINUTES
# gain is EU/%OP
# =========================================================
class PT1_Aspen_Min:
    def __init__(self, gain: float, tau_min: float, dead_min: float, dt_min: float, y0: float = 0.0, u0: float = 0.0):
        self.K = float(gain)
        self.tau = max(float(tau_min), 1e-12)
        self.dt = float(dt_min)
        self.y = float(y0)
        self.delay = DeadTimeMin(dead_min, dt_min, init=u0)

    def step(self, u: float) -> float:
        u_d = self.delay.step(float(u))
        dy = (-self.y + self.K * u_d) / self.tau
        self.y += dy * self.dt
        return self.y

class PT2_AspenSecondOrder_Min:
    def __init__(self, gain: float, tau_min: float, damp: float, dead_min: float, dt_min: float, y0: float = 0.0, u0: float = 0.0):
        self.K = float(gain)
        self.tau = max(float(tau_min), 1e-12)
        self.zeta = float(damp)
        self.dt = float(dt_min)
        self.y = float(y0)
        self.ydot = 0.0
        self.delay = DeadTimeMin(dead_min, dt_min, init=u0)

    def step(self, u: float) -> float:
        u_d = self.delay.step(float(u))
        tau = self.tau
        z = self.zeta
        yddot = (self.K/(tau**2))*u_d - (2*z/tau)*self.ydot - (1/(tau**2))*self.y
        self.ydot += yddot * self.dt
        self.y += self.ydot * self.dt
        return self.y

# =========================================================
# Honeywell EqA..EqE PID in MINUTES, PV% domain
# Output is delta OP% around 0; OP0 handled outside as bias
# =========================================================
class HoneywellPID_Min_PVpct:
    def __init__(self, eq: str, Kc: float, Ti_min: float, Td_min: float, dt_min: float,
                 reverse: bool = False, d_filter_min: float = 0.0):
        self.eq = eq
        self.Kc = -float(Kc) if reverse else float(Kc)
        self.Ti = max(float(Ti_min), 0.0)
        self.Td = max(float(Td_min), 0.0)
        self.dt = max(float(dt_min), 1e-12)
        self.d_filter = max(float(d_filter_min), 0.0)

        self.i = 0.0
        self.prev_e = 0.0
        self.prev_pv = 0.0
        self.d_f = 0.0

    def _p_sig(self, e, pv_pct):
        if self.eq in ("EqA", "EqB", "EqE"):
            return e
        if self.eq == "EqC":
            return -pv_pct
        if self.eq == "EqD":
            return 0.0
        return e

    def step(self, sp_pct: float, pv_pct: float):
        e = float(sp_pct - pv_pct)

        if self.eq == "EqA":
            use_i = True; d_on_pv = False; use_d = self.Td > 1e-12
        elif self.eq == "EqB":
            use_i = True; d_on_pv = True;  use_d = self.Td > 1e-12
        elif self.eq == "EqC":
            use_i = True; d_on_pv = True;  use_d = self.Td > 1e-12
        elif self.eq == "EqD":
            use_i = True; d_on_pv = False; use_d = False
        elif self.eq == "EqE":
            use_i = False; d_on_pv = False; use_d = False
        else:
            use_i = True; d_on_pv = True;  use_d = self.Td > 1e-12

        p_sig = self._p_sig(e, pv_pct)
        up = self.Kc * p_sig

        ui = 0.0
        if use_i and self.Ti > 1e-12:
            self.i += (self.Kc / self.Ti) * e * self.dt
            ui = self.i

        ud = 0.0
        if use_d:
            if d_on_pv:
                raw = -self.Kc * self.Td * (pv_pct - self.prev_pv) / self.dt
            else:
                raw =  self.Kc * self.Td * (e - self.prev_e) / self.dt

            if self.d_filter > 1e-12:
                a = self.dt / (self.d_filter + self.dt)
                self.d_f += a * (raw - self.d_f)
                ud = self.d_f
            else:
                ud = raw

        self.prev_e = e
        self.prev_pv = float(pv_pct)
        return up + ui + ud

    def bumpless_init(self, sp_pct: float, pv_pct: float):
        e0 = float(sp_pct - pv_pct)
        self.prev_e = e0
        self.prev_pv = float(pv_pct)
        self.d_f = 0.0

        if self.eq == "EqE":
            return
        if self.eq == "EqD":
            self.i = 0.0
            return

        p_sig0 = self._p_sig(e0, pv_pct)
        if self.Ti > 1e-12:
            self.i = -self.Kc * p_sig0
        else:
            self.i = 0.0

# =========================================================
# IMC recommendations in PV% domain
# gain EU/%OP -> %PV/%OP via PV span
# =========================================================
def imc_lambda(tau, dead, mode):
    if mode == "conservative":
        return 3.0 * tau
    if mode == "normal":
        return max(2.0 * tau, 2.0 * dead)
    if mode == "aggressive":
        return 1.0 * tau
    raise ValueError("mode")

def imc_tune(model_type, gain_eu_per_op, tau, dead, pv_span_eu, mode="normal", controller_type="PI"):
    lam = imc_lambda(tau, dead, mode)
    denom = (lam + dead)

    Kpct = 0.0 if abs(pv_span_eu) < 1e-12 else (100.0 * gain_eu_per_op / pv_span_eu)  # %PV/%OP
    if abs(Kpct) < 1e-12:
        Kc = 0.0
    else:
        Kc = tau / (Kpct * denom)  # %OP/%PV

    if model_type == "PT1":
        Ti = tau
        Td = 0.0
    else:
        Ti = 2.0 * tau
        Td = 0.5 * tau if controller_type == "PID" else 0.0

    return Kc, Ti, Td, lam

# =========================================================
# Simulation
# OP(t) = clamp(OP0 + PID_delta + OP_dist, OPmin, OPmax)
# Modes:
#  - "SP": SP step, no OP disturbance
#  - "OP": SP constant, OP disturbance step
# =========================================================
def simulate_case(
    model_type: str,
    gain_eu_per_op: float, tau_min: float, dead_min: float, damp: float,
    pv_lrv: float, pv_urv: float,
    op_min: float, op_max: float,
    eq: str, Kc: float, Ti_min: float, Td_min: float, reverse: bool, d_filter_min: float,
    dt_min: float, t_end_min: float,
    test_mode: str,                 # "SP" or "OP"
    sp0_eu: float, sp_step_eu: float,
    op0_pct: float, op_step_pct: float,
    t_step_min: float
):
    dt_min = max(float(dt_min), 1e-12)
    n = int(t_end_min / dt_min) + 1
    t = np.linspace(0, t_end_min, n)

    sp = np.ones(n) * float(sp0_eu)
    op_dist = np.zeros(n)

    if test_mode == "SP":
        sp[t >= float(t_step_min)] = float(sp0_eu) + float(sp_step_eu)
    else:
        op_dist[t >= float(t_step_min)] = float(op_step_pct)

    pv0 = float(sp0_eu)
    op0 = clamp(float(op0_pct), float(op_min), float(op_max))

    if model_type == "PT1":
        proc = PT1_Aspen_Min(gain=gain_eu_per_op, tau_min=tau_min, dead_min=dead_min, dt_min=dt_min, y0=pv0, u0=op0)
    else:
        proc = PT2_AspenSecondOrder_Min(gain=gain_eu_per_op, tau_min=tau_min, damp=damp, dead_min=dead_min, dt_min=dt_min, y0=pv0, u0=op0)

    pid = HoneywellPID_Min_PVpct(eq=eq, Kc=Kc, Ti_min=Ti_min, Td_min=Td_min, dt_min=dt_min, reverse=reverse, d_filter_min=d_filter_min)

    sp0_pct = eu_to_pct(sp0_eu, pv_lrv, pv_urv)
    pv0_pct = eu_to_pct(pv0,    pv_lrv, pv_urv)
    pid.bumpless_init(sp0_pct, pv0_pct)

    pv = np.zeros(n); op = np.zeros(n)
    pv[0] = pv0; op[0] = op0

    for k in range(1, n):
        sp_pct = eu_to_pct(sp[k], pv_lrv, pv_urv)
        pv_pct = eu_to_pct(pv[k-1], pv_lrv, pv_urv)

        du = pid.step(sp_pct, pv_pct)
        op_cmd = op0 + du + op_dist[k]
        op[k] = clamp(op_cmd, op_min, op_max)

        pv[k] = proc.step(op[k])

    return t, sp, pv, op

# =========================================================
# Plot (dual axis)
# =========================================================
def plot_trends(t, sp, pv, op, title, op_min, op_max):
    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=170)

    ax.step(t, sp, where="post", label="SP [EU]", linewidth=2.0)
    ax.plot(t, pv, label="PV [EU]", linewidth=3.0)
    ax.set_xlabel("t [min]")
    ax.set_ylabel("PV / SP [EU]")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    y_min = float(min(np.min(sp), np.min(pv)))
    y_max = float(max(np.max(sp), np.max(pv)))
    pad = max(0.5, 0.08 * (y_max - y_min + 1e-9))
    ax.set_ylim(y_min - pad, y_max + pad)

    ax2 = ax.twinx()
    ax2.plot(t, op, label="OP [%]", linewidth=2.0, color="green")
    ax2.set_ylabel("OP [%]")

    op_lo = float(np.min(op)); op_hi = float(np.max(op))
    span = max(1e-6, op_hi - op_lo)
    pad2 = max(1.0, 0.10 * span)
    lo = max(float(op_min), op_lo - pad2)
    hi = min(float(op_max), op_hi + pad2)
    if hi - lo < 2.0:
        mid = 0.5 * (hi + lo)
        lo = max(float(op_min), mid - 1.5)
        hi = min(float(op_max), mid + 1.5)
    ax2.set_ylim(lo, hi)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")

    return fig

# =========================================================
# Streamlit App
# =========================================================
st.title("PID Tuning Simulator (Streamlit)")

# --- session defaults ---
def ss_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

# PID sets
ss_init("before", dict(eq="EqB", reverse=False, Kc=0.5, Ti=12.0, Td=0.0, d_filter=0.0))
ss_init("after",  dict(eq="EqB", reverse=False, Kc=0.5, Ti=12.0, Td=0.0, d_filter=0.0))
ss_init("imc_last", "")

# Cases (full snapshots)
ss_init("cases", [])
ss_init("case_name", f"Case {now_berlin().strftime('%Y-%m-%d %H:%M:%S')}")
ss_init("case_comment", "")

# last sim persistence
ss_init("last_sim", None)
ss_init("last_metrics", None)

# =========================================================
# Case snapshot / restore
# =========================================================
def snapshot_case() -> dict:
    """Snapshot ALL settings + PID before/after into one dict."""
    return {
        "name": (st.session_state.get("case_name","") or "").strip() or f"Case {now_berlin().strftime('%Y-%m-%d %H:%M:%S')}",
        "comment": (st.session_state.get("case_comment","") or "").strip(),
        "ts": now_berlin().strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            # Prozess / ranges / work / test / sim
            "proc_model_type": st.session_state.get("proc_model_type"),
            "proc_gain": st.session_state.get("proc_gain"),
            "proc_tau": st.session_state.get("proc_tau"),
            "proc_dead": st.session_state.get("proc_dead"),
            "proc_damp": st.session_state.get("proc_damp"),

            "rng_pv_lrv": st.session_state.get("rng_pv_lrv"),
            "rng_pv_urv": st.session_state.get("rng_pv_urv"),
            "rng_op_min": st.session_state.get("rng_op_min"),
            "rng_op_max": st.session_state.get("rng_op_max"),

            "wrk_sp0": st.session_state.get("wrk_sp0"),
            "wrk_op0": st.session_state.get("wrk_op0"),
            "wrk_op0_auto": st.session_state.get("wrk_op0_auto"),

            "tst_mode": st.session_state.get("tst_mode"),
            "tst_sp_step": st.session_state.get("tst_sp_step"),
            "tst_op_step": st.session_state.get("tst_op_step"),
            "tst_t_step": st.session_state.get("tst_t_step"),

            "sim_dt": st.session_state.get("sim_dt"),
            "sim_t_end": st.session_state.get("sim_t_end"),

            # PID widgets
            "before_eq": st.session_state.get("before_eq"),
            "before_reverse": st.session_state.get("before_reverse"),
            "before_Kc": st.session_state.get("before_Kc"),
            "before_Ti": st.session_state.get("before_Ti"),
            "before_Td": st.session_state.get("before_Td"),
            "before_df": st.session_state.get("before_df"),

            "after_eq": st.session_state.get("after_eq"),
            "after_reverse": st.session_state.get("after_reverse"),
            "after_Kc": st.session_state.get("after_Kc"),
            "after_Ti": st.session_state.get("after_Ti"),
            "after_Td": st.session_state.get("after_Td"),
            "after_df": st.session_state.get("after_df"),
        }
    }

def restore_case(case: dict):
    """Restore snapshot into session_state and rerun."""
    data = (case or {}).get("data", {})
    # write all known keys
    for k, v in data.items():
        st.session_state[k] = v

    # Update dicts used elsewhere (optional, keeps consistency)
    st.session_state.before = dict(
        eq=st.session_state.get("before_eq", "EqB"),
        reverse=bool(st.session_state.get("before_reverse", False)),
        Kc=float(st.session_state.get("before_Kc", 0.5)),
        Ti=float(st.session_state.get("before_Ti", 12.0)),
        Td=float(st.session_state.get("before_Td", 0.0)),
        d_filter=float(st.session_state.get("before_df", 0.0)),
    )
    st.session_state.after = dict(
        eq=st.session_state.get("after_eq", "EqB"),
        reverse=bool(st.session_state.get("after_reverse", False)),
        Kc=float(st.session_state.get("after_Kc", 0.5)),
        Ti=float(st.session_state.get("after_Ti", 12.0)),
        Td=float(st.session_state.get("after_Td", 0.0)),
        d_filter=float(st.session_state.get("after_df", 0.0)),
    )
    st.rerun()

# =========================================================
# Sidebar: use explicit keys so restore works reliably
# =========================================================
with st.sidebar:
    st.header("Prozess (dmcplus)")
    model_type = st.selectbox("Modell", ["PT1", "PT2"], index=1, key="proc_model_type")
    gain = st.number_input("Gain (EU/%OP)", value=1.954, format="%.6f", key="proc_gain")
    tau  = st.number_input("Tau (min)", value=6.0, format="%.6f", key="proc_tau")
    dead = st.number_input("Deadtime (min)", value=1.0, format="%.6f", key="proc_dead")
    damp = st.number_input("Damp ζ (nur PT2)", value=1.0, format="%.6f",
                           disabled=(st.session_state.proc_model_type != "PT2"), key="proc_damp")

    st.header("Ranges (DCS)")
    pv_lrv = st.number_input("PV LRV (EU)", value=0.0, format="%.6f", key="rng_pv_lrv")
    pv_urv = st.number_input("PV URV (EU)", value=200.0, format="%.6f", key="rng_pv_urv")
    op_min = st.number_input("OP Min (%)", value=0.0, format="%.3f", key="rng_op_min")
    op_max = st.number_input("OP Max (%)", value=100.0, format="%.3f", key="rng_op_max")

    st.header("Arbeitspunkt")
    sp0 = st.number_input("SP0 (EU)", value=102.0, format="%.6f", key="wrk_sp0")
    op0 = st.number_input("OP0 (%)", value=52.0, format="%.6f", key="wrk_op0")
    op0_auto = st.checkbox("OP0 aus SP0/Gain schätzen", value=True, key="wrk_op0_auto")

    st.header("Test")
    test_mode = st.selectbox("Modus", ["SP (Servo)", "OP (Disturbance)"], index=0, key="tst_mode")
    if test_mode.startswith("SP"):
        sp_step = st.number_input("SP Step (EU)", value=1.0, format="%.6f", key="tst_sp_step")
        # store 0 for op_step
        st.session_state.tst_op_step = 0.0
    else:
        op_step = st.number_input("OP Step (%)", value=1.0, format="%.6f", key="tst_op_step")
        st.session_state.tst_sp_step = 0.0
    t_step = st.number_input("t_step (min)", value=1.0, format="%.6f", key="tst_t_step")

    st.header("Simulation")
    dt = st.number_input("dt (min)", value=0.01, format="%.6f", key="sim_dt")
    t_end = st.number_input("t_end (min)", value=30.0, format="%.3f", key="sim_t_end")

    st.divider()
    st.subheader("IMC → Nachher")
    imc_mode = st.selectbox("Aggressivität", ["conservative", "normal", "aggressive"], index=1, key="imc_mode")
    imc_ctrl = st.selectbox("PI / PID", ["PI", "PID"], index=0, key="imc_ctrl")

    col_imc1, col_imc2 = st.columns(2)
    with col_imc1:
        do_imc = st.button("IMC berechnen", type="secondary")
    with col_imc2:
        do_sim = st.button("Simulieren", type="primary")

    st.divider()
    st.subheader("Cases (alle Einstellungen)")
    st.text_input("Case Name", key="case_name")
    st.text_area("Case Kommentar", key="case_comment", height=80)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("Case speichern"):
            st.session_state.cases.append(snapshot_case())
            # kein grüner Erfolgstext
    with col_c2:
        if st.button("Alle Cases löschen"):
            st.session_state.cases.clear()
            st.rerun()

    if st.session_state.cases:
        st.caption("Gespeicherte Cases")
        for i, c in enumerate(st.session_state.cases):
            head = f"{c.get('name','(ohne Name)')} — {c.get('ts','')}"
            with st.expander(head, expanded=False):
                if c.get("comment"):
                    st.caption(c["comment"])
                a, b = st.columns(2)
                with a:
                    if st.button("Laden", key=f"case_load_{i}"):
                        restore_case(c)
                with b:
                    if st.button("Löschen", key=f"case_del_{i}"):
                        st.session_state.cases.pop(i)
                        st.rerun()

# =========================================================
# PID panels (with keys so restore works)
# =========================================================
# Initialize PID widget keys once from dicts
if "before_eq" not in st.session_state:
    st.session_state.before_eq = st.session_state.before["eq"]
    st.session_state.before_reverse = st.session_state.before["reverse"]
    st.session_state.before_Kc = st.session_state.before["Kc"]
    st.session_state.before_Ti = st.session_state.before["Ti"]
    st.session_state.before_Td = st.session_state.before["Td"]
    st.session_state.before_df = st.session_state.before["d_filter"]

if "after_eq" not in st.session_state:
    st.session_state.after_eq = st.session_state.after["eq"]
    st.session_state.after_reverse = st.session_state.after["reverse"]
    st.session_state.after_Kc = st.session_state.after["Kc"]
    st.session_state.after_Ti = st.session_state.after["Ti"]
    st.session_state.after_Td = st.session_state.after["Td"]
    st.session_state.after_df = st.session_state.after["d_filter"]

top = st.container()
with top:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.subheader("PID vorher")
        st.selectbox("EQ (vorher)", ["EqA","EqB","EqC","EqD","EqE"], key="before_eq")
        st.checkbox("Reverse (vorher)", key="before_reverse")
        st.number_input("Kc (vorher)", format="%.6f", key="before_Kc")
        st.number_input("Ti (min) (vorher)", format="%.6f", key="before_Ti")
        st.number_input("Td (min) (vorher)", format="%.6f", key="before_Td")
        st.number_input("D-Filter (min) (vorher)", format="%.6f", key="before_df")

    with c2:
        st.subheader("PID nachher")
        st.selectbox("EQ (nachher)", ["EqA","EqB","EqC","EqD","EqE"], key="after_eq")
        st.checkbox("Reverse (nachher)", key="after_reverse")
        st.number_input("Kc (nachher)", format="%.6f", key="after_Kc")
        st.number_input("Ti (min) (nachher)", format="%.6f", key="after_Ti")
        st.number_input("Td (min) (nachher)", format="%.6f", key="after_Td")
        st.number_input("D-Filter (min) (nachher)", format="%.6f", key="after_df")

    with c3:
        st.subheader("Tools")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Vorher → Nachher"):
                for fld in ["eq","reverse","Kc","Ti","Td","df"]:
                    st.session_state[f"after_{fld}"] = st.session_state[f"before_{fld}"]
                st.rerun()
        with b2:
            if st.button("Nachher → Vorher"):
                for fld in ["eq","reverse","Kc","Ti","Td","df"]:
                    st.session_state[f"before_{fld}"] = st.session_state[f"after_{fld}"]
                st.rerun()

        if st.session_state.imc_last:
            st.caption(st.session_state.imc_last)

# Keep dicts in sync (used later)
st.session_state.before = dict(
    eq=st.session_state.before_eq,
    reverse=bool(st.session_state.before_reverse),
    Kc=float(st.session_state.before_Kc),
    Ti=float(st.session_state.before_Ti),
    Td=float(st.session_state.before_Td),
    d_filter=float(st.session_state.before_df),
)
st.session_state.after = dict(
    eq=st.session_state.after_eq,
    reverse=bool(st.session_state.after_reverse),
    Kc=float(st.session_state.after_Kc),
    Ti=float(st.session_state.after_Ti),
    Td=float(st.session_state.after_Td),
    d_filter=float(st.session_state.after_df),
)

# =========================================================
# IMC button action -> set AFTER keys
# =========================================================
if 'do_imc' in locals() and do_imc:
    span = float(st.session_state.rng_pv_urv - st.session_state.rng_pv_lrv)
    Kc_imc, Ti_imc, Td_imc, lam = imc_tune(
        model_type=st.session_state.proc_model_type,
        gain_eu_per_op=float(st.session_state.proc_gain),
        tau=float(st.session_state.proc_tau),
        dead=float(st.session_state.proc_dead),
        pv_span_eu=float(span),
        mode=st.session_state.imc_mode,
        controller_type=st.session_state.imc_ctrl
    )
    st.session_state.after_Kc = float(Kc_imc)
    st.session_state.after_Ti = float(Ti_imc)
    st.session_state.after_Td = float(Td_imc)
    st.session_state.imc_last = f"IMC({st.session_state.imc_mode},{st.session_state.imc_ctrl}): λ={lam:.3f} min | Kc={Kc_imc:.4f} | Ti={Ti_imc:.4f} | Td={Td_imc:.4f}"
    st.rerun()

# =========================================================
# Simulation (persist results so they stay visible)
# =========================================================
if 'do_sim' in locals() and do_sim:
    pv_lrv = float(st.session_state.rng_pv_lrv)
    pv_urv = float(st.session_state.rng_pv_urv)
    op_min = float(st.session_state.rng_op_min)
    op_max = float(st.session_state.rng_op_max)
    dt = float(st.session_state.sim_dt)
    t_end = float(st.session_state.sim_t_end)

    if abs(pv_urv - pv_lrv) < 1e-9:
        st.error("PV URV und LRV dürfen nicht gleich sein.")
    elif dt <= 0 or t_end <= 0:
        st.error("dt und t_end müssen > 0 sein.")
    else:
        # OP0 auto option
        op0 = float(st.session_state.wrk_op0)
        if bool(st.session_state.wrk_op0_auto):
            if abs(float(st.session_state.proc_gain)) < 1e-12:
                st.warning("Gain ist 0 → OP0 aus SP0/Gain nicht möglich. OP0 bleibt wie eingegeben.")
            else:
                op0 = clamp(float(st.session_state.wrk_sp0) / float(st.session_state.proc_gain), op_min, op_max)
                st.session_state.wrk_op0 = op0

        mode = "SP" if st.session_state.tst_mode.startswith("SP") else "OP"
        damp_used = float(st.session_state.proc_damp) if st.session_state.proc_model_type == "PT2" else 0.0

        sp_step = float(st.session_state.tst_sp_step)
        op_step = float(st.session_state.tst_op_step)

        common = dict(
            model_type=st.session_state.proc_model_type,
            gain_eu_per_op=float(st.session_state.proc_gain),
            tau_min=float(st.session_state.proc_tau),
            dead_min=float(st.session_state.proc_dead),
            damp=float(damp_used),
            pv_lrv=pv_lrv,
            pv_urv=pv_urv,
            op_min=op_min,
            op_max=op_max,
            dt_min=dt,
            t_end_min=t_end,
            test_mode=mode,
            sp0_eu=float(st.session_state.wrk_sp0),
            sp_step_eu=float(sp_step),
            op0_pct=float(op0),
            op_step_pct=float(op_step),
            t_step_min=float(st.session_state.tst_t_step),
        )

        t1, sp1, pv1, op1 = simulate_case(
            **common,
            eq=st.session_state.before["eq"],
            Kc=float(st.session_state.before["Kc"]),
            Ti_min=float(st.session_state.before["Ti"]),
            Td_min=float(st.session_state.before["Td"]),
            reverse=bool(st.session_state.before["reverse"]),
            d_filter_min=float(st.session_state.before["d_filter"]),
        )
        t2, sp2, pv2, op2 = simulate_case(
            **common,
            eq=st.session_state.after["eq"],
            Kc=float(st.session_state.after["Kc"]),
            Ti_min=float(st.session_state.after["Ti"]),
            Td_min=float(st.session_state.after["Td"]),
            reverse=bool(st.session_state.after["reverse"]),
            d_filter_min=float(st.session_state.after["d_filter"]),
        )

        # Metrics
        if mode == "SP":
            m1 = metrics_servo(t1, sp1, pv1, op1, t_step_min=float(st.session_state.tst_t_step), band=0.02)
            m2 = metrics_servo(t2, sp2, pv2, op2, t_step_min=float(st.session_state.tst_t_step), band=0.02)
            keys = ["Overshoot [%]", "Rise time [min]", "Settling time [min]", "Steady-state error [EU]",
                    "IAE [EU·min]", "ISE [EU²·min]", "ITAE [EU·min²]", "Max OP [%]", "Min OP [%]"]
        else:
            m1 = metrics_disturbance(t1, sp1, pv1, op1, t_step_min=float(st.session_state.tst_t_step), band=0.02)
            m2 = metrics_disturbance(t2, sp2, pv2, op2, t_step_min=float(st.session_state.tst_t_step), band=0.02)
            keys = ["Max deviation [EU]", "Recovery time [min]",
                    "IAE [EU·min]", "ISE [EU²·min]", "ITAE [EU·min²]", "Max OP [%]", "Min OP [%]"]

        st.session_state.last_sim = dict(t1=t1, sp1=sp1, pv1=pv1, op1=op1,
                                         t2=t2, sp2=sp2, pv2=pv2, op2=op2,
                                         op_min=op_min, op_max=op_max, mode=mode)
        st.session_state.last_metrics = dict(m1=m1, m2=m2, keys=keys)
        st.rerun()

# =========================================================
# Render last results if present
# =========================================================
last = st.session_state.get("last_sim")
lm = st.session_state.get("last_metrics")

if last is not None and lm is not None:
    pcol1, pcol2 = st.columns([1, 1], gap="large")
    with pcol1:
        st.subheader("Trend: Vorher")
        fig1 = plot_trends(last["t1"], last["sp1"], last["pv1"], last["op1"], "PID Regler vorher", last["op_min"], last["op_max"])
        st.pyplot(fig1, clear_figure=True, use_container_width=True)
        plt.close(fig1)
    with pcol2:
        st.subheader("Trend: Nachher")
        fig2 = plot_trends(last["t2"], last["sp2"], last["pv2"], last["op2"], "PID Regler nachher", last["op_min"], last["op_max"])
        st.pyplot(fig2, clear_figure=True, use_container_width=True)
        plt.close(fig2)

    rows = []
    for k in lm["keys"]:
        nd = 2 if ("OP" in k or "Overshoot" in k) else 3
        rows.append([k, fmt(lm["m1"].get(k), nd), fmt(lm["m2"].get(k), nd)])

    st.subheader("Kennzahlen (Vorher vs. Nachher)")
    st.table({"Kennzahl": [r[0] for r in rows],
              "Vorher":   [r[1] for r in rows],
              "Nachher":  [r[2] for r in rows]})
else:
    st.info("Links in der Sidebar **IMC berechnen** oder **Simulieren** drücken.")
