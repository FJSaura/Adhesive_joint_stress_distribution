
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def omega_double_lap(G, h, Eo, to, Ei, ti):
    return np.sqrt((G / h) * (1.0 / (Eo * to) + 2.0 / (Ei * ti)))


def tau_double_lap(x, P, b, l, Eo, to, Ei, ti, G, h, alpha_o=0.0, alpha_i=0.0, dT=0.0):
    x = np.asarray(x)
    w = omega_double_lap(G, h, Eo, to, Ei, ti)

    term1 = (P * w / (4.0 * b * np.sinh(w * l / 2.0))) * np.cosh(w * x)

    bracket_mech = (P * w / (4.0 * b * np.cosh(w * l / 2.0))) * (
        (2.0 * Eo * to - Ei * ti) / (2.0 * Eo * to + Ei * ti)
    )

    denom_thermal = (1.0 / (Eo * to) + 2.0 / (Ei * ti)) * np.cosh(w * l / 2.0)
    bracket_therm = ((alpha_i - alpha_o) * dT * w) / denom_thermal

    return term1 + (bracket_mech + bracket_therm) * np.sinh(w * x)


def peak_tau(params):
    x = np.linspace(-params["l"] / 2.0, params["l"] / 2.0, 1200)
    tau = tau_double_lap(
        x,
        params["P"], params["b"], params["l"],
        params["Eo"], params["to"], params["Ei"], params["ti"],
        params["G"], params["h"],
        params["alpha_o"], params["alpha_i"], params["dT"]
    )
    return np.max(np.abs(tau))


def relative_sensitivity(base_params, frac=0.2):
    baseline = peak_tau(base_params)
    out = []

    for k in ["P", "b", "l", "Eo", "to", "Ei", "ti", "G", "h", "alpha_o", "alpha_i", "dT"]:
        val = base_params[k]
        if val == 0:
            continue

        p1 = base_params.copy()
        p2 = base_params.copy()
        p1[k] = val * (1 + frac)
        p2[k] = val * (1 - frac)

        y1 = peak_tau(p1)
        y2 = peak_tau(p2)

        avg_pct_change = 100.0 * 0.5 * (abs(y1 - baseline) + abs(y2 - baseline)) / baseline
        out.append((k, avg_pct_change))

    out.sort(key=lambda x: x[1], reverse=True)
    return out


IN_TO_M = 0.0254
LBF_TO_N = 4.4482216152605
PSI_TO_PA = 6894.757293168
GPA_TO_PA = 1e9
PA_TO_MPA = 1e-6
PA_TO_PSI = 1 / PSI_TO_PA


def length_to_si(value, system):
    if system == "SI":
        return value * 1e-3
    return value * IN_TO_M


def force_to_si(value, system):
    if system == "SI":
        return value
    return value * LBF_TO_N


def modulus_to_si(value, system):
    if system == "SI":
        return value * GPA_TO_PA
    return value * PSI_TO_PA


def tau_from_si(value_pa, system):
    if system == "SI":
        return value_pa * PA_TO_MPA
    return value_pa * PA_TO_PSI


st.set_page_config(page_title="Double Lap Joint Explorer", layout="wide")
st.title("Double lap joint shear-stress explorer")

st.subheader("Model based in Shigley's Mechanical Engineering Design 8th Ed: Chapter 9-9; Eq 9-7")

st.image("Diagram.png", caption="Double lap joint geometry")

st.latex(r"""
\tau(x)=\frac{P\omega}{4b\sinh(\omega l/2)}\cosh(\omega x)
+\left[
\frac{P\omega}{4b\cosh(\omega l/2)}
\left(\frac{2E_ot_o-E_it_i}{2E_ot_o+E_it_i}\right)
+\frac{(\alpha_i-\alpha_o)\Delta T\,\omega}
{\left(\frac{1}{E_ot_o}+\frac{2}{E_it_i}\right)\cosh(\omega l/2)}
\right]\sinh(\omega x)
""")

st.write("con")

st.latex(r"""
\omega=\sqrt{\frac{G}{h}\left(\frac{1}{E_ot_o}+\frac{2}{E_it_i}\right)}
""")

with st.sidebar:
    st.header("Configuration")
    unit_system = st.radio("Unit system", ["SI", "Imperial"], horizontal=True)
    input_mode = st.radio("Input mode", ["Sliders", "Manual values"], horizontal=False)

    if unit_system == "SI":
        length_label = "mm"
        force_label = "N"
        modulus_label = "GPa"
        stress_label = "MPa"
        alpha_label = "µm/mK"
        default_vals = {
            "b": 1, "l": 15, "to": 0.7, "ti": 1.6, "h": 0.3,
            "Eo": 2.6, "Ei": 73.0, "G": 0.90, "P": 0,
            "dT": 100, "alpha_o": 88.4, "alpha_i": 10,
        }
        ranges = {
            "b": (1.0, 100.0, 1.0),
            "l": (1.0, 200.0, 1.0),
            "to": (0.2, 10.0, 0.1),
            "ti": (0.2, 10.0, 0.1),
            "h": (0.01, 2.0, 0.01),
            "Eo": (1.0, 250.0, 1.0),
            "Ei": (1.0, 250.0, 1.0),
            "G": (0.01, 10.0, 0.01),
            "P": (0.0, 100000.0, 500.0),
            "dT": (-200.0, 200.0, 1.0),
            "alpha_o": (0.0, 100.0, 1.0),
            "alpha_i": (0.0, 100.0, 1.0),
        }
    else:
        length_label = "in"
        force_label = "lbf"
        modulus_label = "psi"
        stress_label = "psi"
        alpha_label = "µin/in·°F"
        default_vals = {
            "b": 1.0, "l": 2.0, "to": 0.08, "ti": 0.08, "h": 0.008,
            "Eo": 10_000_000.0, "Ei": 10_000_000.0, "G": 116_000.0, "P": 2250.0,
            "dT": 0.0, "alpha_o": 12.8, "alpha_i": 12.8,
        }
        ranges = {
            "b": (0.01, 5.0, 0.01),
            "l": (0.01, 10.0, 0.01),
            "to": (0.01, 0.5, 0.005),
            "ti": (0.01, 0.5, 0.005),
            "h": (0.001, 0.1, 0.001),
            "Eo": (100_000.0, 40_000_000.0, 100_000.0),
            "Ei": (100_000.0, 40_000_000.0, 100_000.0),
            "G": (1_000.0, 2_000_000.0, 1_000.0),
            "P": (0.0, 50_000.0, 100.0),
            "dT": (-400.0, 400.0, 1.0),
            "alpha_o": (0.0, 100.0, 0.1),
            "alpha_i": (0.0, 100.0, 0.1),
        }

    def get_input(name, label):
        lo, hi, step = ranges[name]
        default = default_vals[name]
        if input_mode == "Sliders":
            return st.slider(label, lo, hi, default, step)
        return st.number_input(label, min_value=lo, max_value=hi, value=default, step=step, format="%.6g")

    st.subheader("Geometry")
    b_user = get_input("b", f"Width b [{length_label}]")
    l_user = get_input("l", f"Overlap length l [{length_label}]")
    to_user = get_input("to", f"Outer thickness to [{length_label}]")
    ti_user = get_input("ti", f"Inner thickness ti [{length_label}]")
    h_user = get_input("h", f"Adhesive thickness h [{length_label}]")

    st.subheader("Materials")
    Eo_user = get_input("Eo", f"Eo [{modulus_label}]")
    Ei_user = get_input("Ei", f"Ei [{modulus_label}]")
    G_user = get_input("G", f"Adhesive shear modulus G [{modulus_label}]")

    st.subheader("Load and thermal")
    P_user = get_input("P", f"Applied load P [{force_label}]")
    dT = get_input("dT", "Delta T [K or °F]")
    alpha_o_user = get_input("alpha_o", f"alpha_o [{alpha_label}]")
    alpha_i_user = get_input("alpha_i", f"alpha_i [{alpha_label}]")

    st.subheader("Sensitivity")
    sens_frac_pct = st.slider("Perturbation for sensitivity [%]", 5, 50, 20, 5)

params = {
    "P": force_to_si(P_user, unit_system),
    "b": length_to_si(b_user, unit_system),
    "l": length_to_si(l_user, unit_system),
    "Eo": modulus_to_si(Eo_user, unit_system),
    "to": length_to_si(to_user, unit_system),
    "Ei": modulus_to_si(Ei_user, unit_system),
    "ti": length_to_si(ti_user, unit_system),
    "G": modulus_to_si(G_user, unit_system),
    "h": length_to_si(h_user, unit_system),
    "alpha_o": alpha_o_user * 1e-6,
    "alpha_i": alpha_i_user * 1e-6,
    "dT": dT,
}

x = np.linspace(-params["l"] / 2.0, params["l"] / 2.0, 1200)
tau = tau_double_lap(
    x,
    params["P"], params["b"], params["l"],
    params["Eo"], params["to"], params["Ei"], params["ti"],
    params["G"], params["h"],
    params["alpha_o"], params["alpha_i"], params["dT"]
)
w = omega_double_lap(params["G"], params["h"], params["Eo"], params["to"], params["Ei"], params["ti"])

x_plot = x * (1000 if unit_system == "SI" else 1 / IN_TO_M)
tau_plot = tau_from_si(tau, unit_system)
tau_abs_peak = tau_from_si(np.max(np.abs(tau)), unit_system)
tau_max = tau_from_si(np.max(tau), unit_system)
tau_min = tau_from_si(np.min(tau), unit_system)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_plot, tau_plot)
    ax.set_xlabel(f"x [{length_label}]")
    ax.set_ylabel(f"tau(x) [{stress_label}]")
    ax.set_title("Shear stress distribution")
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.metric(f"Peak |tau| [{stress_label}]", f"{tau_abs_peak:,.3f}")
    st.metric(f"tau max [{stress_label}]", f"{tau_max:,.3f}")
    st.metric(f"tau min [{stress_label}]", f"{tau_min:,.3f}")
    st.metric("omega [1/m]", f"{w:.3e}")
    st.metric("G/h [SI]", f"{(params['G'] / params['h']):.3e}")

st.subheader("Variables Sensitivity")


sens = relative_sensitivity(params, frac=sens_frac_pct / 100.0)
names = [k for k, _ in sens]
vals = [v for _, v in sens]

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
ax2.bar(names, vals)
ax2.set_ylabel(f"Average change in peak |tau| for ±{sens_frac_pct}% perturbation [%]")
ax2.set_title("One-at-a-time sensitivity ranking")
ax2.grid(True, axis="y")
st.pyplot(fig2)

st.dataframe(
    [{"parameter": k, "avg_change_in_peak_tau_percent": round(v, 2)} for k, v in sens],
    use_container_width=True
)

with st.expander("Notes"):
    st.markdown("""
- Formulation checked with bibliography examples
- This is a local ranking. Changes with the joint design and operation point.
- If `dT = 0` or `alpha_i = alpha_o`, the thermal term disappears.
- If only thermal effect -> `P = 0`.
- If only mechanical effec ->`dT = 0`.
""")
