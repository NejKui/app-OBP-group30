import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Compute the Stationary Distribution and Up-Time Fraction
# ------------------------------------------------------------------
def stationary_distribution(mu, gamma, warm_standby, n, k, r):
    """
    Compute the stationary distribution for a k-out-of-n maintenance system.

    The state is defined as the number of working machines (0 ≤ i ≤ n).
    The system is up if i ≥ k.

    Transitions:
    - Repair (i -> i+1): rate λ(i) = min(n-i, r) * gamma.
    - Failure (i -> i-1):
         • For warm standby: rate = i * mu.
         • For cold standby:
             - If i ≥ k (system is up): only k machines are active → rate = k * mu.
             - If i < k (system is down): all i machines are active → rate = i * mu.
    """

    # Define repair rates
    def repair_rate(i):
        return min(n - i, r) * gamma if i < n else 0         # Transition from state i to i+1

    # Define failure rates
    def failure_rate(i):
        # Transition from state i to i-1 (i > 0)
        if i == 0:
            return 0
        if warm_standby:
            return i * mu
        else:
            return k * mu if i >= k else i * mu

    # the standard birth–death ratio method:
    # Let ratio[i] = π(i)/π(0). Then for i>=0:
    #   π(i+1) = π(i) * (repair_rate(i) / failure_rate(i+1))
    ratios = [1.0]  # π(0) relative value
    for i in range(0, n):
        # For i -> i+1, failure_rate in denominator is for state (i+1)
        ratios.append(ratios[-1] * (repair_rate(i) / failure_rate(i + 1)))

    Z = sum(ratios)
    pi = [val / Z for val in ratios]
    return pi


def system_uptime(pi, k):
    """
    The system is up if the number of working machines i ≥ k.
    The up-time fraction is the sum of π(i) for i = k to n.
    """
    return sum(pi[k:])


# ------------------------------------------------------------------
# Optimal Configuration Search
# ------------------------------------------------------------------
def optimal_configuration(mu, gamma, warm_standby, k,
                          cost_component, cost_repairman, downtime_cost,
                          n_min, n_max, r_min, r_max):
    """
    Search for the optimal configuration by varying n (from k to n_max) and r (from 1 to r_max)
    to minimize the total cost per unit time.

    Total Cost = n*(cost per component) + r*(cost per repairman) + downtime_cost*(1 - up-time)
    """
    best_cost = float('inf')
    best_n = None
    best_r = None
    best_uptime = None

    cost_grid = np.zeros((r_max-r_min+1, n_max - n_min + 1))
    uptime_grid = np.zeros_like(cost_grid)

    for i,n in enumerate(range(n_min, n_max + 1)):
        for j,r in enumerate(range(r_min, r_max + 1)):
            pi = stationary_distribution(mu, gamma, warm_standby, n, k, r)
            uptime = system_uptime(pi, k)
            total_cost = n * cost_component + r * cost_repairman + downtime_cost * (1 - uptime)

            cost_grid[j, i] = total_cost
            uptime_grid[j, i] = uptime

            if total_cost < best_cost:
                best_cost = total_cost
                best_n = n
                best_r = r
                best_uptime = uptime
    return best_n, best_r, best_uptime, best_cost, cost_grid, uptime_grid


# ------------------------------------------------------------------
# Diagram for the Birth–Death Process
# ------------------------------------------------------------------
def birth_death_graph(mu, gamma, warm_standby, n, k, r):
    """
    Generate a Graphviz DOT diagram for the birth–death process.

    States: i = number of working machines (0 ≤ i ≤ n).
    Transitions:
      - From i to i+1 (repair): rate = min(n-i, r) * gamma.
      - From i to i-1 (failure):
            • Warm standby: rate = i * mu.
            • Cold standby: if i ≥ k, rate = k * mu; if i < k, rate = i * mu.
    """
    dot_str = "digraph G {\n    rankdir=LR;\n    node [shape=circle];\n"
    for i in range(n + 1):
        # Transition from i -> i+1 (repair), if possible
        if i < n:
            rep_rate = min(n - i, r) * gamma
            dot_str += f'    "{i}" -> "{i + 1}" [label="λ={rep_rate:.2f}"];\n'
        # Transition from i -> i-1 (failure), if possible
        if i > 0:
            if warm_standby:
                fail_rate = i * mu
            else:
                fail_rate = k * mu if i >= k else i * mu
            dot_str += f'    "{i}" -> "{i - 1}" [label="μ={fail_rate:.2f}"];\n'
    dot_str += "}"
    return dot_str

# ------------------------------------------------------------------
# Heatmaps for grid search
# ------------------------------------------------------------------

def plot_cost_heatmap(cost_grid, n_vals, r_vals, opt_n, opt_r):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cost_grid, xticklabels=n_vals, yticklabels=r_vals, ax=ax,
                cmap="viridis", annot=False, cbar_kws={"label": "Total Cost"})
    ax.set_title("Total Cost")
    ax.set_xlabel("n (Components)")
    ax.set_ylabel("r (Repairmen)")
    ax.scatter(n_vals.index(opt_n) + 0.5, r_vals.index(opt_r) + 0.5, color="red", s=60, label="Optimal")
    ax.legend(loc="lower right", fontsize="small")
    return fig

def plot_uptime_heatmap(uptime_grid, n_vals, r_vals, opt_n, opt_r):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(uptime_grid, xticklabels=n_vals, yticklabels=r_vals, ax=ax,
                cmap="YlGnBu", annot=False, cbar_kws={"label": "Uptime Fraction"})
    ax.set_title("Uptime Fraction")
    ax.set_xlabel("n (Components)")
    ax.set_ylabel("r (Repairmen)")
    ax.scatter(n_vals.index(opt_n) + 0.5, r_vals.index(opt_r) + 0.5, color="red", s=60, label="Optimal")
    ax.legend(loc="lower right", fontsize="small")
    return fig


# ------------------------------------------------------------------
# Streamlit Application Interface
# ------------------------------------------------------------------
st.title("Maintenance System: k-out-of-n Model")

st.markdown("""
This app computes the **stationary distribution** for a *k-out-of-n* maintenance system.
The state is defined as the number of **working** machines (i.e. not failed).
The system is considered UP if at least \( k \) machines are working.

Parameters:
- **Failure rate** (μ) per component.
- **Repair rate** (γ) per repairman.
- **Standby mode:** Components are in *warm* or *cold* standby.
- **n:** Total number of components.
- **k:** Number of components required for the system to function.
- **r:** Number of repairmen.
""")

options = st.radio("Select an option:", ["Compute UP time", "Find optimal configuration"])

if options == "Compute UP time":
    st.header("A) Compute Up-Time")

    st.markdown("**Enter the following parameters:**")
    st.session_state.mu = st.number_input("Failure rate (μ) per component", value=0.1, min_value=0.0, step=0.01)
    st.session_state.gamma = st.number_input("Repair rate (γ) per repairman", value=1.0, min_value=0.0, step=0.1)
    st.session_state.warm_standby = st.checkbox("Warm standby? (Unchecked = Cold standby)", value=True)
    st.session_state.n = st.number_input("Total number of components (n)", value=5, min_value=1, step=1)
    st.session_state.k = st.number_input("Number of components required for system operation (k)", value=3, min_value=1, step=1)
    st.session_state.r = st.number_input("Number of repairmen", value=1, min_value=1, step=1)

    if st.session_state.k > st.session_state.n:
        st.error("The required number of working components (k) cannot exceed n.")
    else:
        if st.button("Compute Up-Time"):
            pi = stationary_distribution(st.session_state.mu, st.session_state.gamma, st.session_state.warm_standby, st.session_state.n, st.session_state.k, st.session_state.r)
            uptime_fraction = system_uptime(pi, st.session_state.k)
            st.write(f"**Up-time fraction:** {uptime_fraction:.4f}")
            st.write("**Stationary distribution** (state i = number of working machines):")
            for i, p in enumerate(pi):
                st.write(f"State {i}: π({i}) = {p:.4f}")
            st.subheader("Birth–Death Process Diagram")
            dot_graph = birth_death_graph(st.session_state.mu, st.session_state.gamma, st.session_state.warm_standby, st.session_state.n, st.session_state.k, st.session_state.r)
            st.graphviz_chart(dot_graph)

# st.markdown("---")
elif options == "Find optimal configuration":
    st.header("B) Optimal Configuration")

    st.markdown("""
    Find the optimal configuration by varying the number of components (n) and repairmen (r).
    Total Cost per unit time is calculated as:

    **Total Cost** = (n × cost per component) + (r × cost per repairman) + (downtime cost × (1 - up-time))
    """)
    st.session_state.mu = st.number_input("Failure rate (μ) per component", value=0.1, min_value=0.0, step=0.01)
    st.session_state.gamma = st.number_input("Repair rate (γ) per repairman", value=1.0, min_value=0.0, step=0.1)
    st.session_state.warm_standby = st.checkbox("Warm standby? (Unchecked = Cold standby)", value=True)
    st.session_state.k = st.number_input("Number of components required for system operation (k)", value=3, min_value=1, step=1)
    st.session_state.cost_component = st.number_input("Cost per component", value=10.0, min_value=0.0, step=0.5)
    st.session_state.cost_repairman = st.number_input("Cost per repairman", value=50.0, min_value=0.0, step=1.0)
    st.session_state.downtime_cost = st.number_input("Downtime cost (per unit time)", value=100.0, min_value=0.0, step=1.0)

    st.markdown("**Search range for optimization:**")
    st.session_state.n_range = st.slider("Range for number of components (n)", min_value=1, max_value=30, value=(st.session_state.k, 30))
    st.session_state.r_range = st.slider("Range for number of repairmen (r)", min_value=1, max_value=30, value=(1, 5))
    st.session_state.n_min, st.session_state.n_max = st.session_state.n_range
    st.session_state.r_min, st.session_state.r_max = st.session_state.r_range
    if st.session_state.n_min < st.session_state.k:
        st.error("Minimum number of components (n_min) must be at least equal to k (number of machines working for the system to be up).")
    elif st.session_state.n_min >= st.session_state.n_max:
        st.error("n_min must be strictly less than n_max.")
    elif st.session_state.r_min >= st.session_state.r_max:
        st.error("r_min must be strictly less than r_max.")
    else:
        if st.button("Optimize Configuration"):
            if st.session_state.k > st.session_state.n_max:
                st.error("Maximum number of machines must be at least equal to k (number of machines working for the system to be up).")
            else:
                opt_n, opt_r, opt_uptime, opt_total_cost, cost_grid, uptime_grid = optimal_configuration(
                    st.session_state.mu, st.session_state.gamma, st.session_state.warm_standby, st.session_state.k,
                    st.session_state.cost_component, st.session_state.cost_repairman, st.session_state.downtime_cost,
                    int(st.session_state.n_min), int(st.session_state.n_max), int(st.session_state.r_min), int(st.session_state.r_max)
                )
                st.markdown("#### Optimal Configuration Results:")
                st.write(f"**Optimal total number of components (n):** {opt_n}")
                st.write(f"**Optimal number of repairmen (r):** {opt_r}")
                st.write(f"**Up-time fraction:** {opt_uptime:.4f}")
                st.write(f"**Minimum Total Cost:** {opt_total_cost:.2f}")

                # Value ranges for axes
                n_vals = list(range(st.session_state.n_min, st.session_state.n_max + 1))
                r_vals = list(range(st.session_state.r_min, st.session_state.r_max + 1))

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Total Cost")
                    fig1 = plot_cost_heatmap(cost_grid, n_vals, r_vals, opt_n, opt_r)
                    st.pyplot(fig1)

                with col2:
                    st.subheader("Uptime Fraction")
                    fig2 = plot_uptime_heatmap(uptime_grid, n_vals, r_vals, opt_n, opt_r)
                    st.pyplot(fig2)

# st.markdown("---")
# st.markdown("""
# ### PDF Submission Instructions
#
# 1. Include a screenshot of the **birth–death process diagram** (which shows the transitions and rates).
# 2. Briefly explain how the stationary distribution is computed:
#    - The state is the number of working machines.
#    - The system is UP if \( i \ge k \).
#    - For **warm standby**, all working machines are active: failure rate = \( i \times \mu \).
#    - For **cold standby**, if \( i \ge k \) only \( k \) machines are active (failure rate = \( k \times \mu \)); if \( i < k \), then failure rate = \( i \times \mu \).
#    - Repairs occur at rate \( \min(n-i, r) \times \gamma \).
# 3. Show a sample example (e.g., with \( \mu=0.1 \), \( \gamma=1.0 \), warm standby, \( n=5 \), \( k=3 \), \( r=1 \)) and discuss the computed up-time.
# 4. Provide the link to your working Streamlit app.
# """)

