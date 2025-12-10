import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib as mpl

# --- Refined Configuration ---
# Modern color theme and frame styling
sns.set_theme(style="whitegrid")

# Refined color palette
plt.rcParams.update({
    'font.family': 'serif',
    'figure.dpi': 300,
    'axes.facecolor': '#FAFAFA',      # Light gray background
    'figure.facecolor': 'white',
    'axes.edgecolor': '#2E2E2E',      # Dark gray frame
    'axes.linewidth': 1.5,            # Thicker frame
    'xtick.color': '#2E2E2E',
    'ytick.color': '#2E2E2E',
    'text.color': '#2E2E2E',
    'axes.labelcolor': '#2E2E2E',
    'grid.color': '#E0E0E0',          # Subtle grid
    'grid.linewidth': 0.8,
    'grid.alpha': 0.7
})

# Unified color scheme
colors = {
    'primary': '#1f77b4',      # Professional blue
    'secondary': '#ff7f0e',    # Professional orange  
    'accent': '#d62728',       # Professional red
    'success': '#2ca02c',      # Professional green
    'neutral': '#7f7f7f'       # Professional gray
}

# File paths (assumes script is in a 'tools' folder and data is in a 'data' folder)
try:
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    DATA_DIR = os.path.join(project_root, 'data')
    OUTPUT_DIR = project_root
except NameError:
    # Fallback for environments where __file__ is not defined
    DATA_DIR = 'data'
    OUTPUT_DIR = '.'


FILE_30DVS = os.path.join(DATA_DIR, 'full_optimization_history-30dvs.csv')
FILE_38DVS = os.path.join(DATA_DIR, 'full_optimization_history_38dvs.csv')

# --- Data Loading and Preparation ---
try:
    df_30 = pd.read_csv(FILE_30DVS)
    df_38 = pd.read_csv(FILE_38DVS)
    print("Successfully loaded both optimization history files.")
except FileNotFoundError as e:
    print(f"Error: Could not find data file. Make sure your CSV files are in the 'data' directory. Details: {e}")
    exit()

# --- Figure Generation Functions ---

def plot_figure_3_convergence(df_30, df_38, output_path):
    """Generates Figure 3: Comparative Convergence Efficiency."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # Increased font sizes for publication
    font_title = 18
    font_axis = 14
    font_tick = 12
    font_legend = 12

    ax.plot(df_30['evaluation'], df_30['convergence_metric'], marker='o', markersize=4, 
            linestyle='-', color=colors['primary'], label='30-DV (WDA-Informed)')
    ax.plot(df_38['evaluation'], df_38['convergence_metric'], marker='s', markersize=4, 
            linestyle='--', color=colors['secondary'], label='38-DV (Conventional FFD)')

    # Annotations
    conv_thresh = 0.98
    max_hv_30 = np.nanmax(df_30['convergence_metric'])
    max_hv_38 = np.nanmax(df_38['convergence_metric'])
    
    conv_point_30 = df_30[df_30['convergence_metric'] >= conv_thresh * max_hv_30]['evaluation'].iloc[0]
    conv_point_38 = df_38[df_38['convergence_metric'] >= conv_thresh * max_hv_38]['evaluation'].iloc[0]

    ax.axvline(x=conv_point_30, color=colors['primary'], linestyle=':', linewidth=1.5, alpha=0.9)
    ax.axvline(x=conv_point_38, color=colors['secondary'], linestyle=':', linewidth=1.5, alpha=0.9)

    ax.annotate(f'Converged at ~{conv_point_30} evals', xy=(conv_point_30, max_hv_30 * 0.4), xytext=(conv_point_30 + 5, max_hv_30 * 0.4),
                arrowprops=dict(facecolor=colors['neutral'], shrink=0.05, width=1, headwidth=5),
                ha='left', va='center', fontsize=10)
    
    ax.annotate(f'Converged at ~{conv_point_38} evals', xy=(conv_point_38, max_hv_38 * 0.6), xytext=(conv_point_38 + 5, max_hv_38 * 0.6),
                arrowprops=dict(facecolor=colors['neutral'], shrink=0.05, width=1, headwidth=5),
                ha='left', va='center', fontsize=10)

    # ax.set_title('Comparative Optimization Efficiency')
    ax.set_xlabel('Number of Evaluations', fontsize=font_axis)
    ax.set_ylabel('Feasible Hypervolume', fontsize=font_axis)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=font_legend)
    ax.tick_params(axis='both', labelsize=font_tick)
    ax.title.set_fontsize(font_title)
    # ax.set_xlim(0, max(len(df_30), len(df_38)))
    ax.set_xlim(0, len(df_30))  # Focus on the range of the 30-DV case
    print("Number of iteration:", len(df_30))
    ax.set_ylim(0)

    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
    # Line drawing for Fig3 -> save high-resolution bitmap per guideline (1000 dpi)
    plt.savefig(output_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 3 to {output_path}")

def plot_figure_4_objective_space(df_30, df_38, output_path):
    """Generates Figure 4: Final Objective Space and Pareto Fronts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    # Increased font sizes for publication
    font_title = 18
    font_axis = 14
    font_tick = 12
    font_legend = 12

    titles = ['(a) 30-DV (WDA-Informed)', '(b) 38-DV (Conventional FFD)']
    dfs = [df_30, df_38]

    sc = None
    for panel_idx, (ax, df) in enumerate(zip(axes, dfs)):
        feasible_mask = df['con_Maintain_L/D_Ratio'] >= 0
        infeasible = df[~feasible_mask]
        feasible = df[feasible_mask]

        ax.scatter(
            infeasible['obj_Maximize_Volume'],
            infeasible['con_Maintain_L/D_Ratio'],
            color=colors['neutral'],
            alpha=0.4,
            s=25,
            label='Infeasible'
        )

        if not feasible.empty:
            # Blue-Orange color schema: primary (blue) -> secondary (orange)
            wda_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                "wda_blue_orange", [colors['primary'], colors['secondary']]
            )
            sc = ax.scatter(
                feasible['obj_Maximize_Volume'],
                feasible['con_Maintain_L/D_Ratio'],
                c=feasible['evaluation'],
                cmap=wda_cmap,
                s=40,
                label='Feasible',
                edgecolors='none'
            )

            # Ensure a plotting color is available for the Pareto line
            plot_color = colors['primary'] if panel_idx == 0 else colors['secondary']

            # Compute Pareto frontier among feasible points (nondominated in both axes: larger is better)
            pts = feasible[['obj_Maximize_Volume', 'con_Maintain_L/D_Ratio']].values
            # quick nondominated filter (O(n^2) but n is small here)
            is_pareto = []
            for idx_p, p in enumerate(pts):
                dominated = False
                for idx_q, q in enumerate(pts):
                    if idx_q == idx_p:
                        continue
                    # q dominates p if q is >= p in both dims and strictly > in at least one
                    if (q[0] >= p[0] and q[1] >= p[1]) and (q[0] > p[0] or q[1] > p[1]):
                        dominated = True
                        break
                is_pareto.append(not dominated)
            pareto_pts = pts[np.array(is_pareto)]
            if pareto_pts.size > 0:
                # sort by objective for a clean line
                pareto_pts = pareto_pts[pareto_pts[:, 0].argsort()]
                # Mark Pareto frontier as red-outlined scatter (no filled marker)
                label_str = 'Pareto frontier' if panel_idx == 0 else None
                ax.scatter(
                    pareto_pts[:, 0],
                    pareto_pts[:, 1],
                    facecolors='none',          # no fill
                    edgecolors='black',          # black outline
                    linewidths=1.2,
                    s=60,                      # marker size
                    marker='o',
                    label=label_str,
                    zorder=9
                )

        ax.axhline(0, color=colors['accent'], linestyle='--', linewidth=1.5, label='Feasibility Boundary')
        ax.set_title(titles[panel_idx], fontsize=font_title)
        ax.set_xlabel('Objective: Internal Volume', fontsize=font_axis)

    axes[0].set_ylabel(r'Constraint: L/D - $L/D_{baseline}$', fontsize=font_axis)
    for ax in axes:
        ax.tick_params(axis='both', labelsize=font_tick)
        # Soften axes spines (avoid heavy black frame)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor('#BFBFBF')

    if sc is not None:
        cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.65])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        # Soften colorbar outline and ticks for a cleaner look
        cbar.set_label('Evaluation Number', fontsize=font_axis)
        cbar.ax.tick_params(labelsize=font_tick)
        try:
            cbar.outline.set_edgecolor('#BFBFBF')
            cbar.outline.set_linewidth(0.8)
            cbar.ax.tick_params(colors='#2E2E2E')
            # optional: hide the thin spine on colorbar to avoid heavy box
            for spine in cbar.ax.spines.values():
                spine.set_visible(False)
        except Exception:
            pass

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.25, top=0.9)
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, 0.05),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=font_legend
    )
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 4 to {output_path}")

def plot_figure_5_boxplot_distribution(df_30, df_38, output_path, n_top=10):
    """
    Generates an elegant Figure 5 using box plots to show the distribution
    of the top feasible design variables, maintaining the established style.
    """
    
    # --- Local font-size tuning (increase these numbers as needed) ---
    font_title = 20
    font_axis = 18
    font_tick = 12
    font_legend = 18
    # --------------------------------------------------------------
    
    def prep_df(df, n, dv_cols_map):
        feasible = df[df['con_Maintain_L/D_Ratio'] >= 0].copy()
        top_feasible = feasible.nlargest(n, 'obj_Maximize_Volume')
        
        plot_df = top_feasible[list(dv_cols_map.keys())].copy()
        # Melt the dataframe to a long format for seaborn, preserving original names for mapping
        df_melt = plot_df.melt(var_name='Design Variable', value_name='Normalized Value')
        # Map to the new, clean names for plotting
        df_melt['Design Variable'] = df_melt['Design Variable'].map(dv_cols_map)
        return df_melt

    # Define the mapping from CSV columns to plot labels
    dv_map_30 = {f'offline_var_{i}': f'DV {i+1}' for i in range(30)}
    dv_map_38 = {f'offline_var_{i}': f'DV {i+1}' for i in range(38)}
    
    df_30_plot = prep_df(df_30, n_top, dv_map_30)
    df_38_plot = prep_df(df_38, n_top, dv_map_38)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharey=True)

    # --- Plotting ---
    sns.boxplot(ax=ax1, x='Design Variable', y='Normalized Value', data=df_30_plot, color=colors['primary'], fliersize=0, width=0.6)
    sns.stripplot(ax=ax1, x='Design Variable', y='Normalized Value', data=df_30_plot, color=colors['neutral'], size=3, alpha=0.6)
    
    sns.boxplot(ax=ax2, x='Design Variable', y='Normalized Value', data=df_38_plot, color=colors['secondary'], fliersize=0, width=0.6)
    sns.stripplot(ax=ax2, x='Design Variable', y='Normalized Value', data=df_38_plot, color=colors['neutral'], size=3, alpha=0.6)

    # --- Styling and Grouping ---
    def style_grouped_axis(ax, is_30_dv_case):
        # Background coloring
        ax.axvspan(-0.5, 11.5, facecolor=colors['accent'], alpha=0.1, zorder=-1)
        ax.axvspan(11.5, 29.5, facecolor=colors['primary'], alpha=0.1, zorder=-1)
        if not is_30_dv_case:
            ax.axvspan(29.5, 37.5, facecolor=colors['accent'], alpha=0.1, zorder=-1)

        # Set shared x-axis limits and ticks for perfect alignment
        ax.set_xlim(-0.5, 37.5)
        tick_locs = [0, 11,  29, 37]
        tick_labs = ['DV 1', 'DV 12', 'DV 30', 'DV 38']
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labs)
        ax.set_xlabel('')
    
    style_grouped_axis(ax1, is_30_dv_case=True)
    style_grouped_axis(ax2, is_30_dv_case=False)

    ax1.set_title("(a) Top Designs: 30-DV (WDA-Informed)", loc='left', fontsize=font_title)
    ax2.set_title("(b) Top Designs: 38-DV (Conventional FFD)", loc='left', fontsize=font_title)
    
    # --- NEW: shaded infeasible band + crisp dashed boundary line ---
    # Shade infeasible region (below 0) for visual emphasis
    # ymin, ymax = ax1.get_ylim()
    # ax1.axhspan(ymin, 0.0, facecolor=colors['accent'], alpha=0.06, zorder=0)
    # ax2.axhspan(ymin, 0.0, facecolor=colors['accent'], alpha=0.06, zorder=0)

    # Crisp dashed boundary line; label only on the first axis to avoid duplicate legend entries
    boundary_kwargs = dict(color=colors['accent'], linestyle='--', linewidth=1.25, alpha=0.95, zorder=5)
    ax1.axhline(0.0, label='baseline', **boundary_kwargs)
    ax2.axhline(0.0, **boundary_kwargs)
    # --- END NEW ---

    # --- Apply increased font sizes ---
    # Titles
    ax1.title.set_fontsize(font_title)
    ax2.title.set_fontsize(font_title)
    # Y-axis label (shared)
    ax1.set_ylabel("Value", fontsize=font_axis)
    ax1.set_xlabel("Design Variables", fontsize=font_axis)
    ax2.set_ylabel("Value", fontsize=font_axis)
    ax2.set_xlabel("Design Variables", fontsize=font_axis)
    # Tick labels
    for ax in (ax1, ax2):
        ax.tick_params(axis='x', labelsize=font_tick, rotation=0)
        ax.tick_params(axis='y', labelsize=font_tick)
        # increase x tick label padding to avoid clipping
        plt.setp(ax.get_xticklabels(), fontsize=font_tick)
    # Legend font (feasibility boundary)
    leg = ax1.legend(fontsize=font_legend)
    if leg is not None:
        leg.set_frame_on(False)
    # Adjust layout after font changes
    # Increase vertical spacing between the two subplots and give more headroom for larger titles
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.95, hspace=0.35)
    # -------------------------------------------------
    
    # fig.suptitle('Distribution of Top Feasible Design Variables', fontsize=20, y=0.96)
    # Boxplot (line/halftone combination) -> save at 500 dpi per guideline
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Final Box Plot Figure 5 to {output_path}")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"\nGenerating publication-quality figures...")
    plot_figure_3_convergence(df_30, df_38, os.path.join(OUTPUT_DIR, 'Fig3_Final_new.png'))
    plot_figure_4_objective_space(df_30, df_38, os.path.join(OUTPUT_DIR, 'Fig4_Final_new.png'))
    plot_figure_5_boxplot_distribution(df_30, df_38, os.path.join(OUTPUT_DIR, 'Fig5_Final_Boxplot_new.png'))
    print("\nAll final figures generated successfully.")

