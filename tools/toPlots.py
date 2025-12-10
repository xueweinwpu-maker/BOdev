import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Refined Configuration ---
# Modern color theme and frame styling
sns.set_theme(style="whitegrid")

# Refined color palette
plt.rcParams.update({
    'font.family': 'serif',
    'figure.dpi': 300,
    'axes.facecolor': '#FAFAFA',          # Light gray background
    'figure.facecolor': 'white',
    'axes.edgecolor': '#2E2E2E',          # Dark gray frame
    'axes.linewidth': 1.5,               # Thicker frame
    'xtick.color': '#2E2E2E',
    'ytick.color': '#2E2E2E',
    'text.color': '#2E2E2E',
    'axes.labelcolor': '#2E2E2E',
    'grid.color': '#E0E0E0',             # Subtle grid
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
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..')

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

    ax.set_title('Comparative Optimization Efficiency')
    ax.set_xlabel('Number of Evaluations')
    ax.set_ylabel('Feasible Hypervolume')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(0, max(len(df_30), len(df_38)))
    ax.set_ylim(0)

    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved Figure 3 to {output_path}")

def plot_figure_4_objective_space(df_30, df_38, output_path):
    """Generates Figure 4: Final Objective Space and Pareto Fronts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    
    titles = ['(a) 30-DV (WDA-Informed)', '(b) 38-DV (Conventional FFD)']
    dfs = [df_30, df_38]
    
    for i, (ax, df) in enumerate(zip(axes, dfs)):
        feasible_mask = df['con_Maintain_L/D_Ratio'] >= 0
        infeasible = df[~feasible_mask]
        feasible = df[feasible_mask]

        ax.scatter(infeasible['obj_Maximize_Volume'], infeasible['con_Maintain_L/D_Ratio'],
                   color=colors['neutral'], alpha=0.4, s=25, label='Infeasible')
        
        sc = ax.scatter(feasible['obj_Maximize_Volume'], feasible['con_Maintain_L/D_Ratio'],
                        c=feasible['evaluation'], cmap='viridis', s=40, label='Feasible')

        if not feasible.empty:
            best_feasible = feasible.loc[feasible['obj_Maximize_Volume'].idxmax()]
            ax.scatter(best_feasible['obj_Maximize_Volume'], best_feasible['con_Maintain_L/D_Ratio'],
                       marker='*', s=250, edgecolor='black', facecolor=colors['accent'], 
                       zorder=10, label='Best Feasible')
        
        ax.axhline(0, color=colors['accent'], linestyle='--', linewidth=1.5, label='Feasibility Boundary')
        ax.set_title(titles[i])
        ax.set_xlabel('Objective: Maximize Volume')
    
    axes[0].set_ylabel('Constraint: L/D Ratio - Baseline L/D')
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.25, top=0.9)
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.05),
              frameon=True, fancybox=True, shadow=True)

    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.65])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Evaluation Number')
    
    fig.suptitle('Final Objective Space Comparison', fontsize=16, y=0.98)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved Figure 4 to {output_path}")

def plot_figure_5_parallel_coords(df_30, df_38, output_path, n_top=15):
    """Generates Figure 5: Aligned parallel coordinates with corrected custom grouping."""
    
    all_dv_cols_ordered = [f'offline_var_{i}' for i in range(38)]
    
    def prep_df_for_plotting(df, n_top, total_dvs, missing_dvs_indices=[]):
        feasible = df[df['con_Maintain_L/D_Ratio'] >= 0].copy()
        top_feasible = feasible.nlargest(n_top, 'obj_Maximize_Volume')
        
        current_dv_cols = [f'offline_var_{i}' for i in range(total_dvs)]
        plot_df = top_feasible[current_dv_cols + ['obj_Maximize_Volume']].copy()
        
        for i in missing_dvs_indices:
            plot_df[f'offline_var_{i}'] = np.nan
            
        plot_df = plot_df[all_dv_cols_ordered + ['obj_Maximize_Volume']]
        rename_map = {f'offline_var_{i}': f'DV {i+1}' for i in range(38)}
        rename_map['obj_Maximize_Volume'] = 'Volume'
        plot_df.rename(columns=rename_map, inplace=True)
        return plot_df

    df_30_plot = prep_df_for_plotting(df_30, n_top, 30, missing_dvs_indices=range(30, 38))
    df_38_plot = prep_df_for_plotting(df_38, n_top, 38)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharey=True)

    pd.plotting.parallel_coordinates(df_30_plot, 'Volume', ax=ax1, colormap='viridis', alpha=0.7)
    pd.plotting.parallel_coordinates(df_38_plot, 'Volume', ax=ax2, colormap='viridis', alpha=0.7)

    def style_grouped_axis(ax):
        # Refined background regions with professional colors
        ax.axvspan(-0.5, 11, facecolor=colors['accent'], alpha=0.1, zorder=-1)
        ax.axvspan(29, 37, facecolor=colors['accent'], alpha=0.1, zorder=-1) # DVs 31-38
        ax.axvspan(11, 29, facecolor=colors['primary'], alpha=0.1, zorder=-1) # DVs 13-30
        
        tick_locs = [0, 11, 12, 29, 30, 37]
        tick_labs = ['DV 1', 'DV 12', 'DV 13', 'DV 30', 'DV 31', 'DV 38']
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labs, rotation=45, ha='right')
        ax.xaxis.grid(False)

    style_grouped_axis(ax1)
    style_grouped_axis(ax2)

    ax1.set_title('(a) Top Designs: 30-DV (WDA-Informed)', loc='left')
    ax2.set_title('(b) Top Designs: 38-DV (Conventional FFD)', loc='left')
    if ax1.get_legend() is not None: ax1.get_legend().remove()
    if ax2.get_legend() is not None: ax2.get_legend().remove()
    
    ax1.set_ylabel('Normalized DV Value')
    ax2.set_ylabel('Normalized DV Value')
    
    norm = plt.Normalize(vmin=min(df_30_plot['Volume'].min(), df_38_plot['Volume'].min()),
                         vmax=max(df_38_plot['Volume'].max(), df_38_plot['Volume'].max()))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.75])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Objective: Maximize Volume')
    
    fig.suptitle('Analysis of Top Feasible Design Variables', fontsize=16, y=0.99)
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.15, top=0.9, hspace=0.6)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved Figure 5 to {output_path}")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"\nGenerating publication-quality figures in '{OUTPUT_DIR}'...")
    plot_figure_3_convergence(df_30, df_38, os.path.join(OUTPUT_DIR, 'Fig3_Convergence_Refined.png'))
    plot_figure_4_objective_space(df_30, df_38, os.path.join(OUTPUT_DIR, 'Fig4_ObjectiveSpace_Refined.png'))
    plot_figure_5_parallel_coords(df_30, df_38, os.path.join(OUTPUT_DIR, 'Fig5_ParallelCoords_Refined.png'))
    print("\nAll refined figures generated successfully.")