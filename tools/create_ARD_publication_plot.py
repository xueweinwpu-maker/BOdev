"""
This script loads a saved SurrogateBackedPipeline and generates a combined,
publication-quality ARD relevance plot for specified outputs.
MODIFIED: Set sharey=False to allow independent y-axes for each subplot,
as the variable relevance ranking is different for each model.
MODIFIED BY AI: Changed the color scheme to match a user-provided reference image.
"""
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.font_manager as fm

# Prefer "Cambria Math" if available, otherwise fall back to "Cambria" then default sans-serif.
preferred_font = "Cambria Math"
available_names = {f.name for f in fm.fontManager.ttflist}
if preferred_font in available_names:
    mpl.rcParams['font.family'] = preferred_font
    # Apply Cambria Math to math text rendering where possible
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = preferred_font
    mpl.rcParams['mathtext.it'] = preferred_font
    mpl.rcParams['mathtext.bf'] = preferred_font
else:
    # graceful fallback
    mpl.rcParams['font.family'] = "Cambria" if "Cambria" in available_names else mpl.rcParams.get('font.family', 'sans-serif')

# --- NEW: prefer Times New Roman for actual figure text, but keep Cambria logic above ---
preferred_final = "Times New Roman"
if preferred_final in available_names:
    mpl.rcParams['font.family'] = preferred_final
else:
    # Keep Cambria entries available while selecting a serif family that prefers Times-like fonts
    serif_list = []
    if "Times New Roman" in available_names:
        serif_list.append("Times New Roman")
    if "Times" in available_names:
        serif_list.append("Times")
    if "Cambria" in available_names:
        serif_list.append("Cambria")
    if "Cambria Math" in available_names:
        serif_list.append("Cambria Math")
    # extend with any existing serif fallbacks
    serif_list.extend(mpl.rcParams.get('font.serif', []))
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = serif_list
# Add project root to path to allow for imports
# Ensure the path is correct relative to where you run the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# It's good practice to handle potential import errors
try:
    from surrogates.pipelines import SurrogateBackedPipeline
    from surrogates.model import GPyTorchSurrogateModel
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the script is run from the 'tools' directory or that the project root is in the Python path.")
    sys.exit(1)

def generate_combined_ard_plot(pipeline_path: str):
    """
    Loads a saved SurrogateBackedPipeline and generates a combined ARD relevance plot.
    """
    if not os.path.exists(pipeline_path):
        print(f"Error: Pipeline file not found at '{pipeline_path}'")
        print("Please ensure you have run 'run_validation.py' first and that the path is correct.")
        return
        
    print(f"Loading pipeline from {pipeline_path}...")
    try:
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
    except Exception as e:
        print(f"An error occurred while loading the pipeline: {e}")
        return

    # Define the outputs we want to plot
    outputs_to_plot = ['cl', 'cd', 'Volume']
    
    for out in outputs_to_plot:
        if out not in pipeline.models:
            print(f"Error: Model for '{out}' not found in the loaded pipeline.")
            return

    # --- Create the Combined Figure ---
    num_plots = len(outputs_to_plot)
    # CRITICAL CHANGE: Set sharey=False to allow independent y-axes
    # Increase overall figure size (wider + taller) and allow independent y-axes.
    # Use a slightly narrower per-panel width and tighten spacing between panels
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_plots,
        figsize=(8 * num_plots, 12),  # narrower width per subplot
        sharey=False
    )
    # Tighten horizontal spacing so subfigures are closer
    fig.subplots_adjust(wspace=0.20)
    
    # --- AI MODIFICATION: Define the new custom colormap ---
    # Colors are visually sampled from the user's reference image (image_d03bc0.png)
    # The colormap goes from dark blue -> light blue -> white/gray -> orange -> red
    colors = [
        (0.0, '#0d235f'),    # Dark Blue (for low relevance values)
        (0.25, '#5291c9'),   # Medium Blue
        (0.5, '#f0f0f0'),    # White/Light Gray (mid-point)
        (0.75, '#f07f50'),   # Orange
        (1.0, '#c22222')     # Red (for high relevance values)
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)
    # --- END AI MODIFICATION ---
    
    print("Generating plots for each output...")

    for i, coeff_name in enumerate(outputs_to_plot):
        model_wrapper = pipeline.models[coeff_name]
        var_names = pipeline.input_cols
        
        # Define subplot titles with labels (a), (b), (c)
        if coeff_name == 'cl':
            title = f'(a) Lift Coefficient ($C_L$)'
        elif coeff_name == 'cd':
            title = f'(b) Drag Coefficient ($C_D$)'
        elif coeff_name == 'Volume':
            title = f'(c) Internal Volume'
        else:
            title = f'({chr(97+i)}) {coeff_name}'

        # Call the original plotting function on the specific subplot axis
        # The function will draw the bars with its default colors first.
        model_wrapper.plot_ard_relevance(name=title, var_names=var_names, ax=axes[i], save_plot=False)

        # --- AI MODIFICATION: Recolor the bars using the custom colormap ---
        # The plot_ard_relevance function creates a sorted horizontal bar chart.
        # The bars are stored in ax.patches, sorted from lowest value (bottom) to highest (top).
        bars = axes[i].patches
        num_bars = len(bars)
        
        # We generate a range of colors from our custom colormap.
        # np.linspace creates evenly spaced values from 0 to 1, which map to the colormap.
        color_map_values = custom_cmap(np.linspace(0, 1, num_bars))
        
        # We iterate through each bar and its corresponding new color and apply it.
        for bar, color in zip(bars, color_map_values):
            bar.set_color(color)
        # --- END AI MODIFICATION ---

        # --- NEW: compress (zoom) bar lengths (widths) to reduce visual dominance ---
        try:
            # Choose a power < 1 to compress dynamic range (e.g. 0.6); adjust as needed
            compress_power = getattr(pipeline, "ard_length_compress_power", 0.6)
            widths = np.array([bar.get_width() for bar in bars], dtype=float)
            # Avoid zero/negative issues; preserve sign if necessary
            signs = np.sign(widths)
            abs_w = np.abs(widths)
            # Apply power-law compression
            new_widths = signs * (abs_w ** compress_power)
            # If all widths are zero (degenerate), skip
            if new_widths.size > 0 and np.max(new_widths) > 0:
                for bar, nw in zip(bars, new_widths):
                    bar.set_width(float(nw))
                # Adjust x-axis to fit compressed bars
                ax = axes[i]
                ax.set_xlim(0.0, float(np.max(new_widths) * 1.05))
        except Exception:
            # If anything fails, silently continue (keep original lengths)
            pass
        # --- END NEW ---

        # --- NEW: enlarge fonts for publication-quality figures ---
        font_title = 26
        font_axis = 24
        font_tick = 22
        font_legend = 22

        # Ensure the subplot title uses the larger font
        try:
            axes[i].set_title(title, fontsize=font_title)
        except Exception:
            pass

        # Increase tick label size and axis label sizes
        axes[i].tick_params(axis='both', which='major', labelsize=font_tick)
        # If the plotting function set an x-label / y-label, increase their fontsize
        try:
            xlabel = axes[i].get_xlabel()
            if xlabel:
                axes[i].set_xlabel(xlabel, fontsize=font_axis)
        except Exception:
            pass
        try:
            ylabel = axes[i].get_ylabel()
            if ylabel:
                axes[i].set_ylabel(ylabel, fontsize=font_axis)
        except Exception:
            pass

        # Enlarge potential legend text
        leg = axes[i].get_legend()
        if leg is not None:
            leg.set_fontsize(font_legend)
            leg.set_frame_on(False)
        # --- END NEW ---
#
    # Final adjustments and save
    # fig.suptitle('Comparison of Design Variable Relevance', fontsize=22, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Frame/tick adjustments to avoid tick label overlap ---
    # Increase horizontal space between subplots and margins
    # Final layout: keep margins but reduce inter-subplot gap
    fig.subplots_adjust(wspace=0.18, left=0.08, right=0.98, top=0.95, bottom=0.12)
    
    # Reduce number of x-ticks and rotate labels for each axis to avoid crowding
    for ax in axes if isinstance(axes, (list, np.ndarray)) else [axes]:
        try:
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))      # cap number of x ticks
            ax.tick_params(axis='x', rotation=45, labelsize=font_tick - 4)
            ax.tick_params(axis='y', labelsize=font_tick - 2)
            plt.setp(ax.get_xticklabels(), ha='right')         # right-align rotated labels
            plt.setp(ax.get_yticklabels(), va='center')        # vertical alignment for y labels
        except Exception:
            pass
    # --- end frame/tick adjustments ---
    
    # Save the final figure inside the same directory as the pipeline
    # NOTE: The output filename is changed to avoid overwriting the original file.
    output_dir = os.path.dirname(pipeline_path) if os.path.dirname(pipeline_path) else '.'
    output_filename_pdf = os.path.join(output_dir, "publication_ARD_comparison_recolored.pdf")
    output_filename_png = os.path.join(output_dir, "publication_ARD_comparison_recolored.png")

    # Save both vector/pdf and high-resolution PNG for raster requirements
    fig.savefig(output_filename_pdf, dpi=500, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(output_filename_png, dpi=500, bbox_inches='tight', pad_inches=0.1)
    print(f"\nSuccessfully saved combined publication plot to '{output_filename_pdf}' and '{output_filename_png}'")

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # IMPORTANT: UPDATE THIS PATH after running 'run_validation.py'.
    # It will now be inside the 'validation_results' directory.
    # Example: 'validation_results/38_DV_Model_Ma5_ARD_20251009-073000/final_pipeline.pkl'
    
    PIPELINE_FILE = 'trained_models/38_DV_Model_Ma5_ARD.pkl' # <-- UPDATE THIS PATH

    if not os.path.exists(PIPELINE_FILE):
        print(f"ERROR: The specified pipeline file does not exist:\n{PIPELINE_FILE}")
        print("Please run 'run_validation.py' first and update the path above.")
    else:
        generate_combined_ard_plot(PIPELINE_FILE)