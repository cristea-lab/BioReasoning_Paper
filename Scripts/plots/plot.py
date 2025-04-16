'''
Functions for making bar and heat plots in the Deepseek R1 projects
'''

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import math
import os
import seaborn as sns

class MetricPlotter:
    """
    A class to plot bar charts for comparing metrics.
    Provides methods for grouped and simple bar charts.
    Allows optional fixed y-axis limits. Includes dynamic limit calculation.

    Attributes:
        figure_height (float): Default height of the plot figure in inches.
        bar_width (float): Width of each bar in the plot.
        group_gap (float): Minimum gap between groups or bars.
        annotation_digits (int): Number of decimal places for metric annotations.
        text_margin (float): Vertical margin for annotations (relative to data scale 0-1).
        min_figure_width (float): Minimum width of the plot figure.
        width_scale_factor (float): Scaling factor for dynamic figure width.
        percentage (bool): If True, scale y-axis and annotations by 100.
        y_lim_lower (float, optional): Fixed lower y-axis limit (in display scale). Defaults to None (dynamic).
        y_lim_upper (float, optional): Fixed upper y-axis limit (in display scale). Defaults to None (dynamic).
    """
    def __init__(self,
                 figure_height=6,
                 bar_width=0.15,
                 group_gap=0.2,
                 annotation_digits=2,
                 text_margin=0.005,
                 min_figure_width=8,
                 width_scale_factor=1.2,
                 percentage=True,
                 y_lim_lower=None,
                 y_lim_upper=None):
        self.figure_height = figure_height
        self.bar_width = bar_width
        self.group_gap = group_gap
        self.annotation_digits = annotation_digits
        self.text_margin = text_margin
        self.min_figure_width = min_figure_width
        self.width_scale_factor = width_scale_factor
        self.percentage = percentage
        self.y_lim_lower = y_lim_lower
        self.y_lim_upper = y_lim_upper

        self.colors = ['#D4AFB9', '#A9B2C3', '#C3CBD5', '#EAE7DC', '#B8B8D1', '#A7C7E7', '#B5EAD7', '#FFDAC1', '#FF9AA2', '#C7CEEA','#71C9CE', '#A6E3E9', '#CBF1F5', '#FFE6E6', '#FFB6B9']
        self.rotation = 0 # rotation degree for x_label
        self.display_axes_borders = "all"  # "all", "xy" or "none", controls borders of the plot
        self.ncol_legend = 3

    def _calculate_ci_half_width(self, n, metric_val):
        # (Implementation remains the same)
        if not (0 <= metric_val <= 1): return 0.0
        if not isinstance(n, (int, float, np.integer, np.floating)) or n <= 0 or (isinstance(n, float) and not n.is_integer()): return 0.0
        n_int = int(n)
        z = 1.96
        if metric_val == 0 or metric_val == 1: return 0.0
        try:
            if n_int <= 0: return 0.0
            standard_error = math.sqrt(metric_val * (1 - metric_val) / n_int)
        except (ValueError, ZeroDivisionError): return 0.0
        return z * standard_error

    def _setup_figure_axes(self, n_elements, element_width, gap):
        # (Implementation remains the same)
        center_spacing = max(1.0, element_width + gap)
        total_x_span = center_spacing * (n_elements - 1) + element_width if n_elements > 1 else element_width
        dynamic_figure_width = max(self.min_figure_width, total_x_span * self.width_scale_factor + 2)
        fig, ax = plt.subplots(figsize=(dynamic_figure_width, self.figure_height))
        return fig, ax, center_spacing

    def _finalize_plot(self, fig, ax, title, x_label, y_label, x_ticks, x_tick_labels, y_lim_01, has_legend, save_path):
        # (Implementation remains the same)
        ax.set_xlabel(x_label)
        effective_y_label = y_label + " (%)" if self.percentage else y_label
        ax.set_ylabel(effective_y_label)
        ax.set_title(title)
        ax.set_xticks(x_ticks)
        if self.rotation:
            ax.set_xticklabels(x_tick_labels, rotation=self.rotation, ha="right")
        else: ax.set_xticklabels(x_tick_labels)
        ax.set_ylim(y_lim_01[0], y_lim_01[1])
        if self.percentage:
            scaled_upper_lim = y_lim_01[1] * 100
            tick_precision = 0 if scaled_upper_lim > 10 else 1
            formatter = mtick.FuncFormatter(lambda y, _: f'{y * 100:.{tick_precision}f}')
            ax.yaxis.set_major_formatter(formatter)
        if has_legend:
            ax.legend(bbox_to_anchor=(0.5, 1.05),  # 将图例放在轴的上方中央
              loc='lower center',         # 将图例的下边缘中心与 bbox_to_anchor 对齐
              borderaxespad=0.,
              ncol=self.ncol_legend)
            # ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)

        # axes border display
        if self.display_axes_borders not in ['all', "xy", 'none']:
            print('display_axes_borders has to be one of all, xy or none')
        if self.display_axes_borders == 'xy':
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        elif self.display_axes_borders == 'none':
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)


        try:
            right_margin = 0.85 if has_legend else 0.95
            fig.tight_layout(rect=[0.03, 0.03, right_margin, 0.95])
        except ValueError:
             print("Warning: tight_layout failed.")
             plt.subplots_adjust(left=0.1, right=0.8 if has_legend else 0.9, bottom=0.1, top=0.9)
        # Defer saving and showing to main plot methods

    def _annotate_bars_above(self, ax, bars, metric_values_01, errors_01=None):
        # (Implementation remains the same)
        scale_factor = 100.0 if self.percentage else 1.0
        errors_01 = errors_01 if errors_01 is not None else np.zeros_like(metric_values_01)
        for i, bar in enumerate(bars):
            height_01 = metric_values_01[i]
            error_01 = errors_01[i]
            annotation_y_pos_01 = height_01 + error_01 + self.text_margin
            display_value = height_01 * scale_factor
            annotation_text = f"{display_value:.{self.annotation_digits}f}"
            ax.text(bar.get_x() + bar.get_width() / 2.0, annotation_y_pos_01,
                    annotation_text, ha='center', va='bottom', fontsize=9)
            
    def _calculate_dynamic_y_limits(self, data_np, ci_half_np):
        """
        Calculates dynamic y-axis limits based on data and confidence intervals.

        Args:
            data_np (np.ndarray): Array of metric/value data (0-1 scale).
            ci_half_np (np.ndarray or None): Array of CI half-widths (0-1 scale), or None.

        Returns:
            tuple: (dynamic_lower_lim_01, dynamic_upper_lim_01) in 0-1 scale.
        """
        if data_np.size == 0:
            return (0.0, 1.0) # Default for empty data

        # Calculate min/max based on data and potential CIs
        min_val_01 = np.min(data_np)
        max_val_01 = np.max(data_np)
        max_err_01 = 0.0 # Max error needed for upper limit buffer

        if ci_half_np is not None and ci_half_np.size > 0:
            valid_ci = ~np.isnan(ci_half_np)
            if np.any(valid_ci):
                # Check min value including negative CI boundary
                min_val_with_ci_01 = np.min(data_np[valid_ci] - ci_half_np[valid_ci])
                min_val_01 = min(min_val_01, min_val_with_ci_01)

                # Check max value including positive CI boundary
                max_val_with_ci_01 = np.max(data_np[valid_ci] + ci_half_np[valid_ci])
                max_val_01 = max(max_val_01, max_val_with_ci_01)
                # Find max error relevant for annotation spacing
                max_err_01 = np.max(ci_half_np[valid_ci])

        # Calculate buffer based on the actual data range
        y_range_01 = max_val_01 - min_val_01
        # Ensure buffer is at least enough for text margin considerations
        y_buffer_01 = max(y_range_01 * 0.05, self.text_margin * 3)

        # Calculate dynamic limits including buffer
        dynamic_lower_lim_01 = min_val_01 - y_buffer_01
        # Prevent lower limit from going above 0 unless data warrants it
        if min_val_01 >= 0:
             dynamic_lower_lim_01 = max(0, dynamic_lower_lim_01)

        # Dynamic upper limit needs to accommodate highest annotation
        dynamic_upper_lim_01 = max_val_01 + max_err_01 + self.text_margin + y_buffer_01 * 0.5
        # Apply capping relative to 1.0
        dynamic_upper_lim_01 = min(dynamic_upper_lim_01, 1.0 + y_buffer_01)

        return (dynamic_lower_lim_01, dynamic_upper_lim_01)

    def group_barplot(self, group_names, item_names, metrics, n_samples=None, x_label='Group', y_label='Metric', title="", save_path=None):
        """
        Plots a grouped bar chart. Uses fixed y-limits if provided, else dynamic.
        """
        # --- Input Handling & Validation --- 
        try:
            metrics_np = np.array(metrics, dtype=float)
            n_samples_np = np.array(n_samples, dtype=float) if n_samples is not None else None
            group_names, item_names = list(group_names), list(item_names)
        except Exception as e: raise TypeError(f"Input conversion failed: {e}")
        n_items, n_groups = metrics_np.shape
        if n_items == 0 or n_groups == 0: raise ValueError("Inputs cannot be empty.")
        
        scale_factor = 100.0 if self.percentage else 1.0

        # --- Setup Figure & Axes --- 
        total_item_width = n_items * self.bar_width
        fig, ax, group_center_spacing = self._setup_figure_axes(n_groups, total_item_width, self.group_gap)
        group_centers = np.arange(n_groups) * group_center_spacing

        # --- Calculate CI --- 
        ci_half_np = None
        if n_samples_np is not None:
            ci_half_np = np.zeros_like(metrics_np)
            for i in range(n_items):
                for j in range(n_groups):
                    ci_half_np[i, j] = self._calculate_ci_half_width(n_samples_np[i, j], metrics_np[i, j])

        # --- Plotting --- 
        all_bars_flat, all_metrics_flat, all_errors_flat = [], [], []
        for i in range(n_items):
            offset = (i - (n_items - 1) / 2.0) * self.bar_width
            bar_positions = group_centers + offset
            yerr_values = ci_half_np[i, :] if ci_half_np is not None else None
            bars = ax.bar(bar_positions, metrics_np[i, :], width=self.bar_width, yerr=yerr_values,
                          capsize=5 if yerr_values is not None else 0, label=item_names[i],
                          error_kw={'elinewidth':1, 'capthick':1}, color = self.colors[i])
            all_bars_flat.extend(bars)
            all_metrics_flat.extend(metrics_np[i, :])
            all_errors_flat.extend(yerr_values if yerr_values is not None else [0] * n_groups)

        # --- Annotations --- 
        self._annotate_bars_above(ax, all_bars_flat, all_metrics_flat, all_errors_flat)

        # --- Calculate Dynamic Y-Limits using Helper --- <<< MODIFIED
        dynamic_lower_lim_01, dynamic_upper_lim_01 = self._calculate_dynamic_y_limits(metrics_np, ci_half_np)

        # --- Determine Final Y-Limits using user settings or dynamic values --- 
        final_lower_lim_01 = self.y_lim_lower / scale_factor if self.y_lim_lower is not None else dynamic_lower_lim_01
        final_upper_lim_01 = self.y_lim_upper / scale_factor if self.y_lim_upper is not None else dynamic_upper_lim_01
        final_y_lim_01 = (final_lower_lim_01, final_upper_lim_01)

        # --- Finalize --- (Same as before, but handle save/show here)
        self._finalize_plot(fig, ax, title, x_label, y_label, group_centers, group_names, final_y_lim_01, n_items > 1, None) # Pass None for save_path

        # --- Save / Show ---
        if save_path:
            if isinstance(save_path, str):
                try:
                    save_dir = os.path.dirname(save_path)
                    if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir)
                    fig.savefig(save_path, bbox_inches='tight', dpi=300)
                    print(f"Plot saved to: {save_path}")
                except Exception as e: print(f"Error saving plot: {e}")
            else: print(f"Warning: 'save_path' not a string.")

        plt.show()


    def barplot(self, bar_names, values, n_samples=None, bottom_annotations=None, x_label='Category', y_label='Value', title="", save_path=None):
        """
        Plots a simple bar chart. Uses fixed y-limits if provided, else dynamic.
        """
        # --- Input Handling & Validation --- 
        try:
            values_np = np.array(values, dtype=float)
            n_samples_np = np.array(n_samples, dtype=float) if n_samples is not None else None
            bottom_annotations = list(bottom_annotations) if bottom_annotations is not None else None
            bar_names = list(bar_names)
        except Exception as e: raise TypeError(f"Input conversion failed: {e}")
        n_bars = len(values_np)
        if n_bars == 0: raise ValueError("Inputs cannot be empty.")
        # ... (other validation checks omitted for brevity) ...
        scale_factor = 100.0 if self.percentage else 1.0

        # --- Setup Figure & Axes --- 
        fig, ax, bar_center_spacing = self._setup_figure_axes(n_bars, self.bar_width, self.group_gap)
        bar_centers = np.arange(n_bars) * bar_center_spacing

        # --- Calculate CI --- 
        ci_half_np = None
        if n_samples_np is not None:
            ci_half_np = np.zeros_like(values_np)
            for i in range(n_bars):
                ci_half_np[i] = self._calculate_ci_half_width(n_samples_np[i], values_np[i])

        # --- Plotting ---
        bars = ax.bar(bar_centers, values_np, width=self.bar_width, color=self.colors[:n_bars],
                      yerr=ci_half_np, capsize=5 if ci_half_np is not None else 0,
                      error_kw={'elinewidth':1, 'capthick':1})

        # --- Annotations Above Bars --- 
        self._annotate_bars_above(ax, bars, values_np, ci_half_np)

        # --- Calculate Dynamic Y-Limits using Helper --- 
        dynamic_lower_lim_01, dynamic_upper_lim_01 = self._calculate_dynamic_y_limits(values_np, ci_half_np)

        # --- Determine Final Y-Limits using user settings or dynamic values --- 
        final_lower_lim_01 = self.y_lim_lower / scale_factor if self.y_lim_lower is not None else dynamic_lower_lim_01
        final_upper_lim_01 = self.y_lim_upper / scale_factor if self.y_lim_upper is not None else dynamic_upper_lim_01
        final_y_lim_01 = (final_lower_lim_01, final_upper_lim_01)

        # --- Finalize (before bottom annotations) --- 
        self._finalize_plot(fig, ax, title, x_label, y_label, bar_centers, bar_names, final_y_lim_01, False, None)

        # --- Bottom Annotations (after initial layout) --- 
        if bottom_annotations:
            plt.subplots_adjust(bottom=0.15) # Increase bottom margin preemptively
            for i in range(n_bars):
                ax.text(bar_centers[i], -0.12, bottom_annotations[i],
                        transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=8, color='gray')

        # --- Save / Show ---
        if save_path:
            if isinstance(save_path, str):
                try:
                    save_dir = os.path.dirname(save_path)
                    if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir)
                    fig.savefig(save_path, bbox_inches='tight', dpi=300)
                    print(f"Plot saved to: {save_path}")
                except Exception as e: print(f"Error saving plot: {e}")
            else: print(f"Warning: 'save_path' not a string.")

        plt.show()


def plot_heatmap(data,
                            row_names=None,
                            col_names=None,
                            title="Heatmap",
                            xlabel="Columns",
                            ylabel="Rows",
                            cmap="viridis",
                            annot=True,
                            fmt=".2f",
                            linewidths=0.5,
                            linecolor='lightgray',
                            cbar=False,
                            figsize=(10, 8),
                            xtick_rotation=45,
                            ax=None,
                            vmin=None,  # Added vmin
                            vmax=None,  # Added vmax
                            **kwargs):
    """
    Generates an aesthetically attractive heatmap for m x n data.

    Args:
        data (array-like or pd.DataFrame): The m x n matrix data to plot.
        row_names (list-like, optional): Names for the rows (y-axis).
            Ignored if 'data' is a pandas DataFrame. Defaults to None.
        col_names (list-like, optional): Names for the columns (x-axis).
            Ignored if 'data' is a pandas DataFrame. Defaults to None.
        title (str, optional): Title for the plot. Defaults to "Heatmap".
        xlabel (str, optional): Label for the x-axis. Defaults to "Columns".
        ylabel (str, optional): Label for the y-axis. Defaults to "Rows".
        cmap (str or Colormap, optional): The colormap to use.
            Examples: 'viridis', 'plasma', 'inferno', 'magma', 'cividis' (sequential)
                      'coolwarm', 'bwr', 'RdBu_r' (diverging)
                      'YlGnBu', 'Blues', 'BuPu', 'Greens' (sequential multi-hue)
            Defaults to "viridis".
        annot (bool or array-like, optional): If True, write the data value in each cell.
            If an array-like object matching 'data' shape, plot those values instead.
            Defaults to True.
        fmt (str, optional): String formatting code to use when annot is True.
            Defaults to ".2f".
        linewidths (float, optional): Width of the lines that will divide each cell.
            Defaults to 0.5.
        linecolor (str, optional): Color of the lines that will divide each cell.
            Defaults to 'lightgray'.
        cbar (bool, optional): Whether to draw a color bar. Defaults to True.
        figsize (tuple, optional): Width, height in inches. Defaults to (10, 8).
            Adjust based on the matrix size for better readability.
        xtick_rotation (int or float, optional): Rotation angle for x-axis tick labels.
            Defaults to 45. Set to 0 or None for no rotation.
        ax (matplotlib.axes.Axes, optional): An existing Axes object to plot on.
            If None, a new figure and axes are created. Defaults to None.
        vmin (float, optional): The minimum value anchoring the colormap.
            Defaults to None.
        vmax (float, optional): The maximum value anchoring the colormap.
            Defaults to None.
        **kwargs: Additional keyword arguments passed directly to seaborn.heatmap().

    Returns:
        matplotlib.axes.Axes: The Axes object with the heatmap.
    """
    if isinstance(data, pd.DataFrame):
        plot_data = data
        _row_names = data.index.tolist() if row_names is None else row_names
        _col_names = data.columns.tolist() if col_names is None else col_names
    else:
        plot_data = pd.DataFrame(data, index=row_names, columns=col_names)
        _row_names = plot_data.index.tolist()
        _col_names = plot_data.columns.tolist()

    if ax is None:
        fig, current_ax = plt.subplots(figsize=figsize)
    else:
        current_ax = ax
        fig = current_ax.figure

    sns.heatmap(plot_data,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                linewidths=linewidths,
                linecolor=linecolor,
                cbar=cbar,
                ax=current_ax,
                xticklabels=True,
                yticklabels=True,
                vmin=vmin,  # Use provided vmin
                vmax=vmax,  # Use provided vmax
                **kwargs)

    current_ax.set_title(title, fontsize=14, fontweight='bold')
    current_ax.set_xlabel(xlabel, fontsize=12)
    current_ax.set_ylabel(ylabel, fontsize=12)

    current_ax.set_xticks(np.arange(len(_col_names)) + 0.5)
    current_ax.set_yticks(np.arange(len(_row_names)) + 0.5)
    current_ax.set_xticklabels(_col_names)
    current_ax.set_yticklabels(_row_names)

    if xtick_rotation is not None:
         plt.setp(current_ax.get_xticklabels(), rotation=xtick_rotation, ha="right",
                  rotation_mode="anchor")

    plt.setp(current_ax.get_yticklabels(), rotation=0)

    if ax is None:
         try:
             fig.tight_layout()
         except ValueError:
             print("Warning: tight_layout failed. Plot elements might overlap.")

    return current_ax
