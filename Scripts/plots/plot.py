# v3 compared to v2: processing input error bars for each bar

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import math
import os # Added for save_path directory creation
NA = np.nan # short for na values when plotting error bars in MetricPlotter

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
        x_label_fontsize (int): Font size for the x-axis label.
        y_label_fontsize (int): Font size for the y-axis label.
        x_tick_fontsize (int): Font size for the x-axis tick labels.
        y_tick_fontsize (int): Font size for the y-axis tick labels.
        title_fontsize (int): Font size for the plot title.
        legend_fontsize (int): Font size for the legend.
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
                 y_lim_upper=None,
                 # New font size parameters
                 x_label_fontsize=12,
                 y_label_fontsize=12,
                 x_tick_fontsize=10,
                 y_tick_fontsize=10,
                 title_fontsize=14,
                 legend_fontsize=10): # Added legend_fontsize
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
        self.x_axis_padding = 0.0 # change the bars position (for the first bar)

        # Store font size attributes
        self.bar_fontsize = 9
        self.x_label_fontsize = x_label_fontsize
        self.y_label_fontsize = y_label_fontsize
        self.x_tick_fontsize = x_tick_fontsize
        self.y_tick_fontsize = y_tick_fontsize
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize # Store legend_fontsize

        self.colors = ['#D4AFB9', '#A9B2C3', '#C3CBD5', '#EAE7DC', '#B8B8D1', '#A7C7E7', '#B5EAD7', '#FFDAC1', '#FF9AA2', '#C7CEEA','#71C9CE', '#A6E3E9', '#CBF1F5', '#FFE6E6', '#FFB6B9']
        self.rotation = 0 # rotation degree for x_label
        self.display_axes_borders = "all"  # "all", "xy" or "none", controls borders of the plot
        self.ncol_legend = 3

    def _build_yerr(self, err_low, err_high, shape):
        """
        Assemble a matplotlib-compatible yerr array (shape: 2 × …) or None.

        Parameters
        ----------
        err_low, err_high : array-like or None
            Lower / upper absolute errors, no conversion for percentage format
            允许 np.nan: ignore bars without error bar values
        shape : tuple
            Should match metrics's shape; for check
        """
        if err_low is None and err_high is None:
            return None, None                # when no value provided for a bar, it's processed with no error bars

        # if only one side's error bar is given, the other is defualt 0
        low = np.zeros(shape, dtype=float) if err_low is None else np.asarray(err_low, dtype=float)
        high = np.zeros(shape, dtype=float) if err_high is None else np.asarray(err_high, dtype=float)

        if low.shape != shape or high.shape != shape:
            raise ValueError("Error-bar arrays must match `metrics` shape.")

        # None would be transformed to np.nan
        low = np.where(np.equal(low, None), NA, low)
        high = np.where(np.equal(high, None), NA, high)                    

        # matplotlib bar()
        yerr = np.vstack([low, high])

        return yerr, high                    # return the upper part for annotation


    def _calculate_ci_half_width(self, n, metric_val):
        # calculate half width of CI based on Bernoulli distribution, not called in this study
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
        # 
        center_spacing = element_width + gap
        if center_spacing <= 0 and n_elements > 1:
            pass
        if n_elements > 1:
            total_x_span = center_spacing * (n_elements - 1) + element_width
        else:
            total_x_span = element_width
        total_x_span = max(total_x_span, element_width, self.bar_width)
        dynamic_figure_width = max(self.min_figure_width, total_x_span * self.width_scale_factor + 2)
        fig, ax = plt.subplots(figsize=(dynamic_figure_width, self.figure_height))
        return fig, ax, center_spacing

    def _finalize_plot(self, fig, ax, title, x_label, y_label, x_ticks, x_tick_labels, y_lim_01, has_legend, save_path):
        ax.set_xlabel(x_label, fontsize=self.x_label_fontsize) # Use attribute
        effective_y_label = y_label + " (%)" if self.percentage else y_label
        ax.set_ylabel(effective_y_label, fontsize=self.y_label_fontsize) # Use attribute
        ax.set_title(title, fontsize=self.title_fontsize) # Use attribute

        ax.set_xticks(x_ticks)
        if self.rotation:
            ax.set_xticklabels(x_tick_labels, rotation=self.rotation, ha="right", fontsize=self.x_tick_fontsize) # Use attribute
        else:
            ax.set_xticklabels(x_tick_labels, fontsize=self.x_tick_fontsize) # Use attribute

        ax.tick_params(axis='y', labelsize=self.y_tick_fontsize) # Use attribute for y-tick labels

        ax.set_ylim(y_lim_01[0], y_lim_01[1])
        if self.percentage:
            scaled_upper_lim = y_lim_01[1] * 100
            tick_precision = 0 if scaled_upper_lim > 10 else 1
            formatter = mtick.FuncFormatter(lambda y, _: f'{y * 100:.{tick_precision}f}')
            ax.yaxis.set_major_formatter(formatter)
            # Apply y_tick_fontsize again after formatter, just in case formatter resets it (unlikely but good practice)
            ax.tick_params(axis='y', labelsize=self.y_tick_fontsize)


        if has_legend:
            ax.legend(bbox_to_anchor=(0.5, 1.05),
                      loc='lower center',
                      borderaxespad=0.,
                      ncol=self.ncol_legend,
                      fontsize=self.legend_fontsize) # Use attribute

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


    def _annotate_bars_above(self, ax, bars, metric_values_01, errors_01=None):
        # 
        scale_factor = 100.0 if self.percentage else 1.0
        if errors_01 is not None and not isinstance(errors_01, np.ndarray):
            errors_01 = np.array(errors_01, dtype=float)
        if errors_01 is None or errors_01.size != len(metric_values_01):
            errors_01 = np.zeros_like(metric_values_01, dtype=float)

        for i, bar in enumerate(bars):
            height_01 = metric_values_01[i]
            error_01 = errors_01[i] if errors_01 is not None and i < len(errors_01) else 0
            annotation_y_pos_01 = height_01 + error_01 + self.text_margin
            display_value = height_01 * scale_factor
            annotation_text = f"{display_value:.{self.annotation_digits}f}"
            ax.text(bar.get_x() + bar.get_width() / 2.0, annotation_y_pos_01,
                    annotation_text, ha='center', va='bottom', fontsize=self.bar_fontsize) # Annotation font size is still 9

    def _calculate_dynamic_y_limits(self, data_np, ci_half_np):
        # 
        if data_np.size == 0:
            return (0.0, 1.0)
        min_val_01 = np.min(data_np)
        max_val_01 = np.max(data_np)
        max_err_01 = 0.0
        if ci_half_np is not None and ci_half_np.size > 0:
            valid_ci = ~np.isnan(ci_half_np)
            if np.any(valid_ci):
                min_val_with_ci_01 = np.min(data_np[valid_ci] - ci_half_np[valid_ci])
                min_val_01 = min(min_val_01, min_val_with_ci_01)
                max_val_with_ci_01 = np.max(data_np[valid_ci] + ci_half_np[valid_ci])
                max_val_01 = max(max_val_01, max_val_with_ci_01)
                max_err_01 = np.max(ci_half_np[valid_ci])
        y_range_01 = max_val_01 - min_val_01
        y_buffer_01 = max(y_range_01 * 0.05, self.text_margin * 3)
        dynamic_lower_lim_01 = min_val_01 - y_buffer_01
        if min_val_01 >= 0:
             dynamic_lower_lim_01 = max(0, dynamic_lower_lim_01)
        dynamic_upper_lim_01 = max_val_01 + max_err_01 + self.text_margin + y_buffer_01 * 0.5
        dynamic_upper_lim_01 = min(dynamic_upper_lim_01, 1.0 + y_buffer_01)
        return (dynamic_lower_lim_01, dynamic_upper_lim_01)

    def group_barplot(self, group_names, item_names, metrics, n_samples=None,err_low=None, err_high=None, x_label='Group', y_label='Metric', title="", save_path=None):
        # ---------- input conversion ----------
        try:
            metrics_np = np.array(metrics, dtype=float)
            n_samples_np = np.array(n_samples, dtype=float) if n_samples is not None else None
            group_names, item_names = list(group_names), list(item_names)
        except Exception as e: raise TypeError(f"Input conversion failed: {e}")

        # ---------- original sanity checks ----------
        if metrics_np.ndim != 2: raise ValueError("metrics must be a 2D array-like.")
        n_items, n_groups = metrics_np.shape
        if n_items == 0 or n_groups == 0: raise ValueError("Inputs cannot be empty.")
        if len(item_names) != n_items: raise ValueError("Length of item_names must match number of rows in metrics.")
        if len(group_names) != n_groups: raise ValueError("Length of group_names must match number of columns in metrics.")
        if n_samples_np is not None and n_samples_np.shape != metrics_np.shape:
            raise ValueError("Shape of n_samples must match shape of metrics.")

        # ---------- OPTIONAL asymmetric-error arrays ----------
        err_low_np  = (np.array(err_low,  dtype=float)
                       if err_low  is not None else None)
        err_high_np = (np.array(err_high, dtype=float)
                       if err_high is not None else None)
        if err_low_np  is not None and err_low_np.shape  != metrics_np.shape:
            raise ValueError("err_low must have same shape as metrics.")
        if err_high_np is not None and err_high_np.shape != metrics_np.shape:
            raise ValueError("err_high must have same shape as metrics.")
        
        scale_factor = 100.0 if self.percentage else 1.0
        # ---------- figure & axes ----------
        total_item_width = n_items * self.bar_width
        fig, ax, group_center_spacing = self._setup_figure_axes(n_groups, total_item_width, self.group_gap)
        group_centers = np.arange(n_groups) * group_center_spacing
        # ---------- containers for unified annotation ----------
        all_bars_flat     = []
        all_metrics_flat  = []
        all_errors_flat   = []     # store *upper* errors only

        # ---------- main drawing loop ----------
        for i in range(n_items):
            offset = (i - (n_items - 1) / 2.0) * self.bar_width

            for j in range(n_groups):                           # loop for each bar in an item
                x_pos = group_centers[j] + offset
                bar_val = metrics_np[i, j]

                # get upper/lower for a single bar
                low_err_val  = None if err_low_np  is None else err_low_np[i, j]   
                high_err_val = None if err_high_np is None else err_high_np[i, j]  

                # --------- decide if a bar has error bars ----------
                if ((low_err_val  is None or np.isnan(low_err_val)) and
                    (high_err_val is None or np.isnan(high_err_val))):
                    yerr_single = None
                    capsize_val = 0
                    upper_for_text = 0.0
                else:
                    low_err_val  = 0.0 if low_err_val  is None or np.isnan(low_err_val)  else float(low_err_val)
                    high_err_val = 0.0 if high_err_val is None or np.isnan(high_err_val) else float(high_err_val)
                    yerr_single = [[low_err_val], [high_err_val]]                      
                    capsize_val = 5
                    upper_for_text = high_err_val

                # --------- draw a single bar ----------
                bar_cont = ax.bar(x_pos, bar_val,
                                  width=self.bar_width,
                                  yerr=yerr_single,
                                  capsize=capsize_val,
                                  error_kw={'elinewidth': 1, 'capthick': 1},
                                  color=self.colors[i % len(self.colors)],
                                  label=item_names[i] if j == 0 else None)  

                all_bars_flat.extend(bar_cont)            
                all_metrics_flat.append(bar_val)
                all_errors_flat.append(upper_for_text)

        # ---------- annotate all bars ----------
        self._annotate_bars_above(
            ax,
            all_bars_flat,
            np.asarray(all_metrics_flat, dtype=float),
            np.asarray(all_errors_flat, dtype=float))
        
        # determine edge before/after the first/last bars
        first_group_left_edge = group_centers[0] - total_item_width / 2
        last_group_right_edge = group_centers[-1] + total_item_width / 2
        ax.set_xlim(
            first_group_left_edge - self.x_axis_padding,
            last_group_right_edge + self.x_axis_padding
        )

        # ---------- dynamic y-limits ----------
        err_upper_mat = np.array(all_errors_flat, dtype=float).reshape(metrics_np.shape)
        dyn_low_01, dyn_up_01 = self._calculate_dynamic_y_limits(
            metrics_np.flatten(), err_upper_mat.flatten())     
        final_low_01 = (self.y_lim_lower / scale_factor
                        if self.y_lim_lower is not None else dyn_low_01)
        final_up_01  = (self.y_lim_upper / scale_factor
                        if self.y_lim_upper  is not None else dyn_up_01)
        self._finalize_plot(fig, ax, title, x_label, y_label,
                            group_centers, group_names,
                            (final_low_01, final_up_01),
                            has_legend=(n_items > 1), save_path=None)
        
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


    def barplot(
        self,
        bar_names,
        values,
        n_samples=None,
        err_low=None,
        err_high=None,
        bottom_annotations=None,
        x_label='Category',
        y_label='Value',
        title="",
        save_path=None
    ):
        """
        Draw a simple bar-plot with optional asymmetric error bars.

        Key improvement
        ---------------
        The y–position of the value annotation is now computed so that it
        always clears the highest point of the error bar *and* its cap.  
        This avoids the annotation text overlapping the error bar even when
        the error bar is very short.
        """
        # ------------------------------------------------------------------
        # 0) --- Input coercion / validation --------------------------------
        # ------------------------------------------------------------------
        try:
            values_np         = np.asarray(values,       dtype=float)
            n_samples_np      = None if n_samples is None else np.asarray(n_samples, dtype=float)
            bottom_annotations = None if bottom_annotations is None else list(bottom_annotations)
            bar_names          = list(bar_names)
        except Exception as e:
            raise TypeError(f"Input conversion failed: {e}")

        n_bars = len(values_np)
        if n_bars == 0:
            raise ValueError("Inputs cannot be empty.")
        if len(bar_names) != n_bars:
            raise ValueError("Length of bar_names must match length of values.")
        if n_samples_np is not None and len(n_samples_np) != n_bars:
            raise ValueError("Length of n_samples must match length of values.")
        if bottom_annotations is not None and len(bottom_annotations) != n_bars:
            raise ValueError("Length of bottom_annotations must match length of values.")

        scale_factor = 100.0 if self.percentage else 1.0  # 0-1 to percentage if requested

        # ------------------------------------------------------------------
        # 1) --- Figure / axis bootstrap ------------------------------------
        # ------------------------------------------------------------------
        fig, ax, bar_center_spacing = self._setup_figure_axes(
            n_bars, self.bar_width, self.group_gap
        )
        bar_centers = np.arange(n_bars) * bar_center_spacing

        # ------------------------------------------------------------------
        # 2) --- Assemble asymmetric y-error matrix -------------------------
        # ------------------------------------------------------------------
        yerr_mat, err_upper = self._build_yerr(err_low, err_high, values_np.shape)

        if yerr_mat is None:      # fall back to symmetric (binomial) CI if no manual error bars
            if n_samples_np is not None:
                ci_half_np = np.fromiter(
                    (self._calculate_ci_half_width(int(n), v) for n, v in zip(n_samples_np, values_np)),
                    dtype=float,
                    count=n_bars,
                )
                yerr_mat  = np.vstack([ci_half_np, ci_half_np])
                err_upper = ci_half_np
            else:
                err_upper = np.zeros_like(values_np)

        # ------------------------------------------------------------------
        # 3) --- Draw bars + error bars ------------------------------------
        # ------------------------------------------------------------------
        bars = ax.bar(
            bar_centers,
            values_np,
            width=self.bar_width,
            yerr=yerr_mat,
            capsize=5 if yerr_mat is not None else 0,
            error_kw={'elinewidth': 1, 'capthick': 1},
            color=[self.colors[i % len(self.colors)] for i in range(n_bars)],
        )

        # ------------------------------------------------------------------
        # 4) --- Y-limit calculation (needs to include annotation padding) --
        # ------------------------------------------------------------------
        # Dynamic limits based on data
        dyn_low, dyn_up = self._calculate_dynamic_y_limits(values_np, err_upper)

        # Extra head-room so the annotation text never touches the error-bar cap
        # We allocate 1.5 % of the axis span (in data units) OR 3 × text_margin,
        # whichever is larger, as an additional vertical buffer.
        extra_headroom = max((dyn_up - dyn_low) * 0.015, self.text_margin * 3)

        # Make sure the very highest annotation is below the final top limit
        highest_annotation_y = np.max(values_np + err_upper + extra_headroom)
        dyn_up = max(dyn_up, highest_annotation_y + self.text_margin)

        # Honour user-provided fixed limits if any
        final_low = (self.y_lim_lower / scale_factor) if self.y_lim_lower is not None else dyn_low
        final_up  = (self.y_lim_upper / scale_factor) if self.y_lim_upper is not None else dyn_up
        final_y_lim_01 = (final_low, final_up)

        # ------------------------------------------------------------------
        # 5) --- Finalise basic styling (labels, ticks, legend, etc.) -------
        # ------------------------------------------------------------------
        self._finalize_plot(
            fig, ax,
            title, x_label, y_label,
            bar_centers, bar_names,
            final_y_lim_01,
            has_legend=False,
            save_path=None,      # saving handled later
        )

        # ------------------------------------------------------------------
        # 6) --- Add numeric annotations *after* y-limits are set ----------
        # ------------------------------------------------------------------
        scale = 100.0 if self.percentage else 1.0
        for i, bar in enumerate(bars):
            # Height of this bar in data units
            height_01   = values_np[i]
            error_01    = err_upper[i]
            # Dynamic offset guarantees clearance over the cap + a little extra
            annotation_y = height_01 + error_01 + extra_headroom
            display_val  = height_01 * scale
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                annotation_y,
                f"{display_val:.{self.annotation_digits}f}",
                ha='center', va='bottom',
                fontsize=self.bar_fontsize,
            )

        # ------------------------------------------------------------------
        # 7) --- Optional bottom annotations (e.g. sample size) ------------
        # ------------------------------------------------------------------
        if bottom_annotations:
            # Leave space under the x-axis if needed
            fig.canvas.draw()  # ensure axis is realised
            plt.subplots_adjust(
                bottom=0.2 if len(max(bottom_annotations, key=len)) > 10 else 0.15
            )
            for i, txt in enumerate(bottom_annotations):
                ax.text(
                    bar_centers[i],
                    -0.12,                       # below x-axis in axis-fraction coords
                    txt,
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top',
                    fontsize=8, color='gray'
                )

        # ------------------------------------------------------------------
        # 8) --- Persist to disk if requested ------------------------------
        # ------------------------------------------------------------------
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


import matplotlib.colors as mcolors
import matplotlib.patches as mpatches # needed for handles
from typing import List, Union, Tuple, Optional

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by mixing it with white.

    Args:
        color: An matplotlib-compatible color string (e.g., 'red', '#FF0000', (1, 0, 0)).
        amount: The factor to lighten by. 0 means no change, 1 means white.

    Returns:
        A lightened color in RGB tuple format.
    """
    try:
        c = mcolors.to_rgb(color)
        c_white = mcolors.to_rgb('white')
        # Linear interpolation towards white
        return tuple(c_val + (white_val - c_val) * amount for c_val, white_val in zip(c, c_white))
    except ValueError:
        print(f"Warning: Could not parse color '{color}'. Using gray.")
        try:
            c_gray = mcolors.to_rgb('gray')
            c_white = mcolors.to_rgb('white')
            return tuple(c_val + (white_val - c_val) * amount for c_val, white_val in zip(c_gray, c_white))
        except ValueError:
            return (0.7, 0.7, 0.7) # Absolute fallback

def plot_model_performance(
    model_names: List[str],
    scores: List[Union[float, Tuple[float, float]]],
    colors: Optional[List[str]] = None,
    title: str = "Model Performance Comparison",
    ylabel: str = "Score",
    display_border: bool = True,
    lighten_factor: float = 0.4,
    figsize: Tuple[int, int] = (10, 6),
    text_offset_factor: float = 0.015,
    # --- Labels for legend ---
    # Tuple for dual scores (base, incremental), String for single score
    legend_labels: Union[Tuple[str, str], str] = ('Base', 'Incremental'),
    single_score_label_override: Optional[str] = None # New: Specify label for single scores explicitly
):
    """
    Generates a bar plot comparing model performance, creating explicit legend
    entries for each bar component.

    Args:
        model_names: A list of strings representing the names of the models.
        scores: A list containing the performance scores. Each element can be:
                - A single float (for models with one score).
                - A tuple of two floats (low, high) for models with a range.
        colors: An optional list of matplotlib-compatible color strings for each bar.
                If None, default colors will be used. Must match the length of model_names.
        title: The title of the plot.
        ylabel: The label for the y-axis.
        display_border: If False, hides the top and right plot borders (spines).
                        If True (default), shows all borders.
        lighten_factor: The factor used to lighten the color for the 'high' part
                        of the bar when two scores are given (0=no lighten, 1=white).
        figsize: The size of the figure (width, height) in inches.
        text_offset_factor: A factor (relative to max score) to offset text labels
                            vertically above the bars for better readability.
        legend_labels: Describes the legend text.
                       - If a tuple (e.g., ('Reasoning', 'Completion')), the first
                         string labels the base part of dual scores, the second labels
                         the incremental part.
                       - If a string (e.g., 'Score'), it labels single-score bars.
                         If dual scores exist, the second label from the default
                         ('Base', 'Incremental') or a sensible guess might be used
                         for the incremental part unless overridden.
        single_score_label_override: Explicitly set the legend label for single-score
                                     bars. If None, uses the rules described for
                                     `legend_labels`. Useful when single scores represent
                                     the same concept as the *incremental* part of dual scores.
    """
    num_models = len(model_names)
    if len(scores) != num_models:
        raise ValueError("Length of 'model_names' and 'scores' must be the same.")

    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']
        colors = [default_colors[i % len(default_colors)] for i in range(num_models)]
    elif len(colors) != num_models:
        raise ValueError("Length of 'colors' must match 'model_names' if provided.")

    # Determine legend labels based on input
    if isinstance(legend_labels, tuple) and len(legend_labels) == 2:
        base_label, incremental_label = legend_labels
        single_label = single_score_label_override if single_score_label_override is not None else incremental_label # Default single to incremental if override not set
    elif isinstance(legend_labels, str):
        base_label = "Base" # Default if only single label provided
        incremental_label = "Incremental" # Default if only single label provided
        single_label = single_score_label_override if single_score_label_override is not None else legend_labels
    else:
        raise ValueError("legend_labels must be a tuple of two strings or a single string.")


    x_pos = np.arange(num_models)
    max_score = 0
    legend_handles = [] # List to store patch handles for the legend
    legend_texts = [] # List to store text labels for the legend

    plt.figure(figsize=figsize)
    ax = plt.gca()

    for i in range(num_models):
        score = scores[i]
        color = colors[i]

        # Determine max_score for plot limits and text offset
        current_max = 0
        if isinstance(score, (tuple, list)):
             current_max = score[1] if len(score) > 1 else (score[0] if len(score) > 0 else 0)
        else:
             current_max = score
        # Ensure current_max is a number before comparison
        if isinstance(current_max, (int, float)) and current_max > max_score:
             max_score = current_max

        text_offset = max_score * text_offset_factor if max_score > 0 else 0.01

        if isinstance(score, (tuple, list)) and len(score) == 2:
            low_score, high_score = score
            if low_score > high_score:
                 print(f"Warning: Model '{model_names[i]}', low score ({low_score}) > high score ({high_score}). Swapping.")
                 low_score, high_score = high_score, low_score

            if low_score < 0 or high_score < 0:
                print(f"Warning: Model '{model_names[i]}' has negative scores.")

            # Plot base bar - NO automatic label
            bar1 = ax.bar(x_pos[i], low_score, color=color)
            # Add handle and label for legend manually
            legend_handles.append(mpatches.Patch(color=color))
            legend_texts.append(base_label)

            # Plot incremental bar - NO automatic label
            if high_score > low_score:
                lighter_c = lighten_color(color, amount=lighten_factor)
                bar2 = ax.bar(x_pos[i], high_score - low_score, bottom=low_score, color=lighter_c)
                # Add handle and label for legend manually
                legend_handles.append(mpatches.Patch(color=lighter_c))
                legend_texts.append(incremental_label)

            # Add text labels for scores
            if low_score > 0:
                ax.text(x_pos[i], low_score + text_offset, f"{low_score:.2f}", ha='center', va='bottom', fontsize=9)
            ax.text(x_pos[i], high_score + text_offset, f"{high_score:.2f}", ha='center', va='bottom', fontsize=9)

        else:
            # Single score provided
            single_score = float(score)
            if single_score < 0:
                 print(f"Warning: Model '{model_names[i]}' has negative score.")

            # Plot single bar - NO automatic label
            bar_single = ax.bar(x_pos[i], single_score, color=color)
            # Add handle and label for legend manually
            legend_handles.append(mpatches.Patch(color=color))
            legend_texts.append(single_label)

            # Add text label
            ax.text(x_pos[i], single_score + text_offset, f"{single_score:.2f}", ha='center', va='bottom', fontsize=9)

    # --- Plot Customization ---
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=0)

    ax.set_ylim(bottom=0, top=max_score * 1.15)

    if not display_border:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # --- Add Explicit Legend ---
    if legend_handles: # Only add legend if there's something to show
        ax.legend(handles=legend_handles, labels=legend_texts,
                  bbox_to_anchor=(0.5, 1.05),  # 将图例放在轴的上方中央
              loc='lower center',         # 将图例的下边缘中心与 bbox_to_anchor 对齐
              borderaxespad=0.,
              ncol=3)

    # Optional grid
    # ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

