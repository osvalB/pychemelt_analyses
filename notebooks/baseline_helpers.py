import string
from pychemelt import Monomer
import numpy as np
import matplotlib.pyplot as plt

def model(x, a, b, c=None, kind=None):
    """
    Encoding functions for baseline fitting
    """
    if kind == "linear":
        return a + x * b
    elif kind == "quadratic":
        return a + b*x + c*(x**2)
    elif kind == "exponential":
        return a + b * np.exp(-c * x)
    else:
        raise ValueError("Unknown model type")
    

def rmse(x1, x2):
    """
    Implementation of RMSE
    """
    return np.sqrt(np.mean((x1 - x2)**2))


#okabe ito colorblind friendly colorscheme
okabe_ito = {
    "black":       "#000000",
    "orange":      "#E69F00",
    "skyblue":     "#56B4E9",
    "bluishgreen": "#009E73",
    "yellow":      "#F0E442",
    "blue":        "#0072B2",
    "vermillion":  "#D55E00",
    "reddishpurple": "#CC79A7",
}



def folded_comp(conditions, filename, ws=12, max_temp=50, signals=["330nm", "350nm"]): 
    """
    Compare quadratic and exponential folded baseline fits for DSF data with folded curves.

    This function loads DSF fluorescence data using the pychemelts `Sample`
    interface, estimates folded-state baselines using both quadratic and
    exponential models by fitting the curves to the data of the first ws °C, 
    compares their performance using RMSE, and visualizes
    the results for two fluorescence signals (330 nm and 350 nm).

    Parameters
    ----------
    conditions : list of bool
        Boolean mask specifying which experimental conditions are selected
        for analysis. Length must match the number of experiments in the input
        dataset. Typically, `True` values correspond to curves without an
        inflection point used for folded baseline estimation.

    filename : str
        Filename of the file containing the DSF data to be used to evaluate 
        the fitting of the folded baseline to the folded data.

    ws : int, optional, Default 12
        Window size (in °C) used for fitting the folded baseline.
        This parameter defines the temperature range over which the baseline
        parameters are estimated. Default is 12.

    Returns
    -------
    None

    Notes
    -----
    - RMSE is computed between the measured fluorescence signal and the fitted
      baseline for each denaturant concentration and signal.
    - Temperatures are internally converted to Kelvin and referenced to 298 K
      before model evaluation.
    - A shared legend and figure-level title are used for visualization.

    Examples
    --------
    >>> conditions = (
    ...     [False] * 18
    ...     + [True] * 3
    ...     + [False] * (48 - 21)
    ... )
    >>> folded_comp(conditions, ws=12)
    """
        
    #using pychemelts sample structure for the loading of the data and prediction

    sample = Monomer()
    sample.read_multiple_files(filename)
    sample.set_denaturant_concentrations()

    #usage of 330nm and 350nm for the initial curves
    sample.set_signal(signals)
    
    #experiments used for evaluation, should be curves without inflection 
    sample.select_conditions(conditions)    

    sample.set_temperature_range(0, max_temp)

    #fit the baselines as linear curve
    sample.estimate_baseline_parameters(
        native_baseline_type="linear",
        unfolded_baseline_type="constant", # not used
        window_range_unfolded=ws
    )

    a_lins = sample.first_param_Ns_per_signal
    b_lins = sample.second_param_Ns_per_signal
    c_lins = sample.third_param_Ns_per_signal

    #fit the baselines as exponential curve
    sample.estimate_baseline_parameters(
        native_baseline_type="exponential",
        unfolded_baseline_type="constant",  # not used
        window_range_unfolded=ws
    )

    a_exps = sample.first_param_Ns_per_signal
    b_exps = sample.second_param_Ns_per_signal
    c_exps = sample.third_param_Ns_per_signal

    #assigning preprocessed signal and temperature for plotting
    temperature = sample.temp_lst_multiple[0][0]     
    fluorescences = sample.signal_lst_multiple  

    #using the removed reference temperature for plotting
    temperature_K = temperature + 273.15
    temperature_K_ref = temperature_K - 298

    #creating all formula results for comparison and plots

    fig, axes = plt.subplots(
    ncols=2,
    nrows=sample.nr_den,
    figsize=(10, 5 * sample.nr_den),
    sharex=True,
    )

    T_window_start = temperature.min() + ws

    for i in range(2):    # 330nm and 350nm

        for j in range(sample.nr_den):
            
            ax = axes[j, i]

            lin_bl = model(temperature_K_ref,
                a_lins[i][j],
                b_lins[i][j],
                kind="linear")

            exp_bl = model(temperature_K_ref,
                a_exps[i][j],
                b_exps[i][j],
                c_exps[i][j],
                kind="exponential")


            #plotting the results
            ax.scatter(
                temperature,
                fluorescences[i][j],
                s=4,
                color=okabe_ito["black"],
                alpha=0.7,
            )

            ax.plot(
                temperature,
                lin_bl,
                color=okabe_ito["blue"],
                alpha=0.6,
                label=f"Linear fit (RMSE={rmse(fluorescences[i][j], lin_bl):.2})",
            )

            ax.plot(
                temperature,
                exp_bl,
                color=okabe_ito["orange"],
                linestyle="--",
                alpha=0.6,
                label=f"Exponential fit (RMSE={rmse(fluorescences[i][j], exp_bl):.2})",
            )

            ax.axvline(
                T_window_start,
                color=okabe_ito["vermillion"],
                linestyle=":",
                linewidth=2,
            )


            if j == 0:
                ax.set_title(["330 nm", "350 nm"][i])

            if i == 0:
                ax.set_ylabel("Fluorescence")

            ax.legend()
            ax.grid(True)

        
        ax.set_xlabel("Temperature (°C)")


    # manual legend (once)
    handles = [
        plt.Line2D([0], [0], color=okabe_ito["black"], label="Signal"),
        plt.Line2D([0], [0], color=okabe_ito["blue"], label="Linear baseline"),
        plt.Line2D([0], [0], color=okabe_ito["orange"], linestyle="--", label="Exponential baseline"),
        plt.Line2D([0], [0], color=okabe_ito["vermillion"], linestyle=":", label="Unfolded fit window start"),
    ]

    fig.suptitle("Baseline curve fitting window size " + str(ws) + "°", fontsize=20)
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.93))
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

    return None

def folded_comp_fig(condition, filename, windows=[12],max_temp=40,signals=["330nm", "350nm"],suptitle='ACBP'): 
    """
    Compare quadratic and exponential folded baseline fits for DSF data with a folded curve.

    This function loads DSF fluorescence data using the pychemelts `Sample`
    interface, estimates folded-state baselines using both quadratic and
    exponential models by fitting the curves to the data of the last °C of 
    different window sizes given in windows, 
    compares their performance using RMSE, and visualizes
    the results for two fluorescence signals (330 nm and 350 nm).

    Parameters
    ----------
    condition : list of bool
        Boolean mask specifying which experimental condition is selected
        for analysis. Length must match the number of experiments in the input
        dataset. The one selected curve should represent a fully folded curve

    filename : str
        Filename of the file containing the DSF data to be used to evaluate 
        the fitting of the folded baseline to the folded data.
    
    windows : int, optional, Default [12]
        Window sizes (in °C) used for fitting the folded baseline.
        This parameter defines the temperature range over which the baseline
        parameters are estimated. Default is [12].

    Returns
    -------
    fig: object
        The matplotlib figure object

    Notes
    -----
    - RMSE is computed between the measured fluorescence signal and the fitted
      baseline for each denaturant concentration and signal.
    - Temperatures are internally converted to Kelvin and referenced to 298 K
      before model evaluation.
    - A shared legend and figure-level title are used for visualization.


    """

    if not isinstance(windows, list):
        windows = [windows]

    fig, axes = plt.subplots(
    ncols=len(windows),
    nrows=2,
    figsize=(4 * len(windows), 8),
    sharex=True,
    )

    for j, ws in enumerate(windows):

        #using pychemelts sample structure for the loading of the data and prediction

        sample = Monomer()
        sample.read_multiple_files(filename)
        sample.set_denaturant_concentrations()
            #usage of 330nm and 350nm for the initial curves
        sample.set_signal(signals)
        
        #experiments used for evaluation, should be curves without inflection 
        sample.select_conditions(condition)    
        sample.set_temperature_range(0, max_temp)

        #fit the baselines as linear curve
        sample.estimate_baseline_parameters(
            native_baseline_type="linear",
            unfolded_baseline_type="constant", # not used
            window_range_native=ws
        )

        a_lins = sample.first_param_Ns_per_signal
        b_lins = sample.second_param_Ns_per_signal

        #fit the baselines as exponential curve
        sample.estimate_baseline_parameters(
            native_baseline_type="exponential",
            unfolded_baseline_type="constant", # not used
            window_range_native=ws
        )

        a_exps = sample.first_param_Ns_per_signal
        b_exps = sample.second_param_Ns_per_signal
        c_exps = sample.third_param_Ns_per_signal

        #assigning preprocessed signal and temperature for plotting
        temperature = sample.temp_lst_multiple[0][0]     
        fluorescences = sample.signal_lst_multiple  

    
        #using the removed reference temperature for plotting
        temperature_K = temperature + 273.15
        temperature_K_ref = temperature_K - 298

        #creating all formula results for comparison and plots

        T_window_start = temperature.min() + ws

        for i in range(2):    # 330nm and 350nm

            if len(windows) == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]

            lin_bl = model(temperature_K_ref,
                a_lins[i][0],
                b_lins[i][0],
                kind="linear")


            exp_bl = model(temperature_K_ref,
                a_exps[i][0],
                b_exps[i][0],
                c_exps[i][0],
                kind="exponential")

            #plotting the results
            ax.scatter(
                temperature,
                fluorescences[i][0],
                s=4,
                color=okabe_ito["black"],
                alpha=0.7,
            )

            ax.plot(
                temperature,
                lin_bl,
                color=okabe_ito["blue"],
                alpha=0.8,
                label=f"Linear fit (RMSE={rmse(fluorescences[i][0], lin_bl):.2})",
            )

            ax.plot(
                temperature,
                exp_bl,
                color=okabe_ito["orange"],
                linestyle="--",
                alpha=0.8,
                label=f"Exponential fit (RMSE={rmse(fluorescences[i][0], exp_bl):.2})",
            )

            ax.axvline(
                T_window_start,
                color=okabe_ito["vermillion"],
                linestyle=":",
                linewidth=2,
            )

            ax.text(
                T_window_start+5,
                ax.get_ylim()[0],                
                f"{int(T_window_start)} °C",
                color="black",
                ha="right",
                va="bottom",
                fontsize=10,
            )           

            if i == 0:
                ax.set_title(f"Fitting Window size of {ws}°", fontsize=14)

            # panel label
            if len(windows) == 1:
                label_idx = i
            else:
                label_idx = i * axes.shape[1] + j


            ax.set_title(f"{string.ascii_uppercase[label_idx]})", loc="left", fontsize=14,
                fontweight="bold",)


            if j == 0:
                ax.set_ylabel("Fluorescence")

            ax.legend(loc='upper right')
            #ax.grid(True)

            if i == 1:
                ax.set_xlabel("Temperature (°C)")


        # manual legend (once)
        handles = [
            plt.Line2D([0], [0], color=okabe_ito["black"], linestyle='None', marker='o', markersize=4, label="Signal"),
            plt.Line2D([0], [0], color=okabe_ito["blue"], label="Linear baseline"),
            plt.Line2D([0], [0], color=okabe_ito["orange"], linestyle="--", label="Exponential baseline"),
            plt.Line2D([0], [0], color=okabe_ito["vermillion"], linestyle=":", label="Folded fit window end"),
        ]

    ws_str = [str(x)for x in windows]

    fig.suptitle(f"{suptitle} - Folded baseline curve fitting with varying window sizes: " + "°, ".join(ws_str) + "°", fontsize=18)
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95))
    
    # Row labels for signals
    row_labels = [signal.replace("nm", " nm") for signal in signals]

    # Y positions are in figure coordinates (0–1)
    for i, label in enumerate(row_labels):
        fig.text(
            0.0,                       # x position (left margin)
            0.675 - i * 0.425,             # y position (top to bottom)
            label,
            va="center",
            ha="left",
            rotation=90,
            fontsize=14,
            fontweight="bold",
        )
    
    fig.tight_layout(rect=[0.01, 0, 1, 0.95])
    
    return fig

def unfolded_comp(conditions, filename, ws=12,signals=["330nm", "350nm"]): 
    """
    Compare quadratic and exponential unfolded baseline fits for DSF data with unfolded curves.

    This function loads DSF fluorescence data using the pychemelts `Sample`
    interface, estimates unfolded-state baselines using both quadratic and
    exponential models by fitting the curves to the data of the last ws °C, 
    compares their performance using RMSE, and visualizes
    the results for two fluorescence signals (330 nm and 350 nm).

    Parameters
    ----------
    conditions : list of bool
        Boolean mask specifying which experimental conditions are selected
        for analysis. Length must match the number of experiments in the input
        dataset. Typically, `True` values correspond to curves without an
        inflection point used for unfolded baseline estimation.

    filename : str
        Filename of the file containing the DSF data to be used to evaluate 
        the fitting of the unfolded baseline to the unfolded data.

    ws : int, optional, Default 12
        Window size (in °C) used for fitting the unfolded baseline.
        This parameter defines the temperature range over which the baseline
        parameters are estimated. Default is 12.

    Returns
    -------
    None

    Notes
    -----
    - RMSE is computed between the measured fluorescence signal and the fitted
      baseline for each denaturant concentration and signal.
    - Temperatures are internally converted to Kelvin and referenced to 298 K
      before model evaluation.
    - A shared legend and figure-level title are used for visualization.

    Examples
    --------
    >>> conditions = (
    ...     [False] * 18
    ...     + [True] * 3
    ...     + [False] * (48 - 21)
    ... )
    >>> unfolded_comp(conditions, ws=12)
    """
        
    #using pychemelts sample structure for the loading of the data and prediction

    sample = Monomer()
    sample.read_multiple_files(filename)
    sample.set_denaturant_concentrations()

    #usage of 330nm and 350nm for the initial curves
    sample.set_signal(signals)
    
    #experiments used for evaluation, should be curves without inflection 
    sample.select_conditions(conditions)    

    #fit the baselines as quadratic curve
    sample.estimate_baseline_parameters(
        native_baseline_type="linear",
        unfolded_baseline_type="quadratic",
        window_range_unfolded=ws
    )

    a_quas = sample.first_param_Us_per_signal
    b_quas = sample.second_param_Us_per_signal
    c_quas = sample.third_param_Us_per_signal

    #fit the baselines as exponential curve
    sample.estimate_baseline_parameters(
        native_baseline_type="linear",
        unfolded_baseline_type="exponential",
        window_range_unfolded=ws
    )

    a_exps = sample.first_param_Us_per_signal
    b_exps = sample.second_param_Us_per_signal
    c_exps = sample.third_param_Us_per_signal

    #assigning preprocessed signal and temperature for plotting
    temperature = sample.temp_lst_multiple[0][0]     
    fluorescences = sample.signal_lst_multiple  

    
    #using the removed reference temperature for plotting
    temperature_K = temperature + 273.15
    temperature_K_ref = temperature_K - 298

    #creating all formula results for comparison and plots

    fig, axes = plt.subplots(
    ncols=2,
    nrows=sample.nr_den,
    figsize=(10, 5 * sample.nr_den),
    sharex=True,
    )

    T_window_start = temperature.max() - ws

    for i in range(2):    # 330nm and 350nm

        for j in range(sample.nr_den):
            
            ax = axes[j, i]

            qua_bl = model(temperature_K_ref,
                a_quas[i][j],
                b_quas[i][j],
                c_quas[i][j],
                kind="quadratic")


            exp_bl = model(temperature_K_ref,
                a_exps[i][j],
                b_exps[i][j],
                c_exps[i][j],
                kind="exponential")


            #plotting the results
            ax.scatter(
                temperature,
                fluorescences[i][j],
                s=4,
                color=okabe_ito["black"],
                alpha=0.7,
            )

            ax.plot(
                temperature,
                qua_bl,
                color=okabe_ito["blue"],
                alpha=0.6,
                label=f"Quadratic fit (RMSE={rmse(fluorescences[i][j], qua_bl):.2})",
            )

            ax.plot(
                temperature,
                exp_bl,
                color=okabe_ito["orange"],
                linestyle="--",
                alpha=0.6,
                label=f"Exponential fit (RMSE={rmse(fluorescences[i][j], exp_bl):.2})",
            )

            ax.axvline(
                T_window_start,
                color=okabe_ito["vermillion"],
                linestyle=":",
                linewidth=2,
            )


            if j == 0:
                ax.set_title(["330 nm", "350 nm"][i])

            if i == 0:
                ax.set_ylabel("Fluorescence")

            ax.legend()
            ax.grid(True)

        
        ax.set_xlabel("Temperature (°C)")

    # manual legend (once)
    handles = [
        plt.Line2D([0], [0], color=okabe_ito["black"], label="Signal"),
        plt.Line2D([0], [0], color=okabe_ito["blue"], label="Quadratic baseline"),
        plt.Line2D([0], [0], color=okabe_ito["orange"], linestyle="--", label="Exponential baseline"),
        plt.Line2D([0], [0], color=okabe_ito["vermillion"], linestyle=":", label="Unfolded fit window start"),
    ]

    fig.suptitle("Baseline curve fitting window size " + str(ws) + "°", fontsize=20)
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.93))
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

    return fig

def unfolded_comp_fig(condition, filename, windows=[12],subtitle='ACBP',signals=["330nm", "350nm"]): 
    """
    Compare quadratic and exponential unfolded baseline fits for DSF data with an unfolded curve.

    This function loads DSF fluorescence data using the pychemelts `Sample`
    interface, estimates unfolded-state baselines using both quadratic and
    exponential models by fitting the curves to the data of the last °C of 
    different window sizes given in windows, 
    compares their performance using RMSE, and visualizes
    the results for two fluorescence signals (330 nm and 350 nm).

    Parameters
    ----------
    condition : list of bool
        Boolean mask specifying which experimental condition is selected
        for analysis. Length must match the number of experiments in the input
        dataset. The one selected curve should represent a fully unfolded curve

    filename : str or Monomer object
        Filename of the file containing the DSF data to be used to evaluate 
        the fitting of the unfolded baseline to the unfolded data. If a Monomer
        object is provided, it will be used directly without reading from a file.
    
    windows : int, optional, Default [12]
        Window sizes (in °C) used for fitting the unfolded baseline.
        This parameter defines the temperature range over which the baseline
        parameters are estimated. Default is [12].

    Returns
    -------
    fig: object
        The matplotlib figure object

    Notes
    -----
    - RMSE is computed between the measured fluorescence signal and the fitted
      baseline for each denaturant concentration and signal.
    - Temperatures are internally converted to Kelvin and referenced to 298 K
      before model evaluation.
    - A shared legend and figure-level title are used for visualization.

    """

    if not isinstance(windows, list):
        windows = [windows]

    fig, axes = plt.subplots(
    ncols=len(windows),
    nrows=2,
    figsize=(4 * len(windows), 8),
    sharex=True,
    )

    for j, ws in enumerate(windows):

        #using pychemelts sample structure for the loading of the data and prediction

        if not isinstance(filename, Monomer):

            sample = Monomer()
            sample.read_multiple_files(filename)
            sample.set_denaturant_concentrations()

            #usage of 330nm and 350nm for the initial curves
            sample.set_signal(signals)
            
            #experiments used for evaluation, should be curves without inflection 
            sample.select_conditions(condition)    

        else:

            # Monomer object is provided directly, use it without reading from a file
            sample = filename

        #fit the baselines as quadratic curve
        sample.estimate_baseline_parameters(
            native_baseline_type="linear",
            unfolded_baseline_type="quadratic",
            window_range_unfolded=ws
        )

        a_quas = sample.first_param_Us_per_signal
        b_quas = sample.second_param_Us_per_signal
        c_quas = sample.third_param_Us_per_signal

        #fit the baselines as exponential curve
        sample.estimate_baseline_parameters(
            native_baseline_type="linear",
            unfolded_baseline_type="exponential",
            window_range_unfolded=ws
        )

        a_exps = sample.first_param_Us_per_signal
        b_exps = sample.second_param_Us_per_signal
        c_exps = sample.third_param_Us_per_signal

        #assigning preprocessed signal and temperature for plotting
        temperature = sample.temp_lst_multiple[0][0]     
        fluorescences = sample.signal_lst_multiple  

        
        #using the removed reference temperature for plotting
        temperature_K = temperature + 273.15
        temperature_K_ref = temperature_K - 298

        #creating all formula results for comparison and plots

        T_window_start = temperature.max() - ws

        for i in range(2):    # 330nm and 350nm

            if len(windows) == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]

            qua_bl = model(temperature_K_ref,
                a_quas[i][0],
                b_quas[i][0],
                c_quas[i][0],
                kind="quadratic")


            exp_bl = model(temperature_K_ref,
                a_exps[i][0],
                b_exps[i][0],
                c_exps[i][0],
                kind="exponential")

            #plotting the results
            ax.scatter(
                temperature,
                fluorescences[i][0],
                s=4,
                color=okabe_ito["black"],
                alpha=0.7,
            )

            ax.plot(
                temperature,
                qua_bl,
                color=okabe_ito["blue"],
                alpha=0.8,
                label=f"Quadratic fit (RMSE={rmse(fluorescences[i][0], qua_bl):.2})",
            )

            ax.plot(
                temperature,
                exp_bl,
                color=okabe_ito["orange"],
                linestyle="--",
                alpha=0.8,
                label=f"Exponential fit (RMSE={rmse(fluorescences[i][0], exp_bl):.2})",
            )

            ax.axvline(
                T_window_start,
                color=okabe_ito["vermillion"],
                linestyle=":",
                linewidth=2,
            )

            ax.text(
                T_window_start+5,
                ax.get_ylim()[0],                
                f"{int(T_window_start)} °C",
                color="black",
                ha="right",
                va="bottom",
                fontsize=10,
            )           

            if i == 0:
                ax.set_title(f"Fitting Window size of {ws}°", fontsize=14)

            # panel label
            if len(windows) == 1:
                label_idx = i
            else:
                label_idx = i * axes.shape[1] + j


            ax.set_title(f"{string.ascii_uppercase[label_idx]})", loc="left", fontsize=14,
                fontweight="bold",)


            if j == 0:
                ax.set_ylabel("Fluorescence")

            ax.legend(loc='upper right')
            #ax.grid(True)

            if i == 1:
                ax.set_xlabel("Temperature (°C)")



        # manual legend (once)
        handles = [
            plt.Line2D([0], [0], color=okabe_ito["black"], linestyle='None', marker='o', markersize=4, label="Signal"),
            plt.Line2D([0], [0], color=okabe_ito["blue"], label="Quadratic baseline"),
            plt.Line2D([0], [0], color=okabe_ito["orange"], linestyle="--", label="Exponential baseline"),
            plt.Line2D([0], [0], color=okabe_ito["vermillion"], linestyle=":", label="Unfolded fit window start"),
        ]

    ws_str = [str(x)for x in windows]

    fig.suptitle(f"{subtitle} - Unfolded baseline curve fitting with varying window sizes: " + "°, ".join(ws_str) + "°", fontsize=18)
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95))
    
    # Row labels for signals
    row_labels = [signal.replace("nm", " nm") for signal in signals]

    # Y positions are in figure coordinates (0–1)
    for i, label in enumerate(row_labels):
        fig.text(
            0.0,                       # x position (left margin)
            0.675 - i * 0.425,             # y position (top to bottom)
            label,
            va="center",
            ha="left",
            rotation=90,
            fontsize=14,
            fontweight="bold",
        )
    
    fig.tight_layout(rect=[0.01, 0, 1, 0.95])
    
    
    return fig