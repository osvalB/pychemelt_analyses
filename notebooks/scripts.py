import numpy as np
import pychemelt as pychem

def display_figure_static(fig, format="png", width=800, height=600, show_interactive=False):
    """
    Display a Plotly figure in both interactive and static formats.

    This function is useful for Jupyter notebooks that need to render properly
    on GitHub, where interactive Plotly figures may not display.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure object to display
    format : str, default "png"
        Image format for static display ("png", "svg", "jpeg")
    show_interactive : bool, default True
        Whether to also show the interactive version

    """
    try:
        # Try to import display functionality
        from IPython.display import Image, display

        # Show interactive version first (for local Jupyter)
        if show_interactive:
            fig.show()

        # Display static version (for GitHub compatibility)
        static_image = fig.to_image(format=format, width=width, height=height, scale=2)
        display(Image(static_image))

    except ImportError:
        # Fallback to regular show if not in Jupyter
        print("IPython not available, showing interactive figure only")
        fig.show()
    except Exception as e:
        print(f"Error creating static image: {e}")
        print("Showing interactive figure only")
        fig.show()

def aux_create_pychem_sim(params,rng_seed=2):

    """
    Generate a Pychemelt Sample object with simulated data.
    Parameters
    ---------
    params : dict
        dictionary containing the parameters for the simulated signal (e.g., Tm, DH)
    Returns
    ------
    pychem.Sample()
    """

    rng = np.random.default_rng(rng_seed)

    concs = np.arange(1,5.5,0.5)

    # Calculate signal range for proper y-axis scaling
    temp_range = np.linspace(20, 90, 100) + 273.15
    signal_list = []
    temp_list   = []

    for i, D in enumerate(concs):

        y = pychem.signal_two_state_tc_unfolding(temp_range, D, **params)

        # Add gaussian error to simulated signal
        noise = rng.normal(0, 0.25, len(y))
        y += noise
        #y += rng.normal(0, 0.02, len(y))

        # Add error to the initial signal to model variance across positions
        y *= rng.uniform(0.9,1.1)

        signal_list.append(y)
        temp_list.append(temp_range)

    pychem_sim = pychem.Monomer()

    pychem_sim.signal_dic['Simulated signal'] = signal_list
    pychem_sim.temp_dic['Simulated signal']   = [temp_range for _ in range(len(concs))]

    pychem_sim.conditions = concs

    pychem_sim.global_min_temp = np.min(temp_range)
    pychem_sim.global_max_temp = np.max(temp_range)

    pychem_sim.set_denaturant_concentrations()

    pychem_sim.set_signal('Simulated signal')

    pychem_sim.select_conditions(normalise_to_global_max=False)
    pychem_sim.expand_multiple_signal()

    return pychem_sim

def fit_pychem_sim(pychem_sim_x,n_residues):

    """
    Given a Pychemelt Sample object with simulated data, do a global fitting

    Parameters
    ---------
    pychem_sim_x : pychemelt.Sample()

    Returns
    ------
    None

    """

    pychem_sim_x.estimate_baseline_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='exponential'
    )

    pychem_sim_x.fit_thermal_unfolding_local()

    pychem_sim_x.n_residues = n_residues # only for cp initial guess
    pychem_sim_x.guess_Cp()

    pychem_sim_x.fit_thermal_unfolding_global()

    pychem_sim_x.fit_thermal_unfolding_global_global()
    pychem_sim_x.fit_thermal_unfolding_global_global_global(model_scale_factor=True)

    return None

temperature_to_kelvin  = lambda T: T + 273.15 if np.max(T) < 270 else T

def DG_at_T(T,Tm, Cp0, DHm):

    """
    Calculates the DG at the given temperature.
    Parameters
    ----------
    T : float
        temperature to calculate DG at. In Kelvin or Celsius units.
    Tm : float
        temperature of melting where [N] = [U]. In Kelvin or Celsius units.
    Cp0 : float
        heat capacity change upon unfolding. In kcal/M
    DHm : float
        enthalpy change upon unfolding. In kcal/mol/K

    Returns
    -------
    DG : float
        Gibbs free energy at temperature T
    """

    T = temperature_to_kelvin(T)
    Tm = temperature_to_kelvin(Tm)

    DG = DHm * (1 - T / Tm) + Cp0 * (T - Tm - T * np.log(T / Tm))

    return DG

def folded_fraction(DG,T):
    """
    Calculates the folded fraction from DG at temperature T.
    Parameters
    ----------
    DG : float
        Gibbs free energy in kcal/mol
    T : float
        temperature in Kelvin

    Returns
    -------
    fN : float
        folded fraction
    """
    R = 1.987 / 1000 # kcal/mol/K

    T = temperature_to_kelvin(T)

    K = np.exp(-DG / (R * T)) # at 5C
    f_n = 1 / (1 + K)

    return f_n