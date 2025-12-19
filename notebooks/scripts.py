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