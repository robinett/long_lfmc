import argparse
import plotting as plot

def main(
    fname,
    var_name,
    proj_in,
    proj_out,
    save_name
):
    """
    Plot a variable from a .nc file using cartopy.
    Parameters
    ----------
    fname : str
        Path to the .nc file to plot.
    var_name : str
        Variable name to plot.
    proj_in : str
        Projection of the input data.
    proj_out : str
        Projection to plot at.
    save_name : str
        File name to save the figure as.
    """
    plot.plot_from_xarray(
        'fname',
        fname,
        var_name,
        proj_in,
        proj_out,
        save_name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name",
        type=str,
        help="Path to the .nc file to plot",
        required=True
    )
    parser.add_argument(
        "--var_name",
        type=str,
        help="Variable name to plot",
        required=True
    )
    parser.add_argument(
        "--proj_in",
        type=str,
        help="Projection of the input data",
        required=True
    )
    parser.add_argument(
        "--proj_out",
        type=str,
        help="Projection to plot at",
        required=True
    )
    parser.add_argument(
        "--save_name",
        type=str,
        help="File name to save the figure as",
        required=True
    )
    fname = parser.parse_args().file_name
    var_name = parser.parse_args().var_name
    proj_in = parser.parse_args().proj_in
    proj_out = parser.parse_args().proj_out
    save_name = parser.parse_args().save_name
    main(
        fname,
        var_name,
        proj_in,
        proj_out,
        save_name
    )

