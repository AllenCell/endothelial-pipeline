"""Supplementary figure: optical-flow coherent vs. incoherent example panels.

Renders a 2x2 figure showing one *coherent* (high migration coherence) and
one *incoherent* (low migration coherence) crop / timepoint pair.  Each row
contains the magenta/green composite of consecutive BF frames and the
TVL1 quiver plot annotated with the migration coherence (R-bar).

The picks live in :mod:`endo_pipeline.settings.examples` as
``SUPP_FIG_OPTICAL_FLOW_{COHERENT,INCOHERENT}_EXAMPLE``.  The figure
generator is in :mod:`endo_pipeline.library.visualize.supp_fig_optical_flow`.
"""


def main() -> None:
    """Build the supplementary optical-flow figure."""
    import matplotlib.pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.supp_fig_optical_flow import build_supp_fig_optical_flow
    from endo_pipeline.settings.examples import (
        SUPP_FIG_OPTICAL_FLOW_COHERENT_EXAMPLE,
        SUPP_FIG_OPTICAL_FLOW_INCOHERENT_EXAMPLE,
    )

    plt.style.use("endo_pipeline.figure")

    output_dir = get_output_path("supp_fig_optical_flow")
    build_supp_fig_optical_flow(
        coherent=SUPP_FIG_OPTICAL_FLOW_COHERENT_EXAMPLE,
        incoherent=SUPP_FIG_OPTICAL_FLOW_INCOHERENT_EXAMPLE,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
