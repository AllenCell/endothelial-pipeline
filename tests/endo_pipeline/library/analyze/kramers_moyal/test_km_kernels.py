import pytest

from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel


@pytest.mark.parametrize(
    "name, bandwidth, specify_period, period, raises_error",
    [
        ("epanechnikov", 0.1, False, None, False),  # valid Epanechnikov kernel
        ("gaussian", 0.2, False, None, False),  # valid Gaussian kernel
        ("periodic", 0.3, True, 1.0, False),  # valid periodic kernel
        ("invalid_kernel", 0.1, False, None, True),  # invalid kernel name
        ("gaussian", -0.1, False, None, True),  # negative bandwidth
        ("epanechnikov", 0.2, True, 1.0, True),  # specifying period for non-periodic kernel
        ("periodic", 0.3, True, -1.0, True),  # negative period for periodic kernel
        ("periodic", 0.3, True, None, True),  # missing period for periodic kernel
        (
            "periodic",
            0.3,
            False,
            None,
            True,
        ),  # missing period for periodic kernel (using default None)
    ],
)
def test_kramers_moyal_kernel(
    name: str, bandwidth: float, specify_period: bool, period: float | None, raises_error: bool
) -> None:
    """Test that KramersMoyalKernel validates its parameters correctly."""
    try:
        if specify_period:
            KramersMoyalKernel(
                name=name,
                bandwidth=bandwidth,
                period=period,
            )
        else:
            KramersMoyalKernel(name=name, bandwidth=bandwidth)
        if raises_error:
            pytest.fail("Expected ValueError was not raised")
    except ValueError:
        if not raises_error:
            pytest.fail("Unexpected ValueError was raised")
