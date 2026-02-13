import pytest

from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel


@pytest.mark.parametrize(
    "name,bandwidth,period",
    [
        ("epanechnikov", 0.1, None),  # valid Epanechnikov kernel
        ("gaussian", 0.2, None),  # valid Gaussian kernel
        ("periodic", 0.3, 1.0),  # valid periodic kernel
    ],
)
def test_kramers_moyal_kernel_valid_parameters(name, bandwidth, period):
    KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period)


@pytest.mark.parametrize(
    "name,bandwidth,period,specify_period",
    [
        ("invalid_kernel", 0.1, None, False),  # invalid kernel name
        ("gaussian", -0.1, None, False),  # negative bandwidth
        ("epanechnikov", 0.2, 1.0, True),  # specifying period for non-periodic kernel
        ("periodic", 0.3, -1.0, True),  # negative period for periodic kernel
        ("periodic", 0.3, None, True),  # missing period for periodic kernel
        ("periodic", 0.3, None, False),  # missing period for periodic kernel (using default None)
    ],
)
def test_kramers_moyal_kernel_invalid_parameters(name, bandwidth, period, specify_period):
    with pytest.raises(ValueError):
        if specify_period:
            KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period)
        else:
            KramersMoyalKernel(name=name, bandwidth=bandwidth)
