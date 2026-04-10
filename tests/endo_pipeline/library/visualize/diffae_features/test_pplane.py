import pytest

from endo_pipeline.library.visualize.diffae_features.fixed_points import (
    get_stability_label_from_fpt_type,
)
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel

# test "easy cases" where the input string exactly matches expected patterns
# as output by get_fpt_type method


@pytest.mark.parametrize(
    "fpt_type",
    [
        "stable node",
        "stable spiral",
    ],
)
def test_get_stability_label_stable(fpt_type: str) -> None:
    assert get_stability_label_from_fpt_type(fpt_type) == StabilityLabel.STABLE


@pytest.mark.parametrize(
    "fpt_type",
    [
        "unstable node",
        "unstable spiral",
    ],
)
def test_get_stability_label_unstable(fpt_type: str) -> None:
    assert get_stability_label_from_fpt_type(fpt_type) == StabilityLabel.UNSTABLE


def test_get_stability_label_saddle() -> None:
    assert get_stability_label_from_fpt_type("saddle point") == StabilityLabel.SADDLE


def test_get_stability_label_indeterminate() -> None:
    assert (
        get_stability_label_from_fpt_type("indeterminate stability") == StabilityLabel.INDETERMINATE
    )


# test ability to distinguish between stable vs. unstable


def test_get_stability_label_stable_not_confused_with_unstable() -> None:
    """'stable' must not match an 'unstable ...' string."""
    assert get_stability_label_from_fpt_type("unstable node") != StabilityLabel.STABLE


def test_get_stability_label_unstable_not_confused_with_stable() -> None:
    """'unstable ...' must not return 'stable'."""
    result = get_stability_label_from_fpt_type("unstable spiral")
    assert result == StabilityLabel.UNSTABLE
    assert result != StabilityLabel.STABLE


# test case insensitivity


@pytest.mark.parametrize(
    "fpt_type, expected",
    [
        ("Stable node", StabilityLabel.STABLE),
        ("STABLE node", StabilityLabel.STABLE),
        ("Unstable node", StabilityLabel.UNSTABLE),
        ("UNSTABLE spiral", StabilityLabel.UNSTABLE),
        ("Saddle point", StabilityLabel.SADDLE),
        ("Indeterminate stability", StabilityLabel.INDETERMINATE),
    ],
)
def test_get_stability_label_case_insensitive(fpt_type: str, expected: StabilityLabel) -> None:
    """The function is case-insensitive and always returns a lowercase label."""
    result = get_stability_label_from_fpt_type(fpt_type)
    assert result == expected
    assert result == result.lower()


# test word-boundary enforcement (regex-specific)


def test_get_stability_label_partial_label_prefix_returns_unknown() -> None:
    assert get_stability_label_from_fpt_type("stableish point") == "unknown"


# test other unrecognized strings


def test_get_stability_label_empty_string_returns_unknown() -> None:
    assert get_stability_label_from_fpt_type("") == "unknown"


def test_get_stability_label_unrecognised_string_returns_unknown() -> None:
    assert get_stability_label_from_fpt_type("center point") == "unknown"
