"""Tests for the dynamic optical-flow column metadata generator.

These tests pin down the contract of
:func:`endo_pipeline.settings.column_metadata._build_optical_flow_metadata`,
which produces every optical-flow entry of
:data:`endo_pipeline.settings.column_metadata.COLUMN_METADATA` from a small
per-base specification table plus a ``(dt, ema_alpha)`` variant table and a
per-key override dict.

Scope is intentionally limited to the optical-flow block; other entries of
``COLUMN_METADATA`` are out of scope for this module.
"""

from typing import Any

import pytest

from endo_pipeline.settings.column_metadata import COLUMN_METADATA, ColumnType
from endo_pipeline.settings.column_names import ColumnName

# ---------------------------------------------------------------------------
# Frozen snapshot of every optical-flow entry that COLUMN_METADATA must expose.
# ---------------------------------------------------------------------------
# Each tuple is ``(key, expected_field_overrides)``.  ``expected_field_overrides``
# only lists fields whose value differs from the ``ColumnMetadata`` defaults
# (``label=None``, ``unit=None``, ``min=None``, ``max=None``,
# ``bin_width=None``, ``ticks=None``, ``description=None``).  ``type`` is
# always pinned explicitly because every entry asserts it.
EXPECTED_OPTICAL_FLOW_ENTRIES: list[tuple[str, dict[str, Any]]] = [
    # -- optical_flow_mean_unit_vector base -----------------------------------
    (
        "optical_flow_mean_unit_vector_dt1",
        {
            "name": "Coherent migration (optical flow mean unit vector)",
            "min": 0,
            "max": 1,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    (
        "ema005_optical_flow_mean_unit_vector_dt1",
        {
            "name": "Coherent migration (EMA 0.05, optical flow mean unit vector)",
            "min": 0,
            "max": 1,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    (
        "ema01_optical_flow_mean_unit_vector_dt1",
        {
            "name": "Coherent migration (EMA 0.1, optical flow mean unit vector)",
            "label": "Patch-based\nmigration coherence",
            "min": 0,
            "max": 1,
            "bin_width": 0.02,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    (
        "ema02_optical_flow_mean_unit_vector_dt1",
        {
            "name": "Coherent migration (EMA 0.2, optical flow mean unit vector)",
            "min": 0,
            "max": 1,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    # -- optical_flow_mean_unit_vector_fast base ------------------------------
    (
        "optical_flow_mean_unit_vector_fast",
        {
            "name": "Coherent migration fast (optical flow unit vectors greater than 1 speed)",
            "min": 0,
            "max": 1,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    (
        "ema005_optical_flow_mean_unit_vector_fast_dt1",
        {
            "name": "Coherent migration (EMA 0.05, optical flow mean unit vector fast)",
            "min": 0,
            "max": 1,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    (
        "ema01_optical_flow_mean_unit_vector_fast_dt1",
        {
            "name": "Coherent migration (EMA 0.1, optical flow mean unit vector fast)",
            "min": 0,
            "max": 1,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    (
        "ema02_optical_flow_mean_unit_vector_fast_dt1",
        {
            "name": "Coherent migration (EMA 0.2, optical flow mean unit vector fast)",
            "min": 0,
            "max": 1,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    # -- optical_flow_radial_coherence base -----------------------------------
    (
        "optical_flow_radial_coherence_dt1",
        {
            "name": "Coherent migration (optical flow radial coherence)",
            "type": ColumnType.CONTINUOUS,
        },
    ),
    (
        "ema01_optical_flow_radial_coherence_dt1",
        {
            "name": "Coherent migration (EMA 0.1, optical flow radial coherence)",
            "type": ColumnType.CONTINUOUS,
        },
    ),
    # -- optical_flow_radial_coherence_weighted base --------------------------
    (
        "optical_flow_radial_coherence_weighted_dt1",
        {
            "name": "Coherent migration (optical flow radial coherence weighted)",
            "type": ColumnType.CONTINUOUS,
        },
    ),
    (
        "ema01_optical_flow_radial_coherence_weighted_dt1",
        {
            "name": "Coherent migration (EMA 0.1, optical flow radial coherence weighted)",
            "type": ColumnType.CONTINUOUS,
        },
    ),
    # -- optical_flow_mean_angle base -----------------------------------------
    (
        "optical_flow_mean_angle_dt1",
        {
            "name": "Optical flow mean angle",
            "unit": "rad",
            "min": 0,
            "max": 8,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    # -- optical_flow_angle_std base ------------------------------------------
    (
        "optical_flow_angle_std_dt1",
        {
            "name": "Coherent migration (optical flow angle std dev)",
            "min": 0,
            "max": 4,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    # -- optical_flow_mean_speed base -----------------------------------------
    (
        "optical_flow_mean_speed_dt1",
        {
            "name": "Optical flow mean speed",
            "label": "Patch-based migration speed",
            "unit": "pixels/frame",
            "min": 0,
            "max": 8,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    # -- optical_flow_std_speed base ------------------------------------------
    (
        "optical_flow_std_speed_dt1",
        {
            "name": "Optical flow speed std dev",
            "min": 0,
            "max": 10,
            "type": ColumnType.CONTINUOUS,
        },
    ),
    # -- speed_above_1_count base ---------------------------------------------
    (
        "speed_above_1_count",
        {
            "name": "N vectors with speed above 1",
            "type": ColumnType.DISCRETE,
        },
    ),
]
"""Frozen snapshot of every optical-flow ``ColumnMetadata`` entry."""


EXPECTED_OPTICAL_FLOW_KEYS: frozenset[str] = frozenset(
    key for key, _ in EXPECTED_OPTICAL_FLOW_ENTRIES
)
"""Set of fully-qualified column keys the generator must produce."""


# Default field values on a vanilla ColumnMetadata; used to assert that
# fields not listed in the snapshot kwargs really are at their defaults.
_COLUMN_METADATA_FIELD_DEFAULTS: dict[str, Any] = {
    "label": None,  # ColumnMetadata.__post_init__ sets label = name when None
    "unit": None,
    "description": None,
    "min": None,
    "max": None,
    "bin_width": None,
    "ticks": None,
}


def _all_optical_flow_keys_in_metadata() -> set[str]:
    """Return all string-form keys in ``COLUMN_METADATA`` that look optical-flow."""
    return {
        str(k)
        for k in COLUMN_METADATA
        if "optical_flow" in str(k) or str(k) == "speed_above_1_count"
    }


# ---------------------------------------------------------------------------
# Presence / shape
# ---------------------------------------------------------------------------
class TestOpticalFlowMetadataPresence:
    """The dictionary contains exactly the optical-flow entries we expect."""

    def test_all_expected_keys_are_present(self):
        present = {str(k) for k in COLUMN_METADATA}
        missing = EXPECTED_OPTICAL_FLOW_KEYS - present
        assert not missing, f"Optical-flow keys missing from COLUMN_METADATA: {sorted(missing)}"

    def test_no_unexpected_optical_flow_keys(self):
        actual = _all_optical_flow_keys_in_metadata()
        unexpected = actual - EXPECTED_OPTICAL_FLOW_KEYS
        assert (
            not unexpected
        ), f"Unexpected optical-flow keys in COLUMN_METADATA: {sorted(unexpected)}"

    @pytest.mark.parametrize(
        "member",
        [
            ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
            ColumnName.OpticalFlow.RADIAL_COHERENCE,
            ColumnName.OpticalFlow.RADIAL_COHERENCE_WEIGHTED,
            ColumnName.OpticalFlow.ANGLE_MEAN,
            ColumnName.OpticalFlow.ANGLE_STD,
            ColumnName.OpticalFlow.SPEED_MEAN,
            ColumnName.OpticalFlow.SPEED_STD,
        ],
        ids=lambda m: m.name,
    )
    def test_enum_keyed_entries_resolve(self, member):
        """Entries inserted under their string value are retrievable via the enum.

        Note: ``UNIT_VECTOR_MEAN_FAST`` and ``SPEED_ABOVE_1_COUNT`` are
        intentionally excluded because their dataframe keys
        (``optical_flow_mean_unit_vector_fast`` and ``speed_above_1_count``)
        are legacy unsuffixed forms that do not match the ``_dt1``-suffixed
        enum values.  This quirk is preserved by this PR; see the override
        and variant entries in ``column_metadata.py``.
        """
        assert member in COLUMN_METADATA
        assert COLUMN_METADATA[member].name


# ---------------------------------------------------------------------------
# Per-entry value snapshot
# ---------------------------------------------------------------------------
class TestOpticalFlowMetadataValues:
    """Each generated entry matches the frozen snapshot field-by-field."""

    @pytest.mark.parametrize(
        "key,expected",
        EXPECTED_OPTICAL_FLOW_ENTRIES,
        ids=[k for k, _ in EXPECTED_OPTICAL_FLOW_ENTRIES],
    )
    def test_entry_matches_expected_metadata(self, key, expected):
        assert key in COLUMN_METADATA, f"{key!r} missing from COLUMN_METADATA"
        md = COLUMN_METADATA[key]

        # Every field listed in the snapshot must match exactly.
        for field, value in expected.items():
            actual = getattr(md, field)
            assert actual == value, f"{key!r}.{field}: expected {value!r}, got {actual!r}"

        # Fields not listed in the snapshot must be at their dataclass default,
        # except for ``label`` which ColumnMetadata.__post_init__ defaults to
        # ``name`` when not explicitly provided.
        for field, default in _COLUMN_METADATA_FIELD_DEFAULTS.items():
            if field in expected:
                continue
            actual = getattr(md, field)
            if field == "label":
                assert actual == md.name, f"{key!r}.label should default to name; got {actual!r}"
            else:
                assert (
                    actual == default
                ), f"{key!r}.{field} expected default {default!r}, got {actual!r}"


# ---------------------------------------------------------------------------
# Override semantics
# ---------------------------------------------------------------------------
class TestOpticalFlowOverrides:
    """Per-key overrides replace only the targeted fields."""

    def test_unit_vector_mean_ema01_has_overrides(self):
        md = COLUMN_METADATA[ColumnName.OpticalFlow.UNIT_VECTOR_MEAN]
        # Overridden:
        assert md.label == "Patch-based\nmigration coherence"
        assert md.bin_width == 0.02
        # Inherited from the base spec (must NOT be clobbered):
        assert md.min == 0
        assert md.max == 1
        assert md.type is ColumnType.CONTINUOUS
        assert md.name == "Coherent migration (EMA 0.1, optical flow mean unit vector)"

    def test_speed_mean_has_label_override(self):
        md = COLUMN_METADATA[ColumnName.OpticalFlow.SPEED_MEAN]
        assert md.label == "Patch-based migration speed"
        assert md.name == "Optical flow mean speed"
        assert md.unit == "pixels/frame"

    def test_legacy_fast_name_preserved(self):
        """The unsuffixed fast-coherence key keeps its legacy verbose name."""
        md = COLUMN_METADATA["optical_flow_mean_unit_vector_fast"]
        assert md.name == "Coherent migration fast (optical flow unit vectors greater than 1 speed)"
        # Confirm the override beat the dynamic template (which would have
        # produced "Coherent migration (...)").
        assert "EMA" not in md.name

    def test_speed_above_1_count_keeps_legacy_unsuffixed_key(self):
        """``speed_above_1_count`` must remain the unsuffixed legacy key."""
        assert "speed_above_1_count" in COLUMN_METADATA
        # The ``_dt1`` suffixed enum value must NOT have been emitted by the
        # generator (legacy quirk preserved in this PR).
        assert "speed_above_1_count_dt1" not in {str(k) for k in COLUMN_METADATA}


# ---------------------------------------------------------------------------
# Generator contract
# ---------------------------------------------------------------------------
class TestOpticalFlowGeneratorContract:
    """Contract tests for the dynamic generator itself."""

    def test_no_module_level_scaffolding_leaks(self):
        """All generator scaffolding is scoped to ``_build_optical_flow_metadata``."""
        import endo_pipeline.settings.column_metadata as mod

        leaked = [
            attr
            for attr in (
                "_OF",
                "_OPTICAL_FLOW_BASES",
                "_OPTICAL_FLOW_VARIANTS",
                "_OPTICAL_FLOW_OVERRIDES",
                "_EMA_TAGS",
                "_OpticalFlowBaseSpec",
                "_format_optical_flow_key",
                "_format_optical_flow_name",
            )
            if hasattr(mod, attr)
        ]
        assert not leaked, f"Generator scaffolding leaked at module scope: {leaked}"

    def test_derived_columnmetadata_fields_are_populated(self):
        """``ColumnMetadata.__post_init__`` ran for every generated entry."""
        for key in EXPECTED_OPTICAL_FLOW_KEYS:
            md = COLUMN_METADATA[key]
            assert md.slug, f"{key!r} has empty slug"
            assert md.name_with_unit, f"{key!r} has empty name_with_unit"
            assert md.label_with_unit, f"{key!r} has empty label_with_unit"
            assert md.limits == (md.min, md.max), f"{key!r} limits mismatch"

    @pytest.mark.parametrize(
        "key,expected_fragment",
        [
            ("ema005_optical_flow_mean_unit_vector_dt1", "EMA 0.05"),
            ("ema01_optical_flow_mean_unit_vector_dt1", "EMA 0.1"),
            ("ema02_optical_flow_mean_unit_vector_dt1", "EMA 0.2"),
            ("ema005_optical_flow_mean_unit_vector_fast_dt1", "EMA 0.05"),
            ("ema01_optical_flow_radial_coherence_dt1", "EMA 0.1"),
            ("ema01_optical_flow_radial_coherence_weighted_dt1", "EMA 0.1"),
        ],
    )
    def test_ema_alpha_appears_in_name_template(self, key, expected_fragment):
        assert expected_fragment in COLUMN_METADATA[key].name

    @pytest.mark.parametrize(
        "key",
        [
            "optical_flow_mean_unit_vector_dt1",
            "optical_flow_radial_coherence_dt1",
            "optical_flow_radial_coherence_weighted_dt1",
        ],
    )
    def test_non_ema_entries_have_no_ema_in_name(self, key):
        assert "EMA" not in COLUMN_METADATA[key].name

    @pytest.mark.parametrize(
        "key",
        [
            "optical_flow_mean_angle_dt1",
            "optical_flow_mean_speed_dt1",
            "optical_flow_std_speed_dt1",
            "speed_above_1_count",
        ],
    )
    def test_non_coherence_entries_skip_coherent_migration_template(self, key):
        """Non-coherence bases render with the plain sentence-case label."""
        assert "Coherent migration" not in COLUMN_METADATA[key].name
