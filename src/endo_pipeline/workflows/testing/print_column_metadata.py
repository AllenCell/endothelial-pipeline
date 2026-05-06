"""Print every entry in ``COLUMN_METADATA`` for human inspection.

Because ``settings.column_metadata`` should contain only constants (no logic
worth unit-testing), this workflow exists as the lightweight
"is-the-table-still-correct?" check.  Run it whenever you change
``COLUMN_METADATA`` or ``ColumnName`` and eyeball the output.
"""

from endo_pipeline.cli import tags

TAGS = [tags.TEST_READY, tags.CPU_ONLY]


def main(filter: str = "") -> None:
    """
    Print every entry in COLUMN_METADATA, optionally filtered by substring.

    Parameters
    ----------
    filter
        If non-empty, only entries whose key contains this substring (case
        insensitive) are printed.  Useful for inspecting just the optical-flow
        block, for example: ``--filter optical_flow``.
    """
    import logging

    from endo_pipeline.settings.column_metadata import COLUMN_METADATA

    logger = logging.getLogger(__name__)

    needle = filter.lower()
    matches = [
        (key, md) for key, md in COLUMN_METADATA.items() if not needle or needle in str(key).lower()
    ]

    if not matches:
        logger.warning("No COLUMN_METADATA entries match filter %r.", filter)
        return

    logger.info("Printing %d / %d COLUMN_METADATA entries.", len(matches), len(COLUMN_METADATA))
    for key, md in matches:
        print(f"\n{key!s}")
        print(f"    type            = {md.type.value}")
        print(f"    name            = {md.name!r}")
        print(f"    label           = {md.label!r}")
        print(f"    unit            = {md.unit!r}")
        print(f"    description     = {md.description!r}")
        print(f"    min             = {md.min!r}")
        print(f"    max             = {md.max!r}")
        print(f"    bin_width       = {md.bin_width!r}")
        print(f"    ticks           = {md.ticks!r}")
        print(f"    slug            = {md.slug!r}")
        print(f"    name_with_unit  = {md.name_with_unit!r}")
        print(f"    label_with_unit = {md.label_with_unit!r}")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
