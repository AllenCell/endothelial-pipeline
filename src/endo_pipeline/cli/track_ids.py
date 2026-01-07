from collections.abc import Sequence
from typing import Annotated

from cyclopts import Parameter, Token


def _track_id_converter(_, tokens: Sequence[Token]) -> list[int]:
    """Convert CLI tokens into list of track IDs."""

    track_ids = {int(token.value) for token in tokens}

    return sorted(track_ids)


TrackIds = Annotated[
    list[int],
    Parameter(
        consume_multiple=True,  # allows parameter to consume multiple tokens
        negative_iterable=[],  # remove the "--empty" option
        converter=_track_id_converter,  # run track_id converter on tokens
    ),
]
