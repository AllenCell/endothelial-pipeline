import re
from pathlib import Path


def extract_position_from_filepath(
    filepath: str | Path,
    int_only: bool = True,
    use_last_match: bool = True,
    default_if_not_found: int | str = "",
) -> str | int:
    """
    Extract the position value from a string or Path.
    Searches for the pattern "P[0-9]+" to find the position.
    If use_last_match is True then the last match will be used,
    otherwise the first one will be used.

    Parameters
    ----------
    filepath: str or Path
        A string or Path to get the position from.
    int_only: bool
        Whether to return just the position as an integer or
        an entire string (i.e. 10 vs 'P10')
        Default is True (i.e. just an integer).
    use_last_match: bool
        Whether to use the last match (in the event that multiple possible
        position values were found in the string).
        If False then the first match will be used.
        E.g. image_name_P1_P3_etc_T57.tif can return either P1 or P3, but
        will return 3 by default. Ideally the position in fp_as_string
        would be unambiguous.
        Default is True.
    default_if_not_found: int or str
        The value to return if no position is found in the string.

    Returns
    -------
    P: int or str
        The position represented as an integer if int_only is True, otherwise
        the position represented as a string including the P before.
    """

    if isinstance(filepath, Path):
        filepath = str(filepath)

    index = -1 if use_last_match else 0
    p = re.findall("P[0-9]+", filepath)
    position_value = int(p[index].split("P")[-1]) if p else default_if_not_found
    if not p:
        print("""No 'P[0-9]+' found in filename. Using P == default_if_not_found.""")

    return position_value if int_only else f"P{position_value}"
