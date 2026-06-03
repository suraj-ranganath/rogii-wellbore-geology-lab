from pathlib import Path

from rogii_wellbore.data import parse_horizontal_filename, parse_typewell_filename


def test_parse_horizontal_filename() -> None:
    assert parse_horizontal_filename(Path("Well10001__horizontal_well.csv")) == "Well10001"


def test_parse_typewell_filename() -> None:
    assert parse_typewell_filename(Path("Well10001__typewell__Typewell20001.csv")) == (
        "Well10001",
        "Typewell20001",
    )
