from pathlib import Path

from rogii_wellbore.data import parse_horizontal_filename, parse_typewell_filename
from rogii_wellbore.submission import parse_submission_id


def test_parse_horizontal_filename() -> None:
    assert parse_horizontal_filename(Path("000d7d20__horizontal_well.csv")) == "000d7d20"


def test_parse_typewell_filename() -> None:
    assert parse_typewell_filename(Path("000d7d20__typewell.csv")) == ("000d7d20", "000d7d20")


def test_parse_submission_id() -> None:
    assert parse_submission_id("000d7d20_1442") == ("000d7d20", 1442)
