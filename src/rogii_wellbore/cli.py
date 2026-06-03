from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from rogii_wellbore.baseline import cross_validate_baseline, train_full_baseline
from rogii_wellbore.data import (
    download_competition,
    find_sample_submission,
    read_csv,
    scan_wells,
    summarize_columns,
)
from rogii_wellbore.features import canonicalize, prediction_mask
from rogii_wellbore.paths import raw_competition_dir

app = typer.Typer(no_args_is_help=True)
console = Console()

DEFAULT_RAW_DIR = raw_competition_dir()
DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_CV_OUTPUT = Path("outputs/baseline_cv.json")

OUTPUT_DIR_OPTION = typer.Option(
    DEFAULT_RAW_DIR,
    "--output-dir",
    "-o",
    help="Directory for competition files.",
)
DATA_DIR_OPTION = typer.Option(DEFAULT_RAW_DIR, "--data-dir", "-d")
CONFIG_OPTION = typer.Option(DEFAULT_CONFIG, "--config", "-c")
CV_OUTPUT_OPTION = typer.Option(DEFAULT_CV_OUTPUT, "--output", "-o")
FORCE_OPTION = typer.Option(False, "--force", help="Force Kaggle API redownload.")
SAMPLE_ROWS_OPTION = typer.Option(200, "--sample-rows", min=10)


@app.command("download-data")
def download_data(
    output_dir: Path = OUTPUT_DIR_OPTION,
    force: bool = FORCE_OPTION,
) -> None:
    path = download_competition(output_dir, force=force)
    console.print(f"Downloaded competition data to [bold]{path}[/bold]")


@app.command("inspect-data")
def inspect_data(
    data_dir: Path = DATA_DIR_OPTION,
    sample_rows: int = SAMPLE_ROWS_OPTION,
) -> None:
    pairs = scan_wells(data_dir)
    table = Table(title="Well Files")
    table.add_column("Split")
    table.add_column("Pairs", justify="right")
    table.add_column("Missing Typewell", justify="right")
    for split in sorted({pair.split for pair in pairs} or {"none"}):
        split_pairs = [pair for pair in pairs if pair.split == split]
        table.add_row(
            split,
            str(len(split_pairs)),
            str(sum(pair.typewell_path is None for pair in split_pairs)),
        )
    console.print(table)

    sample_submission = find_sample_submission(data_dir)
    if sample_submission:
        console.print(f"Sample submission: [bold]{sample_submission}[/bold]")
        console.print(read_csv(sample_submission, nrows=5).head())
    else:
        console.print("Sample submission: not found")

    paths = []
    for pair in pairs[: min(len(pairs), 8)]:
        paths.append(pair.horizontal_path)
        if pair.typewell_path is not None:
            paths.append(pair.typewell_path)
    if paths:
        columns = summarize_columns(paths, nrows=sample_rows)
        console.print(columns.groupby(["column", "dtype"], dropna=False).size().reset_index(name="files"))

    mask_rows: list[dict[str, object]] = []
    for pair in pairs[: min(len(pairs), 20)]:
        horizontal = canonicalize(read_csv(pair.horizontal_path))
        mask = prediction_mask(horizontal, require_target=False)
        mask_rows.append(
            {
                "well_id": pair.well_id,
                "split": pair.split,
                "rows": int(len(horizontal)),
                "prediction_rows": int(mask.sum()),
                "has_tvt": "tvt" in horizontal and bool(horizontal["tvt"].notna().any()),
                "has_tvt_input": "tvt_input" in horizontal,
            }
        )
    if mask_rows:
        console.print(pd.DataFrame(mask_rows))


@app.command("cv-baseline")
def cv_baseline(
    config: Path = CONFIG_OPTION,
    output: Path = CV_OUTPUT_OPTION,
) -> None:
    metrics = cross_validate_baseline(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    console.print_json(json.dumps(metrics))
    console.print(f"Wrote CV metrics to [bold]{output}[/bold]")


@app.command("train-baseline")
def train_baseline(
    config: Path = CONFIG_OPTION,
) -> None:
    model_path, summary = train_full_baseline(config)
    console.print_json(json.dumps(summary))
    console.print(f"Wrote model to [bold]{model_path}[/bold]")
