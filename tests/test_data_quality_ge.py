# tests/test_data_quality_ge.py
"""
Data quality validation using Great Expectations.

Two ways to run this file:

  1. As part of the test suite:
         python3 -m pytest tests/test_data_quality_ge.py -v

  2. Standalone, to generate the HTML data quality report:
         python3 tests/test_data_quality_ge.py
     Then open: gx/uncommitted/data_docs/local_site/index.html

  3. Demo mode, which tightens two thresholds so the report shows
     real failures rather than a wall of green:
         python3 tests/test_data_quality_ge.py --strict
     Nothing needs reverting afterwards — the pytest suite always
     uses the normal thresholds.

The difference from test_data_quality.py: those are hand-written pytest
assertions. These are declarative expectations, and Great Expectations
produces a shareable HTML report from them — something a non-engineer
can read without looking at terminal output.
"""

import pandas as pd
import pytest
import great_expectations as gx

DATA_PATH = "data/drugsComTrain_raw.csv"
SUITE_NAME = "patient_review_quality"


# ── Data loading ─────────────────────────────────────────────

def load_reviews_df() -> pd.DataFrame:
    """Load the patient review dataset."""
    return pd.read_csv(
        DATA_PATH,
        sep=",",
        on_bad_lines="skip",
        encoding="utf-8",
    )


# ── Expectation suite definition ─────────────────────────────

def build_expectation_suite(context, strict: bool = False):
    """
    Declare what a valid patient review dataset looks like.

    Each expectation is a statement about the data, not about code.
    Read top to bottom, this is a specification of the dataset.

    strict=True tightens two thresholds past what the real data can
    meet. Used only for demos, so the HTML report shows what a
    failure actually looks like.
    """
    name = SUITE_NAME + ("_strict" if strict else "")
    suite = context.suites.add(gx.ExpectationSuite(name=name))

    # ── Schema: the columns the pipeline depends on must exist ──
    for column in ["drugName", "condition", "review", "rating"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=column)
        )

    # ── Completeness: reviews are the input to every model ──
    # mostly=0.95 allows a 5% null rate before failing. Real-world
    # data is never perfectly clean, and failing on a single null
    # would make this suite useless.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="review", mostly=1.0 if strict else 0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="drugName", mostly=0.90
        )
    )

    # ── Validity: ratings are a 1-10 scale ──
    # A rating of 0 or 11 means the source data is corrupted.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="rating", min_value=1, max_value=10
        )
    )

    # ── Type safety: reviews must be text for the models to read ──
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="review", type_="str"
        )
    )

    # ── Length: reviews under ~10 chars carry no signal ──
    # mostly=0.99 tolerates the small tail of one-word reviews.
    suite.add_expectation(
        gx.expectations.ExpectColumnValueLengthsToBeBetween(
            column="review",
            min_value=50 if strict else 10,
            max_value=20000,
            mostly=0.99,
        )
    )

    # ── Volume: too few rows means the download failed ──
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=1000, max_value=500000
        )
    )

    # ── Uniqueness: duplicate IDs would break row-level tracing ──
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="uniqueID")
    )

    return suite


def _reset(context, ds_name, suite_name, vd_name):
    """
    Remove artefacts from a previous run.

    A file-backed context persists data sources and suites to disk,
    so re-running would otherwise fail with "already exists".
    """
    for delete_fn, name in (
        (context.validation_definitions.delete, vd_name),
        (context.suites.delete, suite_name),
        (context.data_sources.delete, ds_name),
    ):
        try:
            delete_fn(name)
        except Exception:
            pass  # not present on a first run


def run_validation(context, df, strict: bool = False):
    """Wire up a data source, run the suite, return results."""
    label = "strict" if strict else "standard"
    ds_name = f"patient_reviews_{label}"

    _reset(
        context,
        ds_name,
        SUITE_NAME + ("_strict" if strict else ""),
        f"patient_review_validation_{label}",
    )

    data_source = context.data_sources.add_pandas(ds_name)
    asset = data_source.add_dataframe_asset(name="drug_reviews")
    batch_definition = asset.add_batch_definition_whole_dataframe(
        "full_dataset"
    )

    suite = build_expectation_suite(context, strict=strict)

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            data=batch_definition,
            suite=suite,
            name=f"patient_review_validation_{label}",
        )
    )

    return validation_definition.run(batch_parameters={"dataframe": df})


# ── Pytest integration ───────────────────────────────────────

@pytest.fixture(scope="module")
def validation_results():
    """
    Run the full expectation suite once, share results across tests.

    scope="module" matters: without it, pytest would re-run the whole
    validation for every test below.
    """
    df = load_reviews_df()
    # mode="ephemeral" is required, not optional. A bare get_context()
    # auto-detects any gx/ folder left by the standalone run and uses
    # that file context instead — which then fails on re-runs because
    # the data source already exists.
    context = gx.get_context(mode="ephemeral")
    return run_validation(context, df)


def result_for(results, expectation_type, column=None):
    """Pull one expectation's result out of the validation output."""
    for r in results.results:
        cfg = r.expectation_config
        if cfg.type != expectation_type:
            continue
        if column and cfg.kwargs.get("column") != column:
            continue
        return r
    raise AssertionError(
        f"No result found for {expectation_type} on column {column}"
    )


def test_suite_passes_overall(validation_results):
    """Every expectation in the suite must pass."""
    failed = [
        r.expectation_config.type
        for r in validation_results.results
        if not r.success
    ]
    assert validation_results.success, f"Failed expectations: {failed}"


@pytest.mark.parametrize(
    "column", ["drugName", "condition", "review", "rating"]
)
def test_required_column_exists(validation_results, column):
    """The pipeline reads these four columns — all must be present."""
    r = result_for(validation_results, "expect_column_to_exist", column)
    assert r.success, f"Missing required column: {column}"


def test_reviews_are_mostly_populated(validation_results):
    """At least 95% of reviews must be non-null."""
    r = result_for(
        validation_results,
        "expect_column_values_to_not_be_null",
        "review",
    )
    assert r.success


def test_ratings_within_scale(validation_results):
    """Ratings must fall on the documented 1-10 scale."""
    r = result_for(
        validation_results,
        "expect_column_values_to_be_between",
        "rating",
    )
    assert r.success, (
        f"Out-of-range ratings: "
        f"{r.result.get('unexpected_count')} rows"
    )


def test_reviews_are_strings(validation_results):
    """Reviews must be text — the models cannot read anything else."""
    r = result_for(
        validation_results,
        "expect_column_values_to_be_of_type",
        "review",
    )
    assert r.success


def test_reviews_have_usable_length(validation_results):
    """At least 99% of reviews carry enough text to analyse."""
    r = result_for(
        validation_results,
        "expect_column_value_lengths_to_be_between",
        "review",
    )
    assert r.success


def test_dataset_is_complete(validation_results):
    """Row count in expected range — catches a truncated download."""
    r = result_for(
        validation_results, "expect_table_row_count_to_be_between"
    )
    assert r.success


def test_review_ids_are_unique(validation_results):
    """Duplicate IDs would break row-level traceability."""
    r = result_for(
        validation_results,
        "expect_column_values_to_be_unique",
        "uniqueID",
    )
    assert r.success


# ── Standalone mode: generate the HTML report ────────────────

if __name__ == "__main__":
    import sys

    strict = "--strict" in sys.argv

    print("=" * 60)
    print("CareInsight AI — Data Quality Validation")
    if strict:
        print("STRICT MODE — thresholds tightened to surface failures")
    print("=" * 60)

    df = load_reviews_df()
    print(f"\nLoaded {len(df):,} rows, {len(df.columns)} columns")

    # File-backed context so Data Docs persist to disk
    context = gx.get_context(mode="file", project_root_dir=".")
    results = run_validation(context, df, strict=strict)

    print("\n" + "-" * 60)
    for r in results.results:
        mark = "PASS" if r.success else "FAIL"
        col = r.expectation_config.kwargs.get("column", "table")
        name = r.expectation_config.type.replace("expect_", "")
        line = f"  [{mark}]  {col:<12} {name}"
        if not r.success:
            unexpected = r.result.get("unexpected_percent")
            if unexpected is not None:
                line += f"  ({unexpected:.2f}% of rows)"
        print(line)
    print("-" * 60)

    passed = sum(1 for r in results.results if r.success)
    total = len(results.results)
    print(f"\n{passed}/{total} expectations passed")
    print(f"Overall: {'PASSED' if results.success else 'FAILED'}")

    context.build_data_docs()
    print(
        "\nHTML report generated:\n"
        "  gx/uncommitted/data_docs/local_site/index.html\n"
    )