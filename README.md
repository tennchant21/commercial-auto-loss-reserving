# Loss Reserving: Commercial Auto (CAS Schedule P)

Estimates unpaid claims for the US Commercial Auto industry portfolio using the
[chainladder](https://chainladder-python.readthedocs.io/) package. The project uses
three methods: chain ladder, Mack, and Bornhuetter-Ferguson. The development
factors include a curve-fitted tail.

## Data

`data/clrd_comauto.csv` holds the Commercial Auto rows of the CAS Loss Reserving
Database, which ships with chainladder. It covers 158 company groups, accident
years 1988–1997, and 10 development years, and it is net of reinsurance. The
file was written by `reserving.data.export_sample("comauto", path)`.

| Column | Use |
|---|---|
| `GRCODE`, `GRNAME` | Company group; summed away |
| `AccidentYear`, `DevelopmentYear` | Triangle origin and valuation year |
| `CumPaidLoss` | **Reserved measure** (cumulative paid, net) |
| `EarnedPremNet` | **Exposure** for BF (net earned premium) |
| `IncurLoss`, `BulkLoss`, `EarnedPremDIR`, `EarnedPremCeded`, `PostedReserve97`, `DevelopmentLag`, `Single`, `LOB` | Not used |

A few companies report negative paid losses or premium in some cells. After
aggregating to the portfolio level, no cell is negative. `validate_claims`
enforces this.

## A priori expected loss ratio: 0.68

The ELR is the only judgemental input to Bornhuetter-Ferguson. Losses to date
over net earned premium, by accident year:

| AY | Paid LR to date | Incurred LR to date | Chain ladder paid ultimate LR |
|---|---|---|---|
| 1988 | 0.685 | 0.689 | 0.688 |
| 1989 | 0.699 | 0.705 | 0.707 |
| 1990 | 0.682 | 0.694 | 0.693 |
| 1991 | 0.634 | 0.655 | 0.650 |
| 1992 | 0.625 | 0.660 | 0.654 |
| 1993 | 0.628 | 0.687 | 0.684 |
| 1994 | 0.586 | 0.699 | 0.695 |
| 1995 | 0.505 | 0.695 | 0.703 |
| 1996 | 0.376 | 0.666 | 0.707 |
| 1997 | 0.199 | 0.671 | **0.765** |

- In the mature years 1988–1990, paid and incurred have converged at 0.68–0.70.
- The incurred loss ratio to date across all ten years is **0.682**. For 1988–1992 it is **0.679**.
- Chain ladder puts 1997 at 0.765. That estimate comes from one year of paid
  data multiplied by a 12-to-ultimate factor of 3.85. The case-incurred loss
  ratio for 1997 (0.671) does not support that level. This leverage is the
  reason to use BF.

0.68 is therefore a rounded, data-supported level. One caveat: it comes from the
same triangle it is applied to, so it is not independent of the data. A
pricing-plan loss ratio or an industry benchmark would be a better source if one
were available. Change `Config.apriori_elr` to test sensitivity. BF IBNR is
linear in the ELR, and there is a test for this.

## Method

1. Sum companies to one portfolio triangle (`data.py`).
2. Fit volume-weighted link ratios, then add an exponential `TailCurve` tail
   (1.0037 from 120 months to ultimate) (`methods.develop`).
3. Fit chain ladder, Mack and BF on that developed triangle (`methods.fit_methods`).
4. Summarise ultimates, IBNR, ultimate loss ratios and Mack standard errors
   (`summary.py`). Write CSVs and PNGs to `outputs/`.

The inverse-power tail is much heavier (1.039). To compare, set
`Config.tail_curve = "inverse_power"`.

**Adding methods:** write a `fit_*` function in `methods.py` and add it to the
`fit_methods` dictionary. The summary tables and plots work with any fitted
chainladder estimator. Cape Cod and the ODP bootstrap are planned next.

## Results (paid, net, $000s, as at 1997-12)

| Method | Total IBNR |
|---|---|
| Chain ladder / Mack | 1,773,749 |
| Bornhuetter-Ferguson (ELR 0.68) | 1,660,501 |

The difference between the two is almost entirely accident year 1997 (775k vs
689k). Mack's standard error on the total reserve is 65,436, a coefficient of
variation of 3.7%. The breakdown by accident year is in `outputs/mack_summary.csv`.

## Hand-built check

`manual.py` implements volume-weighted chain ladder in plain numpy. It is the
only code that reimplements chainladder. `tests/test_manual.py` checks it
against a 3×3 triangle worked by hand. It also checks it against chainladder's
factors and ultimates on the real triangle, with and without tail, to a
relative tolerance of 1e-10.

## Usage

Requires Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
python -m reserving        # writes outputs/
pytest                     # 34 tests
ruff check src tests
ruff format --check src tests
```

## Known issue

chainladder 0.10.1 triggers a numpy 2.5 `DeprecationWarning` in
`chainladder/core/base.py` (adding a bare integer to a `datetime64` array).
The results are unaffected. pytest treats all other warnings as errors and
ignores only this one, and `python -m reserving` hides it. If a future numpy
turns it into an error, pin `numpy<2.5` or upgrade chainladder.

## Layout

```
data/            input CSV
src/reserving/   config, data, diagnostics, methods, manual, summary, plots, pipeline
tests/           pytest suite
outputs/         generated tables and figures (git-ignored)
```
