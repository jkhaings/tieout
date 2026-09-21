# SESSION.md — work done after the build, with the evidence for each claim

Kept in the same spirit as `docs/BUILDLOG.md`: what changed, what was actually
measured, and what is still unproven.

## Production workbook audit

An independent audit of two workbooks generated in production (META, MCD)
returned six defects. Every one reproduced locally against the real filings
before anything was changed, and the reproduction is what several of the fixes
were designed against — two of the audit's own premises turned out to be wrong,
and one design decision made on paper was overturned by the data.

The six cluster into one theme: **the pipeline's fail-closed promises were
honoured in the data layer and lost at the presentation layer.** A missing fact
became `"n/r"` text in a statement cell, and a ratio formula divided by that
text. A refused narration became a `Commentary(text=None)`, and the sheet writer
dropped it. A `$1` tolerance meant for float representation was applied to facts
filed in millions. In each case the deterministic layer knew exactly what was
wrong and the workbook said nothing, or said `#VALUE!`.

### Reproduction, before any change

Running `build_statements` + `verify` against live companyfacts for both
tickers reproduced the audit exactly:

```
MCD  fiscal_years=[2021..2025]  report passed=False  checks=5
  [FAIL] insufficient_data_2021
  [PASS] cash_roll_forward_2022 / 2023
  [FAIL] cash_roll_forward_2024   diff=+1,000,000
  [FAIL] cash_roll_forward_2025   diff=-1,000,000
  total_liabilities [None x5]   pretax_income [None x5]   total_equity all negative
META fiscal_years=[2021..2025]  report passed=True   checks=14
  gross_profit [None x5]   inventory [None x5]   ppe_net [None x5]
```

That accounts for the audit's "10 formula errors per workbook" precisely: MCD
loses 5 gross-margin cells (no `GrossProfit`) and 5 debt-to-equity cells (no
`Liabilities`); META loses 5 gross-margin and 5 quick-ratio cells (no
`InventoryNet`). Nothing was taken on faith from the audit report.

### Two premises in the audit that the data contradicted

- **The pretax tag it asked for was already there.** The brief named
  `IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments`
  as a tag to add; it has been in the `pretax_income` chain since the original
  build. MCD's real problem is different and worse: it files **no consolidated
  pretax element at all**, only the jurisdiction split
  (`...BeforeIncomeTaxesDomestic` and `...Foreign`). No tag chain can fix that.
- **The Commentary sheet is not in `app/model/excel.py`.** `build_workbook()`
  writes five sheets and has no `Commentary` parameter; the sheet is appended
  afterwards by `app/agent/workbook.py`, which is where the row-drop lived. The
  fix landed there, leaving `app/model` the deterministic lane it is.

---

### 1. Ratio formulas referencing non-numeric cells (`#VALUE!`)

**Root cause.** `_write_statement_sheet` registers a cell address into the
shared `CellRegistry` unconditionally, including for a cell holding the `"n/r"`
text it writes for an unreported value. The ratio writer's `ref()`/`safe_div()`
tested only that *an address exists*, never that the cell holds a number, so a
formula was emitted whenever the row existed at all. The module docstring
actually argued this was correct — that `#VALUE!` propagating out of an `"n/r"`
cell was a visible failure. It is visible, but it is not legible, and it put ten
error cells in each shipped workbook.

**Fix.** `_write_ratios_sheet` now consults the model's own values before
emitting anything. Each ratio declares the `(key, year)` inputs it would
reference, which of them are divided by, and which must be strictly positive. A
cell renders a formula only when every input is numeric; otherwise `"n/a"` (an
input is not reported) or `"n/m"` (inputs are present but the result is not
meaningful — a zero divisor, or return on equity and debt-to-equity for a filer
with non-positive equity, where the convention is to decline rather than print a
misleading negative). FCF margin divides by the FCF row on the Ratios sheet
itself, so it inherits that row's placeholder instead of deciding again. The
first year's revenue growth became `"n/a"` rather than blank: one rule — every
ratio cell is a formula, `n/a`, or `n/m`, never blank and never an error — is
easier to trust than two. The stale docstring paragraph was rewritten.

**Evidence.** `test_no_ratio_formula_references_a_non_numeric_cell` parses every
formula on the Ratios tab, resolves each cell reference back through the
workbook, and asserts the target is numeric — the mechanical proof of "zero
error-producing formulas" without opening Excel. It runs over all four fixtures.
Regenerated workbooks: **MCD 44 formulas, 0 error-producing, 6 `n/a` + 10 `n/m`;
META 49 formulas, 0 error-producing, 11 `n/a`.** The 10 `n/m` on MCD are exactly
the five ROE and five debt-to-equity cells its negative equity makes meaningless.

### 2. False FAILs from a flat $1 tolerance

**Root cause.** `_TOLERANCE = 1.0` was a single module constant, documented as
guarding float representation. MCD presents in millions, so each of the six
terms summed by the cash roll-forward carries up to $1,000,000 of rounding; the
identity was off by exactly one rounding unit and FAILed. A false alarm erodes
the product's claim as badly as a missed break.

By the time `verify()` runs all provenance is gone — `builder.py` keeps only
`rv.value`, and companyfacts carries no `decimals` field — so granularity is
inferred from the values themselves.

**Fix.** `_rounding_unit()` returns the coarsest of `(1e6, 1e5, 1e4, 1e3)` a
value is an exact multiple of, else `1.0`; units below $1,000 are deliberately
not candidates, since a whole-dollar figure ending in "00" is a coincidence, not
a reporting scale. `_infer_tolerance()` **sums each term's own unit**, floored at
$1. `_check()` takes its terms keyword-only and spells the arithmetic out in the
check's own description, so a reader seeing a $6,000,000 tolerance is told why.
An optional term that was not reported (`fx_effect_on_cash`,
`noncontrolling_interest_in_income`) contributes no rounding and does not count.

**A design decision the data overturned.** The design pass recommended taking
the *finest* unit across terms, as the tightest defensible inference. MCD FY2022
disproves it: `Assets` is filed to the nearest $1M and
`LiabilitiesAndStockholdersEquity` to the nearest $100k, and the two differ by
$400,000. The finest-unit rule allows $200,000 and false-FAILs it. Accumulated
rounding noise is bounded by the **coarsest** rounding applied to any term, never
the finest — hence the sum, which also degrades to "one unit per term" when every
term shares a scale, as the brief specified.

**The trade-off, stated plainly.** A wider tolerance can in principle mask a real
break; the largest undetectable error on a six-term, million-rounded check is now
$6M rather than $1. Three things bound that: the tolerance is derived from the
filed data rather than chosen, it is printed in the workbook next to the check,
and `test_fixture_checks_tie_exactly_not_merely_within_tolerance` asserts that
every scored check on AAPL, MSFT and META still ties **to the dollar** — measured,
every fixture check has `lhs - rhs == 0.0` exactly — so the wider tolerance can
never quietly become cover for a regression.

**Evidence.** MCD `cash_roll_forward_2024/2025` now PASS at a $6,000,000
tolerance against their $1,000,000 difference; the same data with $50M injected
into CFO still FAILs. `balance_sheet_equation_independent_2022` passes at
$1,100,000 ($1,000,000 + $100,000), the mixed-precision case above.

### 3. Empty Commentary tabs

This one had two independent causes, and the workbook-layer defect would have
hidden either.

**Root cause (a): MCD retrieved nothing, silently.** Reproduced locally —
MCD's run returned `0/10 line items have grounding chunks`. `parse_filing`
matches section headings of the form `Item <n>. <Title>`, but McDonald's 10-K
contains the string "Item 1A" **exactly once in the entire document**, in its
contents table; the section bodies are headed with the bare titles `RISK FACTORS`
and `MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF
OPERATIONS`. The contents-table line was the only candidate, the TOC-trap scoring
correctly discarded it, and the filing reduced to **2 chunks** against META's 95.
Nothing raised and nothing logged: the empty Commentary tab was the only symptom
this failure mode ever produced.

**Fix.** `_bare_heading_re()` matches a section title as an *entire line* —
tightened in a different direction from `_heading_re()`, since that is what
separates a heading from the many prose sentences opening with the same words
("Management's Discussion and Analysis ... is based upon the Company's
Consolidated Financial Statements..."). Both heading shapes now compete as
candidates and the existing largest-span-wins scoring arbitrates; the bare
boundary titles come along with them, so a contents-table candidate's span
shrinks to the next contents line rather than running to the end of the document.
Measured before and after on live filings: **MCD 2 → 69 chunks; AAPL 32 → 32,
MSFT 20 → 20, META 95 → 95, unchanged.**

**Root cause (b): the workbook dropped every refusal.**
`write_commentary_sheet` skipped any entry with `text is None`, so a run that
narrated nothing shipped a sheet holding only its headers — and a unit test
asserted that was correct. It is a defensible reading of CLAUDE.md rule 4 for a
single item and the wrong one for all of them: an absent row is not a fail-closed
signal, it is indistinguishable from a sheet that was never written. A second bug
sat next to it: `narrate()` `continue`d when a curated line item was absent from
the filer's statements, dropping it from `commentary` entirely — exactly MCD's
`gross_profit`.

**Fix.** Every attempted line item now gets exactly one row: its commentary, or
`"no grounded commentary available: <reason>"`. The reasons are deterministic
facts about the run recorded in `PipelineState.commentary_refusals` — never model
output — distinguishing a missing API key, a line item the filer does not report,
a line item with no retrieved passage (naming the upstream cause), the circuit
breaker, a failed API call, and a draft that failed grounding validation twice.
An entirely empty `commentary` writes one row saying narration did not run.

**Also fixed: the silence itself.** Both ways a run could produce no commentary
were invisible in `docker logs` — the `narrator_client is None` branch logged
nothing, and a zero-section parse logged nothing. Both now emit a
`logger.warning`, so this class of failure never again has an empty spreadsheet
cell as its only evidence.

**Evidence.** `test_unnarrated_run_still_ships_a_populated_commentary_sheet`
drives the whole graph with no narrator client and asserts the generated
workbook's Commentary sheet has one row per curated line item, each explaining
itself — the regression that would have caught both shipped workbooks.
Regenerated: **MCD 10/10 line items present (3 narrated, 7 explicit refusals);
META 10/10 (8 narrated, 2 refusals).**

**Prod evidence: requested, not yet in hand.** This machine cannot reach the
droplet — no SSH config, no `known_hosts` entry, no IP, and `docs/DEPLOY.md`
names only the hostname. A diagnostic block (the `runs` rows for both tickers, an
`ANTHROPIC_API_KEY` presence check on `/opt/tieout/.env`, and a `docker logs`
grep) was handed over to be run by hand. Cause (a) is confirmed independently of
it, because it reproduces locally. **META's** empty tab is *not* explained by
anything in this repo — META narrates 8/10 locally on the same code — so its
cause is prod-side: either an empty `ANTHROPIC_API_KEY` in `/opt/tieout/.env`, or
a stale EDGAR cache (`docs/DEPLOY.md` records that the cache never expires). The
run log tells them apart: `narrate_ok=0` with `narrate failed for` lines means a
bad key, `narrate_ok=0` with no narration lines at all means an unset one. This
section should be amended once that output exists.

### 4. MCD had zero balance-sheet checks in all five years

**Root cause.** MCD files no `us-gaap:Liabilities` — only
`LiabilitiesAndStockholdersEquity` and `LiabilitiesCurrent`. Total liabilities
was blank in every year, so the balance-sheet equation had no third term and was
never scored once, and debt-to-equity had nothing to divide.

**Fix.** A small declarative derivation mechanism, which did not exist before:
`builder.py` fills `total_liabilities` from `L&SE − total equity` and records
`derived:LiabilitiesAndStockholdersEquity-StockholdersEquity` in `xbrl_tags`
(frozen `app/schemas.py` untouched — the marker rides in the existing field).

**The trap this nearly walked into.** `LiabilitiesAndStockholdersEquity` equals
liabilities plus equity *including* noncontrolling interests, but the
`total_equity` chain prefers parent-only `StockholdersEquity`. For a filer with
NCI that tags parent-only equity, `L&SE − equity` is `liabilities + NCI` —
overstating liabilities, and overstating debt-to-equity with it. Worse, the same
fix swaps that filer's balance-sheet check to `Assets vs L&SE`, which ties
regardless, so **no scored check could ever catch the error**. The derivation
therefore refuses outright when the filer reports `MinorityInterest` and equity
resolved to the parent-only tag. MCD reports no `MinorityInterest` at all, so its
derivation is exact; the refusal is insurance for the general case, and it is
tested.

**Honest scoring.** With liabilities derived, `assets = liabilities + equity`
reduces algebraically to `assets = L&SE` — a check that cannot fail proves
nothing. `verify()` tests `is_derived` first and emits a distinct
`balance_sheet_equation_independent_{fy}` comparing `Assets` against the filed
`LiabilitiesAndStockholdersEquity`, two independently filed facts, with a
description saying so. Filers that do file `Liabilities` keep the stronger
three-fact check. `test_circular_balance_sheet_check_is_never_scored` rigs the
three-term identity to pass while the two filed facts disagree and asserts the
report still fails.

Derived values are marked where they are read, not only in `xbrl_tags` (which
nothing renders): the statement row reads "Total liabilities (derived)".

### 5. Missing tag variants

- **MCD pretax income.** No consolidated element exists, so the fallback-chain
  fix the brief asked for was impossible. Instead `pretax_income` is derived as
  `domestic + foreign`. This is deliberately distinct from the component-summing
  `tags.py` refuses to do for SG&A: domestic and foreign are an exhaustive
  partition, not a never-complete component list, and the result is immediately
  re-checked by the `net_income_buildup` identity it enables. Verified against
  MCD FY2020–2025: the sum equals net income plus tax expense exactly in five of
  six years and is $1,000,000 off in the sixth (FY2022 rounding). MCD now scores
  `net_income_buildup` in all five presented years, and `insufficient_data_2021`
  is gone.
- **META PP&E.** META files **zero** annual `PropertyPlantAndEquipmentNet`
  facts; it reports only
  `PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization`.
  Added strictly as a fallback *after* the plain tag, never ahead of it — it
  folds finance-lease right-of-use assets into PP&E and must not displace the
  narrower concept for a filer reporting both. `test_broader_ppe_element_is_only_ever_a_fallback`
  pins the ordering; AAPL and MCD still resolve to `PropertyPlantAndEquipmentNet`.

### 6. Share counts on two different scales

**Root cause.** Filers tag share counts at whatever scale their income statement
presents. MCD files `751.8` and META `2,574,000,000` for the same concept, both
under the `shares` unit. Both are faithful to the filing; rendering them side by
side without saying which is which is not.

**Fix, display-only.** `LineItem.values` keeps the as-filed value —
`evals/datasets/tieout_gold.json` scores generated cells against values derived
independently from raw companyfacts, so rescaling the model would have broken the
tie-out eval. `share_filing_scale()` decides the filing scale from the filer's own
arithmetic: net income divided by diluted EPS is the share count the filer itself
implies, and the candidate scale (units, thousands, millions) whose product lands
nearest it wins, by majority vote across the years carrying EPS evidence. The
margin is not close — measured, the implied count is within 0.07% of the filed
one, against candidates 1,000× apart. Without EPS evidence it falls back to
magnitude, where thousands is deliberately unreachable so a genuinely small
filer's real count can never be misread by 1,000×.

The statement tab renders every `shares`-unit row in millions and says so in the
label; the header note now reads "...; share counts in millions."
`app/agent/formatting.py` uses the same helper, so narration cannot say
"751.80 shares" while the workbook says 751.8 million. Nothing else consumes the
share count, and `ref()` in the ratio writer now asserts that no formula
references a display-scaled cell, so a future ratio cannot silently consume one.

**The EPS tie, honestly.** With shares in millions and net income in dollars,
`net income / shares` is no longer clickable cell-to-cell; the reader applies the
stated unit, which is the standard "in millions except per share" convention the
label and header note both declare. The tie is preserved where it matters — the
displayed count is the filer's own EPS-implied count divided by exactly 1e6,
verified against EPS rather than guessed. Rendering actual shares instead would
make the division literal at the cost of a 13-digit column; it is the same
function with a different divisor constant.

**Evidence.** All four filers now render on one scale: AAPL 15,004.7, MSFT
7,453.0, MCD 716.4, META 2,574.0, every row labelled "(millions)".

---

## Fixtures

`tests/fixtures/make_fixtures.py` now generates MCD and META alongside AAPL and
MSFT, so every claim above is pinned by a hermetic test against real filed data
rather than a synthetic reconstruction. Its tag allow-list had to grow beyond
`CANONICAL`: the two pretax jurisdiction components and `MinorityInterest` are
read by the builder but are not canonical line items, and trimming them out would
have silently disabled both derivations in the test suite. AAPL and MSFT fixtures
were regenerated in the same pass; `evals/tieout_eval.py` still reports **430/430
cells correct**, confirming no existing value moved.

## Gates

`make test` 334 passed (was 284), `make lint` and `make type` green. `make evals`:
tie-out 430/430, judge grounded 100%, no invented numbers 100%, cited 85.7%.

## Observations left alone, deliberately

- **MSFT's Item 1A extracts to a single chunk**, unchanged by this session's
  ingest work and present before it. It looks wrong for a filer whose risk
  factors run for pages, but it is a separate investigation and nothing in the
  audit pointed at it. Recorded here so it is not lost.
- **Per-year derivation provenance.** `LineItem.xbrl_tags` is item-level, so a
  derivation is all-or-nothing: a filer reporting `Liabilities` for only some
  presented years keeps its real gaps rather than being partly derived, because
  the item could not then describe itself honestly. Fixing this properly needs
  per-year provenance, which the frozen `app/schemas.py` does not carry.
- **The over-length-ticker gap** recorded in `docs/DEPLOY.md` is still open;
  `app/api` was out of scope here.
