# WRDS Data Requirements

last_updated: 2026-07-07T04:36:37Z
repo: /Volumes/Storage/Projects/fjs/repo
scope: analysis and planning only; no raw WRDS data downloaded

## Executive Summary

Current repo reproduction is data-light but data-sensitive. The active research
pipeline is built around:

- `data/returns_daily.csv`: CRSP-derived daily equity returns, registered with
  sha256 `96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`,
  892,529 rows, 300 columns, 2010-01-05 to 2024-12-31.
- `data/factors/ff5mom_daily.csv`: daily FF5 plus momentum factors, registered
  with sha256 `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`,
  2005-01-03 to 2025-08-29.
- Current claim boundaries: daily DoW evidence is empirical-only; weekly/oneway
  detector validation remains blocked by flat-zero injection sensitivity; no
  capped or comparison-invalid run should be treated as headline evidence.

Bottom line for the admin/user: the repo has enough local data to reproduce the
current registered CSV-backed runs if the existing files are trusted, and the
shared WRDS vault appears to contain the main P0/P1 raw backbone for a stronger
rebuild audit: CRSP daily returns, names, delistings, link tables, Compustat,
FF factors, and OptionMetrics. It does not yet have every optimal future layer
locally staged. IBES, TAQ, 13F/Thomson, CIQ event/index layers, Audit Analytics,
BoardEx, ExecComp, TRACE/MSRB, and some CBOE-style derivatives/borrow data are
catalog-confirmed as WRDS-available but should be treated as future-tier data
until a specific research ticket needs them.

Catalog grounding used here:

- `/Volumes/Storage/Data/WRDS/_catalog/20260706_223405_tables.csv`
- `/Volumes/Storage/Data/WRDS/_catalog/20260706_223405_columns.csv`
- `/Volumes/Storage/Data/WRDS/_catalog/20260706_223405_full_catalog.json`

Local vault evidence found during this audit:

- `/Volumes/Storage/Data/WRDS/raw/crsp/`: CRSP daily stock file partitions,
  names, delistings, CCM link tables, and index/list files are present.
- `/Volumes/Storage/Data/WRDS/raw/comp/`: Compustat fundamentals, quarterly
  fundamentals, and company data are present.
- `/Volumes/Storage/Data/WRDS/raw/ff/`: daily factor partitions are present.
- `/Volumes/Storage/Data/WRDS/raw/optionm/`: OptionMetrics security prices,
  option quotes, zero curves, distributions, names, and large yearly option
  partitions are present for the modern research window.

## Human Priority Map

P0 blocking data should be frozen first because it controls exact reproduction
and claim validation. That means CRSP `dsf`, CRSP `dsenames`, FF daily factor
tables, and delisting/corporate-action support. P1 then makes the project
research-grade: monthly CRSP, index/benchmark returns, S&P membership, CIZ v2
cross-checks, CRSP-Compustat links, Compustat fundamentals, industry mappings,
and liquidity/portfolio factor families. P2 and P3 add stronger diagnostics and
future model hypotheses: short interest, borrow costs, OptionMetrics, IBES,
CIQ events, TAQ, holdings, governance, restatements, and bond-market stress
controls.

The strongest near-term recommendation is not to download more raw data first.
Instead:

1. Rebuild or audit the current CRSP/FF extracts from the shared WRDS vault and
   confirm hashes, filters, and whether delisting returns are intentionally
   excluded or incorporated.
2. Add a point-in-time data provenance note for the current 300-asset universe:
   identifier, exchange/share-code filter, market-cap/liquidity ranking date,
   and date validity windows.
3. Only after the flat-zero detector issue is resolved, stage P1/P2 expansions
   as specific research tickets with tight manifests rather than broad dumps.

```yaml
project:
  name: "fjs-dealias-portfolio"
  repo_path: "/Volumes/Storage/Projects/fjs/repo"
  analysis_timestamp_utc: "2026-07-07T04:36:37Z"
  purpose: "Research codebase for testing whether an FJS/MANOVA-style de-aliasing overlay improves covariance forecasts and downstream portfolio risk decisions on balanced equity panels."
  current_claims_or_results_summary: "Current registered results use CRSP-derived daily returns and FF5+MOM daily factors. T-012 daily DoW evidence is recovered and scientifically useful but not cleanly ratified; daily DoW is empirical-only and not detector validation. Weekly/oneway detector validation remains blocked by flat-zero injection sensitivity. Capped, truncated, or comparison-invalid runs are not headline evidence."
  data_policy_notes:
    raw_wrds_must_not_enter_repo: true
    shared_vault_root: "/Volumes/Storage/Data/WRDS"

requirements:
  - requirement_id: "REQ_P0_CRSP_DSF_DAILY_RETURNS"
    priority: "P0_BLOCKING"
    project_need: "Reproduce the registered daily returns file and all daily/weekly equity-panel experiments that start from daily returns."
    quant_use_case: "Primary equity return panel, rolling covariance estimation, realized-risk holdouts, daily DoW/week/vol/dowxvol grouping, weekly balanced panels, universe construction, liquidity filters."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "crsp"
    wrds_table: "dsf"
    wrds_product_or_library: "CRSP US Stock Database, daily stock file"
    logical_dataset: "US common-share daily returns, prices, volume, shares outstanding, adjustment factors"
    required_for_current_reproduction: true
    required_for_current_claim_validation: true
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-01-01 to 2024-12-31, matching data/registry.json"
      optimal: "1925-present for full CRSP history; 2005-present for modern research and OptionMetrics-aligned expansions"
      tier_split: "modern_research"
    frequency: "daily"
    expected_size_class: "huge"
    partition_strategy: "year"
    key_columns:
      identifiers: ["permno", "permco", "cusip"]
      dates: ["date"]
      measures: ["ret", "retx", "prc", "vol", "shrout", "cfacpr", "cfacshr", "openprc", "bid", "ask", "numtrd", "hexcd", "hsiccd"]
    joins:
      required_link_tables: ["crsp.dsenames", "crsp.dsedelist", "crsp.ccmxpf_linktable or crsp.ccmxpf_lnkhist for fundamentals"]
      joins_to_project_data: ["data/returns_daily.csv", "data/meta/universe_2016_2024.json", "experiments/eval/run.py --returns-csv", "experiments/equity_panel/run.py inputs"]
      point_in_time_risks: ["Use PERMNO as the stable security key; ticker is not stable.", "Universe filters must be evaluated as of each date, not from final survivor lists."]
    bias_risks:
      survivorship: "High if universe is reconstructed from surviving tickers only; use names and delisting files."
      lookahead: "High if top-p liquidity or market-cap ranking uses future data outside the pre-period."
      restatement: "Low for CRSP market data, but CRSP vintages can change; record snapshot date."
      delisting: "High if delisting returns are omitted when claiming total-return robustness."
      corporate_actions: "High if cfacpr/cfacshr are mishandled or adjusted/unadjusted prices are mixed."
    notes: "Current registry says the canonical CSV came from crsp.dsf with exchanges [1,2,3], share codes [10,11], min_price 1, min_volume 0, 2010-01-01 to 2024-12-31. Local shared vault contains CRSP dsf partitions including 2005-2024 parquet files and a full-history partitioned snapshot."

  - requirement_id: "REQ_P0_CRSP_DAILY_NAMES_AND_UNIVERSE_FILTERS"
    priority: "P0_BLOCKING"
    project_need: "Reproduce exchange/share-code filters, ticker labels, and canonical security names used in the current returns extract."
    quant_use_case: "Survivorship-bias control, ticker-to-PERMNO mapping, common-share filters, exchange filters, sector/industry controls."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "crsp"
    wrds_table: "dsenames, stocknames_v2, wrds_names_query"
    wrds_product_or_library: "CRSP security name history"
    logical_dataset: "Point-in-time security identity and listing metadata"
    required_for_current_reproduction: true
    required_for_current_claim_validation: true
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-01-01 to 2024-12-31"
      optimal: "full CRSP coverage"
      tier_split: "full_history"
    frequency: "event"
    expected_size_class: "small"
    partition_strategy: "table"
    key_columns:
      identifiers: ["permno", "permco", "ticker", "cusip", "ncusip", "shrcd", "exchcd", "siccd", "naics"]
      dates: ["namedt", "nameendt", "securitybegdt", "securityenddt"]
      measures: ["comnam", "primexch", "trdstat", "secstat"]
    joins:
      required_link_tables: ["crsp.dsf", "crsp.dsedelist"]
      joins_to_project_data: ["src/io/crsp_daily.py lateral name join", "data/meta/universe_2016_2024.json"]
      point_in_time_risks: ["Join on permno with namedt/nameendt windows.", "Never backfill final ticker names across historical windows."]
    bias_risks:
      survivorship: "High if only current names are used."
      lookahead: "Medium if name windows are ignored."
      restatement: "Low, but vendor corrections can alter historical name intervals."
      delisting: "Names do not include returns from delisting; join delisting files separately."
      corporate_actions: "CUSIP/ticker changes around corporate actions require date-bounded joins."
    notes: "Current src/io/crsp_daily.py joins crsp.dsf to crsp.dsenames using date-bounded name intervals and filters share codes and exchanges from the joined name record."

  - requirement_id: "REQ_P0_FF5_MOM_DAILY_FACTORS"
    priority: "P0_BLOCKING"
    project_need: "Reproduce prewhitened runs and factor-baseline ablations that use the registered FF5+MOM daily factor file."
    quant_use_case: "Factor prewhitening, factor covariance baselines, residual covariance diagnostics, market/regime decomposition, benchmark controls."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "ff"
    wrds_table: "fivefactors_daily, factors_daily"
    wrds_product_or_library: "Fama-French factors on WRDS"
    logical_dataset: "Daily market, size, value, profitability, investment, momentum, and risk-free factors"
    required_for_current_reproduction: true
    required_for_current_claim_validation: true
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2005-01-03 to 2025-08-29, matching data/factors/registry.json"
      optimal: "full available daily factor history"
      tier_split: "full_history"
    frequency: "daily"
    expected_size_class: "tiny"
    partition_strategy: "year"
    key_columns:
      identifiers: []
      dates: ["date"]
      measures: ["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]
    joins:
      required_link_tables: []
      joins_to_project_data: ["data/factors/ff5mom_daily.csv", "experiments/prewhiten.py", "src/finance/factors.py", "experiments/eval/run.py --factors-csv"]
      point_in_time_risks: ["Align by trading date; avoid filling missing factor dates across holidays without explicit policy."]
    bias_risks:
      survivorship: "Low for published factor series, but factor construction itself embeds CRSP/Compustat universe choices."
      lookahead: "Low for daily factors when aligned by date; higher if using revised factor vintages without recording vintage."
      restatement: "Medium because factor vendor vintages can be revised; record WRDS/Ken French vintage."
      delisting: "Indirectly depends on factor construction; not a substitute for security-level delisting control."
      corporate_actions: "Handled inside factor construction; still record source vintage."
    notes: "Catalog confirms ff.fivefactors_daily has date,mktrf,smb,hml,rmw,cma,rf,umd and ff.factors_daily has date,mktrf,smb,hml,rf,umd. Local raw FF factor partitions are present in the shared vault."

  - requirement_id: "REQ_P0_CRSP_DELISTINGS_DISTRIBUTIONS_ACTIONS"
    priority: "P0_BLOCKING"
    project_need: "Validate that current return claims are not distorted by delisting-return omission or corporate-action mistakes."
    quant_use_case: "Total-return validation, delisting-bias audit, adjusted-return cross-check, corporate-action correctness."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "crsp"
    wrds_table: "dsedelist, dsedist, dividends, stkdlycumulativeadjfactor, stkdistributions"
    wrds_product_or_library: "CRSP daily event, delisting, distribution, and adjustment files"
    logical_dataset: "Security-level delisting returns and corporate-action adjustments"
    required_for_current_reproduction: false
    required_for_current_claim_validation: true
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-01-01 to 2024-12-31"
      optimal: "full CRSP coverage"
      tier_split: "full_history"
    frequency: "event"
    expected_size_class: "medium"
    partition_strategy: "table"
    key_columns:
      identifiers: ["permno", "permco", "cusip"]
      dates: ["dlstdt", "disexdt", "caldt", "dlycaldt"]
      measures: ["dlret", "dlretx", "dlprc", "dlstcd", "cfacpr", "cfacshr", "dlycumfacpr", "dlycumfacshr", "divamt"]
    joins:
      required_link_tables: ["crsp.dsf", "crsp.dsenames"]
      joins_to_project_data: ["data/returns_daily.csv validation audit", "future total-return rebuild scripts"]
      point_in_time_risks: ["Delisting return needs event-date handling and a documented convention for forward realized-risk windows that cross delistings."]
    bias_risks:
      survivorship: "Very high if failed/delisted securities disappear from panels without explicit treatment."
      lookahead: "Medium if delisting outcomes are used in universe selection before the delisting date."
      restatement: "Low to medium because CRSP event corrections can change."
      delisting: "Primary control table for this risk."
      corporate_actions: "Primary control table family for splits, distributions, and adjusted-price consistency."
    notes: "Local shared vault contains dsedelist.parquet and dsenames.parquet plus partitioned CRSP snapshots. P0 because current claims should be validated against delisting/corporate-action treatment even if exact current CSV reproduction used dsf ret only."

  - requirement_id: "REQ_P1_CRSP_DAILY_INDEX_RETURNS"
    priority: "P1_CORE"
    project_need: "Add market-wide benchmark and regime controls for covariance forecast interpretation."
    quant_use_case: "Benchmark comparisons, crisis slicing, market-regime labels, excess-return conversion, sanity checks against S&P/CRSP index moves."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "crsp"
    wrds_table: "wrds_dailyindexret_query, dsp500, dsp500p, crsp_ziman_daily_index"
    wrds_product_or_library: "CRSP index files"
    logical_dataset: "Daily CRSP value-weighted/equal-weighted index returns and S&P index returns"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-01-01 to 2024-12-31"
      optimal: "full CRSP index history"
      tier_split: "full_history"
    frequency: "daily"
    expected_size_class: "small"
    partition_strategy: "year"
    key_columns:
      identifiers: []
      dates: ["dlycaldt", "caldt"]
      measures: ["vwretd", "vwretx", "ewretd", "ewretx", "sprtrn", "spindx", "vwtotval", "ewtotcnt"]
    joins:
      required_link_tables: []
      joins_to_project_data: ["future benchmark tables", "crisis/regime diagnostics", "summary_perf.csv context"]
      point_in_time_risks: ["Align by trading date; do not use future index composition for ex ante universe selection."]
    bias_risks:
      survivorship: "Low for aggregate index returns, but composition-based comparisons need membership files."
      lookahead: "Low for index returns; higher for derived regime thresholds if estimated on full sample."
      restatement: "Low to medium due vendor corrections."
      delisting: "Index construction embeds delisting/vendor methodology."
      corporate_actions: "Handled by index construction."
    notes: "Catalog confirms crsp.wrds_dailyindexret_query columns dlycaldt, vwretd, ewretd, sprtrn, spindx and year partition recommendation."

  - requirement_id: "REQ_P1_CRSP_SP500_MEMBERSHIP_AND_INDEX"
    priority: "P1_CORE"
    project_need: "Construct transparent S&P 500 benchmark universes and compare current liquid-common-stock universe against index membership."
    quant_use_case: "Benchmark universe, survivorship-bias control, sector-neutral holdout, advisor-friendly comparisons."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "crsp, crsp_a_indexes"
    wrds_table: "dsp500list, msp500list, dsp500, msp500, stkindmembership_ind"
    wrds_product_or_library: "CRSP S&P index membership and returns"
    logical_dataset: "S&P 500 membership intervals and daily/monthly index returns"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-01-01 to 2024-12-31"
      optimal: "full available membership history"
      tier_split: "full_history"
    frequency: "mixed"
    expected_size_class: "small"
    partition_strategy: "table"
    key_columns:
      identifiers: ["permno"]
      dates: ["start", "ending", "mbrstartdt", "mbrenddt", "caldt"]
      measures: ["vwretd", "ewretd", "sprtrn", "spindx"]
    joins:
      required_link_tables: ["crsp.dsf", "crsp.dsenames"]
      joins_to_project_data: ["future S&P 500 benchmark universe", "future paper-v1 universe sensitivity"]
      point_in_time_risks: ["Membership intervals must be date-bounded; current constituents cannot be backfilled."]
    bias_risks:
      survivorship: "High if using current S&P members rather than historical membership intervals."
      lookahead: "High if membership changes are known before effective date."
      restatement: "Low to medium from vendor corrections."
      delisting: "Delisted/deleted constituents must stay in historical panels through exit dates."
      corporate_actions: "Use CRSP PERMNO and names for action-safe joins."
    notes: "Useful once the repo needs an advisor-readable benchmark universe distinct from the current top-liquid CRSP common-share panel."

  - requirement_id: "REQ_P1_CRSP_MONTHLY_STOCK_FILE"
    priority: "P1_CORE"
    project_need: "Support lower-frequency baselines, long-history robustness, and monthly publication-style factor comparisons."
    quant_use_case: "Monthly covariance baselines, long-window robustness, factor model checks, easier Compustat alignment."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "crsp"
    wrds_table: "msf, msenames, msedelist, wrds_monthlyindexret_query"
    wrds_product_or_library: "CRSP monthly stock and index files"
    logical_dataset: "Monthly security returns, names, delistings, and index returns"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2000-present for modern monthly robustness"
      optimal: "full CRSP monthly history"
      tier_split: "full_history"
    frequency: "monthly"
    expected_size_class: "large"
    partition_strategy: "year"
    key_columns:
      identifiers: ["permno", "permco", "cusip"]
      dates: ["date", "namedt", "nameendt", "dlstdt"]
      measures: ["ret", "retx", "prc", "vol", "shrout", "cfacpr", "cfacshr", "dlret", "vwretd", "ewretd"]
    joins:
      required_link_tables: ["crsp.msenames", "crsp.msedelist", "crsp.ccmxpf_linktable"]
      joins_to_project_data: ["future monthly baseline artifacts", "factor and Compustat validation"]
      point_in_time_risks: ["Monthly date alignment with fundamentals must respect reporting lags."]
    bias_risks:
      survivorship: "High without names/delistings."
      lookahead: "Medium when merged with fundamentals."
      restatement: "Low to medium."
      delisting: "Use msedelist for monthly total-return validation."
      corporate_actions: "Use CRSP adjustment factors."
    notes: "Not needed for current daily results, but foundational for serious quant finance robustness and publication comparisons."

  - requirement_id: "REQ_P1_CRSP_CIZ_V2_CROSSCHECKS"
    priority: "P1_CORE"
    project_need: "Cross-check legacy CRSP daily fields against newer CIZ-style daily security data and richer security status fields."
    quant_use_case: "Data-integrity audit, corporate-action validation, modern CRSP migration, trading-status filters, CIZ/SIZ consistency checks."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "crsp"
    wrds_table: "dsf_v2, wrds_dsfv2_query, stocknames_v2, stkdelists, stkdlysecuritydata, stkdlycumulativeadjfactor"
    wrds_product_or_library: "CRSP CIZ/v2 daily stock data"
    logical_dataset: "Modern CRSP daily security data and security information history"
    required_for_current_reproduction: false
    required_for_current_claim_validation: true
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-01-01 to 2024-12-31"
      optimal: "full v2/CIZ coverage"
      tier_split: "modern_research"
    frequency: "daily"
    expected_size_class: "huge"
    partition_strategy: "year"
    key_columns:
      identifiers: ["permno", "permco", "cusip", "hdrcusip", "ticker"]
      dates: ["dlycaldt", "namedt", "nameenddt", "delistingdt"]
      measures: ["dlyret", "dlyretx", "dlyprc", "dlycap", "dlyvol", "dlycumfacpr", "dlycumfacshr", "tradingstatusflg", "sharetype", "securitytype"]
    joins:
      required_link_tables: ["crsp.stocknames_v2", "crsp.stkdelists"]
      joins_to_project_data: ["future data audit comparing legacy data/returns_daily.csv against CIZ/v2 rebuild"]
      point_in_time_risks: ["Need clear migration rules if replacing legacy dsf with v2 fields."]
    bias_risks:
      survivorship: "Lower if using v2 security history correctly; still high if current names are backfilled."
      lookahead: "Medium if trading status is used after the fact."
      restatement: "Medium due CRSP migration/version differences."
      delisting: "Use stkdelists/v2 delisting fields."
      corporate_actions: "Excellent cross-check source for adjustment factor handling."
    notes: "Catalog confirms crsp.dsf_v2 with dlyret, dlyretx, dlyprc, dlycap, dlyvol, dlycumfacpr, dlycumfacshr, trading status, and richer identity fields."

  - requirement_id: "REQ_P1_CRSP_COMPUSTAT_LINKS"
    priority: "P1_CORE"
    project_need: "Enable point-in-time joins from security returns to Compustat fundamentals and company metadata."
    quant_use_case: "Fundamental controls, factor construction, sector/industry conditioning, benchmark models, size/value/profitability/investment features."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "crsp"
    wrds_table: "ccmxpf_linktable, ccmxpf_lnkhist"
    wrds_product_or_library: "CRSP/Compustat Merged link tables"
    logical_dataset: "PERMNO/PERMCO to GVKEY link history"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-01-01 to 2024-12-31"
      optimal: "full link history"
      tier_split: "full_history"
    frequency: "event"
    expected_size_class: "small"
    partition_strategy: "table"
    key_columns:
      identifiers: ["gvkey", "lpermno", "lpermco", "liid", "linktype", "linkprim", "usedflag"]
      dates: ["linkdt", "linkenddt"]
      measures: []
    joins:
      required_link_tables: ["crsp.dsf", "comp.funda", "comp.fundq", "comp.company"]
      joins_to_project_data: ["future fundamental feature store", "future factor baseline feature joins"]
      point_in_time_risks: ["Use linkdt/linkenddt windows and acceptable linktype/linkprim filters; do not use stale or future links."]
    bias_risks:
      survivorship: "High if Compustat-only survivor company universe drives security selection."
      lookahead: "High if links are not date-bounded."
      restatement: "Medium when joined to restated fundamentals."
      delisting: "Link history must preserve firms that delist."
      corporate_actions: "Share-class and link-priority choices can alter mapping around mergers/spinoffs."
    notes: "Local shared vault contains ccmxpf_linktable.parquet and ccmxpf_lnkhist.parquet."

  - requirement_id: "REQ_P1_COMPUSTAT_FUNDAMENTALS"
    priority: "P1_CORE"
    project_need: "Add fundamental baselines and controls needed for serious publication-grade quant finance comparisons."
    quant_use_case: "Size/value/profitability/investment controls, industry/sector conditioning, fundamental factor baselines, quality/leverage/liquidity features."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "comp"
    wrds_table: "funda, fundq, company, co_industry, security"
    wrds_product_or_library: "Compustat North America fundamentals"
    logical_dataset: "Annual and quarterly accounting fundamentals plus company/security metadata"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2000-present for modern research"
      optimal: "1950-present annual and full quarterly availability"
      tier_split: "modern_research"
    frequency: "mixed"
    expected_size_class: "medium"
    partition_strategy: "year"
    key_columns:
      identifiers: ["gvkey", "iid", "cusip", "tic", "cik"]
      dates: ["datadate", "fyear", "fyearq", "fqtr", "rdq", "ipodate", "dldte"]
      measures: ["at", "ceq", "seq", "sale", "ni", "revt", "csho", "prcc_f", "prccq", "dltt", "dlc", "xrd", "capx", "che", "mkvalt", "gsector", "gind", "gsubind", "naics", "sic"]
    joins:
      required_link_tables: ["crsp.ccmxpf_linktable", "crsp.ccmxpf_lnkhist"]
      joins_to_project_data: ["future factor baselines", "future cross-sectional controls", "future sector-neutral tests"]
      point_in_time_risks: ["Use reporting-date lag policy; avoid using restated fundamentals before availability dates."]
    bias_risks:
      survivorship: "High if only currently active companies are retained."
      lookahead: "Very high without reporting-lag and data-availability policy."
      restatement: "High; Compustat values are often restated unless point-in-time products are used."
      delisting: "Company dldte and CRSP delistings must be handled jointly."
      corporate_actions: "Share counts and identifiers around mergers/spinoffs require link-table discipline."
    notes: "Local shared vault contains comp.funda, comp.fundq, and comp.company partitions through modern years."

  - requirement_id: "REQ_P1_FF_PORTFOLIOS_INDUSTRY_LIQUIDITY"
    priority: "P1_CORE"
    project_need: "Improve baseline comparisons and regime diagnostics beyond FF5+MOM."
    quant_use_case: "Industry controls, benchmark portfolios, liquidity-regime controls, validation against canonical portfolio return series."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "ff"
    wrds_table: "industry12, industry48, portfolios, portfolios25, portfolios_d, liq_ps, liq_sadka, factors_monthly, fivefactors_monthly"
    wrds_product_or_library: "Fama-French and liquidity factor libraries on WRDS"
    logical_dataset: "Industry definitions, size/BM portfolio returns, and liquidity factor series"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present"
      optimal: "full available history"
      tier_split: "full_history"
    frequency: "mixed"
    expected_size_class: "small"
    partition_strategy: "table"
    key_columns:
      identifiers: ["ffindnumber", "indtypeabbr"]
      dates: ["date", "yearmm"]
      measures: ["portfolio returns", "nfirms", "msize", "ps_level", "ps_innov", "ps_vwf", "sadka_tf", "sadka_pv"]
    joins:
      required_link_tables: ["crsp.dsenames or Compustat sic/naics for industry mapping when using security-level panels"]
      joins_to_project_data: ["future factor baselines", "future liquidity regime labels", "paper robustness tables"]
      point_in_time_risks: ["Industry mappings from SIC/NAICS should be date-bounded when joined to securities."]
    bias_risks:
      survivorship: "Low for published factors; security-level industry joins can reintroduce survivor bias."
      lookahead: "Medium if liquidity regimes are estimated using full-sample quantiles."
      restatement: "Medium because factor vintages can change."
      delisting: "Published factor methodology handles this, but security-level work still needs CRSP delistings."
      corporate_actions: "Handled by factor provider for portfolio returns."
    notes: "Catalog confirms ff.liq_ps, ff.liq_sadka, ff.industry48, and daily/monthly portfolio/factor tables."

  - requirement_id: "REQ_P1_CRSP_RISKFREE_AND_YIELDS"
    priority: "P1_CORE"
    project_need: "Support excess-return conversion and fixed-income regime controls without relying only on FF RF."
    quant_use_case: "Risk-free alignment, yield regime labels, interest-rate stress controls, robustness checks for factor and portfolio metrics."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "crsp"
    wrds_table: "riskfree, bmyield, bxyield"
    wrds_product_or_library: "CRSP Treasury/risk-free series"
    logical_dataset: "Risk-free and bond yield series"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present"
      optimal: "full available history"
      tier_split: "full_history"
    frequency: "mixed"
    expected_size_class: "small"
    partition_strategy: "table"
    key_columns:
      identifiers: ["crspid"]
      dates: ["qdate"]
      measures: ["bid_1", "ave_1", "ask_1", "dur_1", "bid_3", "ave_3", "ask_3", "yield", "retnua", "duratn"]
    joins:
      required_link_tables: []
      joins_to_project_data: ["future excess-return transforms", "future regime controls"]
      point_in_time_risks: ["Use date-available rates; avoid full-sample regime thresholds without train/test separation."]
    bias_risks:
      survivorship: "Low."
      lookahead: "Medium if regimes are estimated on full sample."
      restatement: "Low to medium."
      delisting: "Not applicable."
      corporate_actions: "Not applicable."
    notes: "Current factors file includes RF, but CRSP riskfree/yield tables give an independent source and richer rate controls."

  - requirement_id: "REQ_P2_COMPUSTAT_SHORT_INTEREST_AND_OWNERSHIP"
    priority: "P2_HIGH_VALUE"
    project_need: "Improve borrow/crowding diagnostics and short-side friction controls."
    quant_use_case: "Short interest, institutional ownership, borrow pressure proxies, crowded-trade diagnostics, liquidity filters."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "comp"
    wrds_table: "sec_shortint, sec_shortint_legacy, io_qholders"
    wrds_product_or_library: "Compustat North America daily/security supplemental tables"
    logical_dataset: "Short interest and institutional ownership snapshots"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present"
      optimal: "full available history"
      tier_split: "modern_research"
    frequency: "mixed"
    expected_size_class: "medium"
    partition_strategy: "date_range"
    key_columns:
      identifiers: ["gvkey", "iid", "ioid"]
      dates: ["datadate", "datadatep", "splitadjdate"]
      measures: ["shortint", "shortintadj", "iohhld", "iohhldp", "iohrank"]
    joins:
      required_link_tables: ["crsp.ccmxpf_linktable", "comp.security"]
      joins_to_project_data: ["future borrow/crowding controls", "future liquidity-risk diagnostics"]
      point_in_time_risks: ["Short-interest publication and settlement lags must be modeled; Compustat GVKEY/IID must be date-linked to PERMNO."]
    bias_risks:
      survivorship: "Medium to high without dead/security history."
      lookahead: "High if report/publication lag is ignored."
      restatement: "Medium."
      delisting: "Crowded distressed names may delist; retain delisted securities."
      corporate_actions: "Use split-adjusted fields and split adjustment dates."
    notes: "Catalog confirms comp.sec_shortint columns gvkey,iid,shortint,shortintadj,datadate,splitadjdate and comp.io_qholders."

  - requirement_id: "REQ_P2_OPTIONMETRICS_UNDERLYING_SECURITY"
    priority: "P2_HIGH_VALUE"
    project_need: "Link options/implied-volatility data to equity panels and validate underlying return/price histories."
    quant_use_case: "Option-implied volatility regimes, optionable-universe filters, underlying liquidity controls, derivatives-aware holdouts."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "optionm"
    wrds_table: "secprd, secnmd, security_name, wrdsapps_opcrsphist"
    wrds_product_or_library: "OptionMetrics IvyDB US security files"
    logical_dataset: "OptionMetrics underlying security prices and name history"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2018-present for recent critical diagnostics"
      optimal: "2005-present, matching local OptionMetrics partitions"
      tier_split: "modern_research"
    frequency: "daily"
    expected_size_class: "medium"
    partition_strategy: "year"
    key_columns:
      identifiers: ["secid", "cusip", "ticker"]
      dates: ["date", "effect_date"]
      measures: ["low", "high", "close", "volume", "return", "cfadj", "open", "cfret", "shrout", "sic"]
    joins:
      required_link_tables: ["optionm.secnmd", "CRSP-OptionMetrics link via CUSIP/ticker/date or WRDS linking table if entitled"]
      joins_to_project_data: ["future option-implied regime controls", "future optionable subset tests"]
      point_in_time_risks: ["CUSIP/ticker matching must be date-bounded; optionable status is not ex ante if inferred from future option data."]
    bias_risks:
      survivorship: "High if only currently optionable names are used."
      lookahead: "High if option availability from future dates determines past universe."
      restatement: "Low to medium."
      delisting: "Need CRSP delisting integration for underlying returns."
      corporate_actions: "Use OptionMetrics adjustment factor and CRSP adjustment cross-checks."
    notes: "Local shared vault contains optionm secprd yearly parquet files and secnmd.parquet."

  - requirement_id: "REQ_P2_OPTIONMETRICS_OPTION_QUOTES_IV"
    priority: "P2_HIGH_VALUE"
    project_need: "Add option-implied forward volatility/skew regimes and stress controls for covariance forecast validation."
    quant_use_case: "Implied volatility, skew/smile features, option volume/open-interest crowding, volatility-risk regimes, diagnostics for crisis periods."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "optionm"
    wrds_table: "opprcdYYYY, opprcbrYYYY, option_price_view, std_option_price_view, opvold"
    wrds_product_or_library: "OptionMetrics IvyDB US option price files"
    logical_dataset: "Daily option quotes, implied volatilities, greeks, open interest, and volume"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2018-present for recent critical diagnostics"
      optimal: "2005-present"
      tier_split: "modern_research"
    frequency: "daily"
    expected_size_class: "huge"
    partition_strategy: "year"
    key_columns:
      identifiers: ["secid", "optionid", "symbol", "root", "suffix"]
      dates: ["date", "exdate", "last_date"]
      measures: ["strike_price", "best_bid", "best_offer", "volume", "open_interest", "impl_volatility", "delta", "gamma", "vega", "theta", "forward_price", "contract_size"]
    joins:
      required_link_tables: ["optionm.secnmd", "optionm.secprd", "optionm.zerocd", "optionm.forward_price"]
      joins_to_project_data: ["future volatility-regime features", "future optionable-universe diagnostics", "future stress and tail-risk appendices"]
      point_in_time_risks: ["Use only quotes available at date; filter stale/zero-bid quotes; respect option contract history and corporate-action adjustments."]
    bias_risks:
      survivorship: "High if expired or delisted option contracts are omitted by a current-active filter."
      lookahead: "High if realized future volatility or later contract filters leak into feature construction."
      restatement: "Low to medium."
      delisting: "Underlying delistings must be preserved."
      corporate_actions: "High; option adjustment factors and deliverables matter around splits/special dividends."
    notes: "Catalog confirms yearly optionm.opprcdYYYY tables with impl_volatility and greeks; local shared vault has large opprcd yearly parquet files from 2005 through 2025."

  - requirement_id: "REQ_P2_OPTIONMETRICS_BORROW_RATES_CURVES_DISTRIBUTIONS"
    priority: "P2_HIGH_VALUE"
    project_need: "Model shorting frictions, borrow cost, zero curves, forwards, and dividend/distribution expectations."
    quant_use_case: "Borrow/cost modeling, short-side feasibility filters, financing-regime controls, option-implied forward/rate sanity checks."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "optionm"
    wrds_table: "borrateYYYY, zerocd, forward_price, distrd, distribution, distribution_projection"
    wrds_product_or_library: "OptionMetrics borrow, rates, forward, and distribution files"
    logical_dataset: "Borrow rates, zero-coupon curves, forward prices, and dividend/distribution records"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2018-present"
      optimal: "2005-present"
      tier_split: "modern_research"
    frequency: "mixed"
    expected_size_class: "medium"
    partition_strategy: "year"
    key_columns:
      identifiers: ["securityid", "secid"]
      dates: ["date", "expirationdate", "expiration", "record_date", "ex_date", "declare_date", "payment_date"]
      measures: ["borrowrate", "days", "rate", "forwardprice", "amount", "adj_factor", "distr_type"]
    joins:
      required_link_tables: ["optionm.secnmd", "optionm.secprd", "crsp.dsf"]
      joins_to_project_data: ["future borrow-cost modeling", "future volatility/option features"]
      point_in_time_risks: ["Use borrow and distribution records by effective/as-of date; avoid future dividend realization leakage."]
    bias_risks:
      survivorship: "Medium if only liquid optionable securities are retained."
      lookahead: "High if distribution projections are treated as known before publication/effective dates."
      restatement: "Medium for corrected option/vendor records."
      delisting: "Distressed/delisting names often have extreme borrow; preserve exits."
      corporate_actions: "High; distributions and adjustment factors are central."
    notes: "Local shared vault contains zerocd and distrd parquet partitions. Catalog confirms yearly borrate tables with securityid,date,expirationdate,days,borrowrate."

  - requirement_id: "REQ_P2_IBES_ESTIMATES_ACTUALS"
    priority: "P2_HIGH_VALUE"
    project_need: "Add earnings-information controls and event/regime variables for covariance behavior around news."
    quant_use_case: "Earnings-event regimes, analyst dispersion/surprise controls, information-flow diagnostics, out-of-sample stress segmentation."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "ibes"
    wrds_table: "det_epsus, statsum_epsus, act_epsus, id, idsum, recdid"
    wrds_product_or_library: "IBES US detail, summary, actuals, and identifier files"
    logical_dataset: "Analyst EPS estimates, summary statistics, actual EPS announcements, and IBES identifiers"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present"
      optimal: "full IBES US history"
      tier_split: "modern_research"
    frequency: "event"
    expected_size_class: "large"
    partition_strategy: "date_range"
    key_columns:
      identifiers: ["ticker", "cusip", "oftic", "estimator", "analys"]
      dates: ["statpers", "fpedats", "anndats", "actdats", "revdats", "sdates"]
      measures: ["value", "actual", "numest", "numup", "numdown", "medest", "meanest", "stdev", "highest", "lowest"]
    joins:
      required_link_tables: ["ibes.id", "ibes.idsum", "crsp.dsenames", "crsp.ccmxpf_linktable if joining through Compustat"]
      joins_to_project_data: ["future earnings-event diagnostics", "future information-regime controls"]
      point_in_time_risks: ["Use announcement/action dates, not fiscal period dates, for availability; CUSIP/ticker links must be date-bounded."]
    bias_risks:
      survivorship: "High if firms without analyst coverage are silently dropped."
      lookahead: "Very high if actuals or revisions are used before announcement/action dates."
      restatement: "Medium because estimate/actual histories can be adjusted."
      delisting: "Analyst coverage can disappear before distress/delisting; preserve no-coverage state."
      corporate_actions: "CUSIP/ticker changes and split adjustment factors require care."
    notes: "Catalog confirms det_epsus, statsum_epsus, act_epsus, id, and idsum with key estimate/actual columns. No local raw ibes directory was observed."

  - requirement_id: "REQ_P2_INDEX_CONSTITUENTS_CIQ_COMPUSTAT"
    priority: "P2_HIGH_VALUE"
    project_need: "Broaden benchmark universe controls beyond CRSP S&P files and support index-family comparisons."
    quant_use_case: "Index constituent holdouts, index benchmark comparison, sector/index-relative covariance diagnostics, publication tables."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "ciq, comp"
    wrds_table: "ciqindex, ciqindexconstituent, ciqindexvalue, ciqindextradingitem, comp.idx_index, comp.indexcst_his"
    wrds_product_or_library: "Capital IQ index data and Compustat index constituents"
    logical_dataset: "Index definitions, constituent intervals, trading items, and index values"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present"
      optimal: "full available history"
      tier_split: "modern_research"
    frequency: "mixed"
    expected_size_class: "medium"
    partition_strategy: "table"
    key_columns:
      identifiers: ["indexid", "constituentid", "tradingitemid", "gvkeyx", "gvkey", "iid"]
      dates: ["fromdate", "todate", "valuedate", "thrudate"]
      measures: ["value", "indexname", "indexproviderid", "indexcat", "indexgeo", "indextype"]
    joins:
      required_link_tables: ["ciq security/trading-item link tables", "comp.security", "crsp.ccmxpf_linktable"]
      joins_to_project_data: ["future benchmark-universe panels", "future index-relative diagnostics"]
      point_in_time_risks: ["Constituent intervals and trading item mappings must be date-bounded."]
    bias_risks:
      survivorship: "High if current constituent lists are backfilled."
      lookahead: "High if membership changes are used before effective dates."
      restatement: "Medium due vendor corrections."
      delisting: "Index exits and delistings must remain visible."
      corporate_actions: "Constituent/trading-item mappings around corporate actions require effective dates."
    notes: "Catalog confirms ciq.ciqindex, ciq.ciqindexconstituent, ciq.ciqindexvalue and comp.indexcst_his."

  - requirement_id: "REQ_P2_CIQ_EVENTS_AND_SECURITY_MASTER"
    priority: "P2_HIGH_VALUE"
    project_need: "Add corporate-event and security master context for regime/event diagnostics."
    quant_use_case: "M&A events, key developments, corporate actions, event windows, abnormal covariance regimes, security identity cross-checks."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "ciq"
    wrds_table: "ciqkeydev, ciqkeydevtoobjecttoeventtype, ciqsecurity, ciqtransactionmergerevent, ciqtransmadoctomergerevent"
    wrds_product_or_library: "Capital IQ company, security, key development, and transaction data"
    logical_dataset: "Corporate events, key developments, securities, and merger transaction records"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present"
      optimal: "full available history"
      tier_split: "modern_research"
    frequency: "event"
    expected_size_class: "large"
    partition_strategy: "date_range"
    key_columns:
      identifiers: ["keydevid", "objectid", "keydeveventtypeid", "securityid", "tradingitemid"]
      dates: ["speffectivedate", "sptodate", "announceddate", "announceddateutc", "entereddateutc", "securitystartdate"]
      measures: ["headline", "situation", "event type ids", "transaction ids"]
    joins:
      required_link_tables: ["ciq security/trading-item tables", "CRSP/Compustat links where needed"]
      joins_to_project_data: ["future event-regime controls", "future M&A/corporate-action exclusions"]
      point_in_time_risks: ["Use announcement/effective timestamps; entered and last-modified timestamps require explicit as-of policy."]
    bias_risks:
      survivorship: "Medium if only events for current firms are kept."
      lookahead: "High if event records are aligned to effective dates but known later."
      restatement: "Medium because event records can be revised."
      delisting: "M&A and bankruptcy events can drive exits; preserve exits."
      corporate_actions: "High; security master and transaction records must be date-bounded."
    notes: "Useful for diagnosing whether covariance overlay behavior is concentrated around corporate event windows."

  - requirement_id: "REQ_P3_TAQ_QUOTES_NBBO"
    priority: "P3_SPECIALIZED"
    project_need: "Support microstructure-grade liquidity and spread diagnostics for ambitious future robustness."
    quant_use_case: "Intraday spread/quote quality, NBBO liquidity filters, microstructure noise controls, transaction-cost stress tests."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "taqmsec"
    wrds_table: "complete_nbbo_YYYYMMDD, cqm_YYYYMMDD, mastm_YYYYMMDD"
    wrds_product_or_library: "NYSE TAQ Monthly Securities"
    logical_dataset: "Intraday NBBO and quote master data"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2018-present only if transaction-cost/microstructure claims are introduced"
      optimal: "sampled recent critical windows first; avoid full dump until hypothesis is specific"
      tier_split: "recent_critical"
    frequency: "intraday"
    expected_size_class: "huge"
    partition_strategy: "month"
    key_columns:
      identifiers: ["sym_root", "sym_suffix", "cusip"]
      dates: ["date", "time_m", "time_m_nano", "effective_date"]
      measures: ["best_bid", "best_bidsizeshares", "best_ask", "best_asksizeshares", "bid", "bidsiz", "ask", "asksiz", "qu_cond", "natbbo_ind"]
    joins:
      required_link_tables: ["taqmsec.mastm_YYYYMMDD", "crsp.dsenames", "crsp.stocknames_v2"]
      joins_to_project_data: ["future liquidity filters", "future transaction-cost models"]
      point_in_time_risks: ["Symbol and CUSIP matching must be date-specific; intraday timezone/session filters must be explicit."]
    bias_risks:
      survivorship: "Medium if only currently listed tickers are queried."
      lookahead: "Medium if full-day liquidity is used for decisions at the open."
      restatement: "Low to medium due corrections."
      delisting: "TAQ symbol history must retain delisted names for historical windows."
      corporate_actions: "High around ticker/CUSIP/split changes."
    notes: "Catalog confirms taqmsec.complete_nbbo_20241231 and cqm_20241231 columns. No local raw TAQ directory was observed; download only by tightly scoped dates/symbols."

  - requirement_id: "REQ_P3_TAQ_TRADES"
    priority: "P3_SPECIALIZED"
    project_need: "Support intraday realized liquidity/impact diagnostics and execution-cost assumptions."
    quant_use_case: "Trade-size distributions, effective spreads, realized volatility at intraday horizons, market impact proxies."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "taqmsec"
    wrds_table: "ctm_YYYYMMDD, mastm_YYYYMMDD"
    wrds_product_or_library: "NYSE TAQ Monthly Securities"
    logical_dataset: "Intraday consolidated trades and security master"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "selected event/crisis dates only"
      optimal: "recent critical sample windows, then broader modern research if transaction-cost claims matter"
      tier_split: "recent_critical"
    frequency: "intraday"
    expected_size_class: "huge"
    partition_strategy: "month"
    key_columns:
      identifiers: ["sym_root", "sym_suffix", "cusip"]
      dates: ["date", "time_m", "time_m_nano", "part_time", "trf_time"]
      measures: ["ex", "tr_scond", "size", "price", "tr_corr", "tr_seqnum", "tr_id", "tr_source"]
    joins:
      required_link_tables: ["taqmsec.mastm_YYYYMMDD", "crsp.dsenames"]
      joins_to_project_data: ["future liquidity/transaction-cost appendix"]
      point_in_time_risks: ["Corrections and late prints need explicit filtering; avoid using end-of-day aggregates for intraday decisions unless documented."]
    bias_risks:
      survivorship: "Medium if symbols are selected from current universe only."
      lookahead: "Medium to high if whole-day trading data drives same-day portfolio decisions."
      restatement: "Low to medium due trade corrections."
      delisting: "Need historical symbol coverage."
      corporate_actions: "High around splits/ticker changes."
    notes: "Catalog confirms taqmsec.ctm_20241231 with trade price, size, correction, sequence, source, and timestamps."

  - requirement_id: "REQ_P3_13F_AND_THOMSON_HOLDINGS"
    priority: "P3_SPECIALIZED"
    project_need: "Add ownership/crowding and institutional-flow diagnostics."
    quant_use_case: "Crowding, common ownership, institutional ownership changes, cross-sectional stress concentration, liquidity fragility."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "tr_13f, tfn"
    wrds_table: "tr_13f.s34, tr_13f.s34names, tfn.s34, tfn.s12, tfn.s12names"
    wrds_product_or_library: "Thomson/Refinitiv 13F and mutual fund holdings on WRDS"
    logical_dataset: "Institutional and mutual fund holdings"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present"
      optimal: "full available history"
      tier_split: "modern_research"
    frequency: "quarterly"
    expected_size_class: "large"
    partition_strategy: "date_range"
    key_columns:
      identifiers: ["mgrno", "fundno", "cusip", "ticker"]
      dates: ["fdate", "rdate", "prdate", "rdate1"]
      measures: ["shares", "sole", "shared", "no", "change", "prc", "shrout1", "shrout2", "assets"]
    joins:
      required_link_tables: ["crsp.dsenames", "crsp.dsf", "crsp.ccmxpf_linktable if joining to fundamentals"]
      joins_to_project_data: ["future crowding diagnostics", "future ownership-regime controls"]
      point_in_time_risks: ["Use filing/report lags; holdings report dates are not necessarily public knowledge on fdate."]
    bias_risks:
      survivorship: "High if only current institutions/funds are retained."
      lookahead: "High without filing-lag policy."
      restatement: "Medium due amended filings."
      delisting: "Holdings of delisted names may vanish if joined through survivor universe."
      corporate_actions: "CUSIP changes and share adjustments require care."
    notes: "Catalog confirms tr_13f.s34 and tfn.s12/s34 tables. No local raw directories observed for tr_13f or tfn."

  - requirement_id: "REQ_P3_TFN_INSIDER_AND_RULE144"
    priority: "P3_SPECIALIZED"
    project_need: "Add insider-sale, rule 10b5-1, and Rule 144 event diagnostics if governance/event hypotheses are pursued."
    quant_use_case: "Information-event controls, insider-flow flags, governance stress, event-window exclusions."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "tfn"
    wrds_table: "table1, table2, form144, rule10b5, header, idfhist, idfnames"
    wrds_product_or_library: "Thomson insider/Form 144 data on WRDS"
    logical_dataset: "Insider transactions, Form 144 sales, and Rule 10b5-1 records"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present"
      optimal: "full available history"
      tier_split: "modern_research"
    frequency: "event"
    expected_size_class: "large"
    partition_strategy: "date_range"
    key_columns:
      identifiers: ["cusip", "ticker", "insider identifiers"]
      dates: ["secdate", "effdate", "maintdate"]
      measures: ["transaction fields", "shares", "prices", "relationship codes"]
    joins:
      required_link_tables: ["tfn.header", "tfn.idfhist", "crsp.dsenames"]
      joins_to_project_data: ["future governance/event diagnostics"]
      point_in_time_risks: ["Use SEC filing dates and effective dates; ticker/CUSIP links must be date-specific."]
    bias_risks:
      survivorship: "Medium if only current firms are matched."
      lookahead: "High if transaction/effective dates are used before public filing dates."
      restatement: "Medium due amendments/corrections."
      delisting: "Distressed delisted names may have important insider activity."
      corporate_actions: "Share adjustments and CUSIP changes require care."
    notes: "Specialized; do not stage until a governance/event hypothesis is active."

  - requirement_id: "REQ_P3_AUDIT_ANALYTICS_EVENTS"
    priority: "P3_SPECIALIZED"
    project_need: "Add restatement, audit, late-filing, legal, and governance event controls for stress/event studies."
    quant_use_case: "Restatement/event exclusion, accounting-risk flags, late-filing and going-concern stress diagnostics, litigation/regulatory event windows."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "audit"
    wrds_table: "feed39_financial_restatements, f39_restatement_filings, feed20_nt, feed74_aqrm, feed90_europe_aqrm, feed03_audit_fees, feed55_auditor_ratification, feed91_aaer"
    wrds_product_or_library: "Audit Analytics on WRDS"
    logical_dataset: "Accounting, restatement, auditor, legal, and regulatory events"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present"
      optimal: "full available history"
      tier_split: "modern_research"
    frequency: "event"
    expected_size_class: "large"
    partition_strategy: "date_range"
    key_columns:
      identifiers: ["cik", "company identifiers", "auditor identifiers"]
      dates: ["file_date", "event_date", "period_start_date", "period_end_date", "first_release_date"]
      measures: ["restatement flags", "audit fees", "late filing flags", "going concern flags", "AAER release fields"]
    joins:
      required_link_tables: ["audit lookup tables", "Compustat company CIK/GVKEY", "CRSP-Compustat link"]
      joins_to_project_data: ["future event filters", "future accounting-risk regimes"]
      point_in_time_risks: ["Use filing/release dates rather than affected fiscal period dates for availability."]
    bias_risks:
      survivorship: "Medium if firms without Audit coverage are dropped."
      lookahead: "Very high if restatement periods are used before restatement filing dates."
      restatement: "Primary topic; restatement timing must be explicit."
      delisting: "High relevance because restatements/legal events can precede delistings."
      corporate_actions: "CIK/GVKEY and company name changes require link discipline."
    notes: "Catalog confirms broad Audit Analytics feed tables. Specialized but valuable for publication-quality event exclusions and stress tests."

  - requirement_id: "REQ_P3_BOARDEX_EXECCOMP_GOVERNANCE"
    priority: "P3_SPECIALIZED"
    project_need: "Add governance, executive, and board controls if future hypotheses require management or governance regimes."
    quant_use_case: "Governance controls, executive turnover flags, compensation incentives, board-network/event diagnostics."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "boardex, execcomp"
    wrds_table: "boardex.na_wrds_company_names, boardex.na_board_characteristics, boardex.na_dir_profile_details, execcomp.anncomp, execcomp.exnames"
    wrds_product_or_library: "BoardEx and ExecComp"
    logical_dataset: "Board, director, executive, and compensation records"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present"
      optimal: "full available North America history"
      tier_split: "modern_research"
    frequency: "annual"
    expected_size_class: "medium"
    partition_strategy: "table"
    key_columns:
      identifiers: ["boardid", "companyid", "directorid", "gvkey", "execid", "ticker", "cusip", "cik"]
      dates: ["annualreportdate", "datestartrole", "becameceo", "joined_co", "leftco", "year"]
      measures: ["board characteristics", "director profile fields", "salary", "bonus", "stock_awards", "option_awards", "tdc1", "tdc2"]
    joins:
      required_link_tables: ["boardex company names", "execcomp.exnames", "comp.company", "crsp.ccmxpf_linktable"]
      joins_to_project_data: ["future governance/event controls"]
      point_in_time_risks: ["Annual report dates and role dates must be treated as availability dates; do not use future board state."]
    bias_risks:
      survivorship: "Medium to high for governance datasets with coverage changes."
      lookahead: "High if annual records are applied before report availability."
      restatement: "Medium due corrected compensation/governance records."
      delisting: "Delisted firms and acquired companies must remain in history."
      corporate_actions: "Company identifiers around M&A need care."
    notes: "Not needed for current detector work, but useful for ambitious governance/event-control extensions."

  - requirement_id: "REQ_P3_CBOE_EOD_OPTIONS_AND_BORROW"
    priority: "P3_SPECIALIZED"
    project_need: "Provide an alternate derivatives/borrow data source and cross-check OptionMetrics-derived option/borrow features."
    quant_use_case: "Borrow-rate validation, implied-volatility cross-checks, optionable-universe verification, EOD option price features."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "cboe, cboe_eod"
    wrds_table: "ivborrowrate, ivlistedYYYY, optpriceYYYY, eqprice, eqmaster, eqhvol"
    wrds_product_or_library: "CBOE EOD option and equity data on WRDS"
    logical_dataset: "CBOE EOD equity, option, implied-volatility, historical-volatility, and borrow-rate data"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2018-present if used as cross-check"
      optimal: "1998-present where available"
      tier_split: "modern_research"
    frequency: "daily"
    expected_size_class: "huge"
    partition_strategy: "year"
    key_columns:
      identifiers: ["eqid", "option identifiers"]
      dates: ["date_", "termdate", "expdate", "startdate"]
      measures: ["rate", "option prices", "implied vol fields", "historical volatility", "equity price fields"]
    joins:
      required_link_tables: ["cboe.eqmaster", "crsp.dsenames"]
      joins_to_project_data: ["future OptionMetrics validation", "future borrow-cost cross-checks"]
      point_in_time_risks: ["Need exact as-of date and contract availability policy."]
    bias_risks:
      survivorship: "High if only current CBOE symbols are used."
      lookahead: "High if term structures are formed using unavailable future contracts."
      restatement: "Medium due EOD vendor corrections."
      delisting: "Underlying delistings must be preserved."
      corporate_actions: "Option deliverables and splits require careful handling."
    notes: "Catalog confirms cboe.ivborrowrate and cboe_eod.ivborrowrate plus yearly option/equity tables."

  - requirement_id: "REQ_P4_TRACE_MSRB_FIXED_INCOME_STRESS"
    priority: "P4_LONG_TAIL"
    project_need: "Add broad credit/rates/liquidity stress context if future work studies cross-asset regimes."
    quant_use_case: "Credit-market stress regimes, funding/liquidity controls, macro crisis diagnostics, bond-equity covariance environment."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "trace, msrb"
    wrds_table: "trace.trace, trace.trace_enhanced, trace.trade_summary, msrb.msrb"
    wrds_product_or_library: "TRACE corporate bond and MSRB municipal bond data"
    logical_dataset: "Corporate and municipal bond transactions and summaries"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "selected crisis periods only"
      optimal: "modern research period if cross-asset regime work becomes central"
      tier_split: "legacy_backfill"
    frequency: "intraday"
    expected_size_class: "huge"
    partition_strategy: "month"
    key_columns:
      identifiers: ["cusip", "rtrs_control_number"]
      dates: ["trd_exctn_dt", "trd_rpt_efctv_dt", "trans_dt", "trade_date", "time_of_trade"]
      measures: ["price", "yield", "par_traded", "trade size", "trade type", "trade summary fields"]
    joins:
      required_link_tables: ["CUSIP issuer mapping if linking to equities", "Compustat company/security links"]
      joins_to_project_data: ["future cross-asset regime controls"]
      point_in_time_risks: ["Trade reporting corrections and delayed dissemination must be modeled for real-time claims."]
    bias_risks:
      survivorship: "Medium if only active CUSIPs are retained."
      lookahead: "Medium to high depending on reporting delay treatment."
      restatement: "Medium due corrected trade records."
      delisting: "Credit distress and equity delistings can be linked but require issuer mapping."
      corporate_actions: "Issuer/security mapping can change around restructuring."
    notes: "Long-tail for this equity covariance project; do not stage before higher tiers unless cross-asset regime research becomes explicit."

  - requirement_id: "REQ_P4_CRSP_MUTUAL_FUND_AND_FUND_HOLDINGS"
    priority: "P4_LONG_TAIL"
    project_need: "Study fund-flow or holdings pressure as an optional crowding/liquidity extension."
    quant_use_case: "Fund flow pressure, mutual-fund holdings crowding, redemption risk regimes, institutional ownership cross-checks."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "crsp"
    wrds_table: "fund_names, mfdbname, holdings, holdings_co_info"
    wrds_product_or_library: "CRSP Survivor-Bias-Free Mutual Fund Database and holdings"
    logical_dataset: "Mutual fund names, holdings, and related company/security information"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present only if fund-flow hypotheses are active"
      optimal: "full available history"
      tier_split: "legacy_backfill"
    frequency: "mixed"
    expected_size_class: "large"
    partition_strategy: "date_range"
    key_columns:
      identifiers: ["fund identifiers", "cusip"]
      dates: ["chgdt", "report_dt", "maturity_dt"]
      measures: ["holdings fields", "fund attributes"]
    joins:
      required_link_tables: ["CRSP mutual fund link/name tables", "crsp.dsenames"]
      joins_to_project_data: ["future crowding/liquidity diagnostics"]
      point_in_time_risks: ["Fund reports have reporting delays; use public availability date if making ex ante claims."]
    bias_risks:
      survivorship: "High without survivor-bias-free fund history."
      lookahead: "High without filing/reporting lag policy."
      restatement: "Medium."
      delisting: "Holdings of distressed/delisted names must remain visible."
      corporate_actions: "CUSIP/security changes require date-bounded mapping."
    notes: "Long-tail; 13F/TFN holdings are higher-priority for crowding diagnostics."

  - requirement_id: "REQ_P4_GLOBAL_EQUITY_EXTENSION"
    priority: "P4_LONG_TAIL"
    project_need: "Enable non-US or global robustness only after the US equity detector path is understood."
    quant_use_case: "International holdout validation, global market regimes, cross-country covariance structure, external validity."
    wrds_availability: "confirmed_in_catalog"
    wrds_schema: "comp, comp_global_daily"
    wrds_table: "g_funda, g_fundq, g_names, g_secnamesd, g_security, g_idx_index, g_indexcst_his"
    wrds_product_or_library: "Compustat Global"
    logical_dataset: "Global fundamentals, securities, names, and index constituents"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2010-present if global holdout is approved"
      optimal: "full available global history"
      tier_split: "legacy_backfill"
    frequency: "mixed"
    expected_size_class: "large"
    partition_strategy: "year"
    key_columns:
      identifiers: ["gvkey", "iid", "cusip", "isin", "tic"]
      dates: ["datadate", "fromdate", "thrudate"]
      measures: ["fundamental fields", "security metadata", "index metadata"]
    joins:
      required_link_tables: ["global security/name/link tables", "local-market price source not identified here"]
      joins_to_project_data: ["future international holdout package"]
      point_in_time_risks: ["Country-specific reporting lags, currency conversion, and trading calendars must be explicit."]
    bias_risks:
      survivorship: "High without global dead/security histories."
      lookahead: "High without report availability dates and link windows."
      restatement: "High for fundamentals."
      delisting: "Global delisting/exit treatment must be sourced."
      corporate_actions: "Complex due ADRs, currency, share classes, and exchange moves."
    notes: "Potentially valuable, but not before the US current research blocker is solved."

  - requirement_id: "REQ_P4_MACRO_FRED_EXTERNAL"
    priority: "P4_LONG_TAIL"
    project_need: "Add macroeconomic regime controls if advisor/paper framing requires macro states beyond market returns and rates."
    quant_use_case: "Inflation/rates/growth regimes, recession controls, macro stress slices, robustness appendices."
    wrds_availability: "uncertain_or_external"
    wrds_schema: ""
    wrds_table: ""
    wrds_product_or_library: "FRED or other macro providers; some macro series may be accessible through WRDS add-ons but catalog verification is needed"
    logical_dataset: "Macro time series such as CPI, unemployment, policy rates, term spreads, credit spreads, VIX-style market stress"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2000-present"
      optimal: "full available history"
      tier_split: "legacy_backfill"
    frequency: "mixed"
    expected_size_class: "tiny"
    partition_strategy: "table"
    key_columns:
      identifiers: ["series_id"]
      dates: ["date", "release_date", "vintage_date"]
      measures: ["value"]
    joins:
      required_link_tables: []
      joins_to_project_data: ["future macro-regime labels"]
      point_in_time_risks: ["Use real-time vintages for ex ante claims; final revised macro series leak future revisions."]
    bias_risks:
      survivorship: "Low."
      lookahead: "High without vintage/release-date policy."
      restatement: "High for revised macro series."
      delisting: "Not applicable."
      corporate_actions: "Not applicable."
    notes: "Not WRDS-core in this local catalog check. Use only if macro framing becomes explicit."

  - requirement_id: "REQ_P4_NEWS_SENTIMENT_EXTERNAL"
    priority: "P4_LONG_TAIL"
    project_need: "Add news/sentiment/event context only if future strategy requires information-flow hypotheses beyond IBES/CIQ/Audit events."
    quant_use_case: "News-driven volatility regimes, event filtering, sentiment stress, headline concentration, information shock diagnostics."
    wrds_availability: "uncertain_or_external"
    wrds_schema: ""
    wrds_table: ""
    wrds_product_or_library: "RavenPack, Refinitiv News Analytics, Dow Jones, or other news/sentiment products; WRDS availability depends on entitlement"
    logical_dataset: "Timestamped firm-level news, sentiment, relevance, novelty, and event classification"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: true
    date_range:
      minimum_required: "2018-present for recent critical diagnostics"
      optimal: "2005-present or product maximum"
      tier_split: "recent_critical"
    frequency: "event"
    expected_size_class: "huge"
    partition_strategy: "month"
    key_columns:
      identifiers: ["ticker", "cusip", "isin", "permno if provided", "company_id"]
      dates: ["timestamp_utc", "publication_time", "event_time"]
      measures: ["sentiment", "relevance", "novelty", "event_type", "source"]
    joins:
      required_link_tables: ["vendor identifier map", "crsp.dsenames", "comp.company"]
      joins_to_project_data: ["future information-shock diagnostics"]
      point_in_time_risks: ["Timestamp availability and revision/correction policy are central; do not use article metadata created after the event as ex ante features."]
    bias_risks:
      survivorship: "High if only current tickers are matched."
      lookahead: "Very high if event timestamps or revisions are mishandled."
      restatement: "Medium due vendor reclassification."
      delisting: "News around distressed exits must be retained."
      corporate_actions: "Ticker/entity changes around M&A require robust mapping."
    notes: "Long-tail and likely external; CIQ key developments and IBES are higher-priority WRDS-grounded event layers."

  - requirement_id: "REQ_P4_SHARADAR_EXTERNAL_FALLBACK"
    priority: "P4_LONG_TAIL"
    project_need: "Document the existing non-WRDS script path so it is not confused with the canonical WRDS pipeline."
    quant_use_case: "Fallback daily price data path for development or comparison only, not canonical reproduction."
    wrds_availability: "not_wrds"
    wrds_schema: ""
    wrds_table: ""
    wrds_product_or_library: "NASDAQ Data Link Sharadar SEP/TICKERS"
    logical_dataset: "Sharadar adjusted daily prices and ticker metadata"
    required_for_current_reproduction: false
    required_for_current_claim_validation: false
    useful_for_future_expansion: false
    date_range:
      minimum_required: "not applicable unless explicitly selected"
      optimal: "not applicable"
      tier_split: "table_level"
    frequency: "daily"
    expected_size_class: "large"
    partition_strategy: "date_range"
    key_columns:
      identifiers: ["ticker", "permaticker"]
      dates: ["date"]
      measures: ["closeadj", "close", "volume"]
    joins:
      required_link_tables: ["SHARADAR/TICKERS"]
      joins_to_project_data: ["scripts/data/fetch_sharadar.py", "scripts/data/make_balanced_weekly.py"]
      point_in_time_risks: ["Ticker-based mapping can be fragile; do not mix with WRDS canonical results without manifesting source differences."]
    bias_risks:
      survivorship: "Depends on vendor and ticker filters; must be audited separately."
      lookahead: "Universe selection can leak if ADV/top-p uses future data."
      restatement: "Vendor revisions possible."
      delisting: "Needs explicit dead/delisted coverage check."
      corporate_actions: "Uses adjusted prices; compare adjustment conventions to CRSP."
    notes: "The repo has scripts/data/fetch_sharadar.py, but current registry identifies CRSP/WRDS as the canonical source."

wrds_catalog_checks_needed:
  - "Confirm the active WRDS catalog snapshot date and whether it reflects current entitlements or a previous user's entitlements."
  - "Confirm exact local raw coverage for crsp.dsf, crsp.dsenames, crsp.dsedelist, ff.fivefactors_daily, ff.factors_daily, comp.funda, comp.fundq, and optionm partitions before any rebuild ticket."
  - "Audit whether current data/returns_daily.csv intentionally excludes separate CRSP delisting returns or whether dsf.ret is treated as sufficient for the current estimand."
  - "Recompute and record a source manifest for data/returns_daily.csv from the shared WRDS vault: filters, SQL, source table snapshots, row counts, columns, hash, and universe selection procedure."
  - "Recompute and record a source manifest for data/factors/ff5mom_daily.csv from ff.fivefactors_daily and ff.factors_daily, including factor vintage and column transforms."
  - "Verify exact WRDS link table choices for CRSP-Compustat joins: linktype/linkprim/usedflag filters and link date convention."
  - "Verify whether OptionMetrics to CRSP linking should use an available WRDS link helper, CUSIP date matching, or a custom audited bridge."
  - "Before using IBES, verify the best identifier bridge from IBES ticker/CUSIP to CRSP PERMNO/GVKEY and define announcement-date availability rules."
  - "Before using TAQ, select a tiny date/symbol pilot and write a manifest; never stage broad TAQ raw data without a scoped hypothesis."
  - "Before using Audit Analytics, BoardEx, ExecComp, CIQ, TRACE/MSRB, or CBOE data, confirm entitlement and local raw staging policy."

entitlement_questions:
  - "Does the project/admin have standing WRDS entitlement for CRSP, Compustat North America, CCM, and FF factors? These are P0/P1."
  - "Does the project/admin have standing entitlement for OptionMetrics IvyDB US and CBOE EOD? These are useful P2/P3 but not current blockers."
  - "Does the project/admin have entitlement for IBES detail/summary/actuals and identifier files?"
  - "Does the project/admin have entitlement for NYSE TAQ Monthly Securities, and are there storage limits for selected-date pulls?"
  - "Does the project/admin have entitlement for Thomson/Refinitiv 13F, mutual fund holdings, insider/Form 144 data, and related name/type tables?"
  - "Does the project/admin have entitlement for Capital IQ company/security/index/key development/event tables?"
  - "Does the project/admin have entitlement for Audit Analytics, BoardEx, and ExecComp, and are these appropriate for this research scope?"
  - "Should external non-WRDS sources such as FRED, RavenPack/news, or Sharadar be allowed in future packages, or should the canonical path remain WRDS-only?"

highest_priority_download_order:
  - requirement_id: "REQ_P0_CRSP_DSF_DAILY_RETURNS"
    reason: "Required for exact current reproduction and all core return panels; verify local raw partitions before downloading anything new."
  - requirement_id: "REQ_P0_CRSP_DAILY_NAMES_AND_UNIVERSE_FILTERS"
    reason: "Required to reproduce share-code, exchange, ticker, and universe filters without survivorship bias."
  - requirement_id: "REQ_P0_FF5_MOM_DAILY_FACTORS"
    reason: "Required for current prewhitened runs and factor-baseline ablations."
  - requirement_id: "REQ_P0_CRSP_DELISTINGS_DISTRIBUTIONS_ACTIONS"
    reason: "Required to validate current claims against delisting and corporate-action bias."
  - requirement_id: "REQ_P1_CRSP_DAILY_INDEX_RETURNS"
    reason: "Small, foundational benchmark/regime layer for interpreting equity covariance results."
  - requirement_id: "REQ_P1_CRSP_SP500_MEMBERSHIP_AND_INDEX"
    reason: "Small, high-value benchmark universe and survivorship-control layer."
  - requirement_id: "REQ_P1_CRSP_COMPUSTAT_LINKS"
    reason: "Foundational for all future fundamental joins and point-in-time controls."
  - requirement_id: "REQ_P1_COMPUSTAT_FUNDAMENTALS"
    reason: "Core future factor/fundamental baseline layer; local raw appears present but needs manifesting."
  - requirement_id: "REQ_P1_FF_PORTFOLIOS_INDUSTRY_LIQUIDITY"
    reason: "Small, high-signal benchmark and liquidity-control layer."
  - requirement_id: "REQ_P2_COMPUSTAT_SHORT_INTEREST_AND_OWNERSHIP"
    reason: "High-value borrow/crowding proxy before heavier options or TAQ work."
  - requirement_id: "REQ_P2_OPTIONMETRICS_UNDERLYING_SECURITY"
    reason: "Needed before any option-implied feature work and already appears locally staged."
  - requirement_id: "REQ_P2_OPTIONMETRICS_OPTION_QUOTES_IV"
    reason: "High-value but huge; stage only after a specific volatility-regime or stress diagnostic ticket."

dedupe_keys_for_global_merge:
  - "crsp.dsf:permno,date"
  - "crsp.dsenames:permno,namedt,nameendt"
  - "crsp.dsedelist:permno,dlstdt"
  - "crsp.dsedist:permno,disexdt"
  - "crsp.stkdlycumulativeadjfactor:permno,dlycaldt"
  - "ff.fivefactors_daily:date"
  - "ff.factors_daily:date"
  - "crsp.wrds_dailyindexret_query:dlycaldt"
  - "crsp.dsp500list:permno,start,ending"
  - "crsp.ccmxpf_linktable:gvkey,lpermno,lpermco,linkdt,linkenddt,linktype,linkprim"
  - "comp.funda:gvkey,datadate,indfmt,consol,popsrc,datafmt"
  - "comp.fundq:gvkey,datadate,fyearq,fqtr,indfmt,consol,popsrc,datafmt"
  - "comp.company:gvkey"
  - "comp.sec_shortint:gvkey,iid,datadate"
  - "comp.io_qholders:gvkey,iid,datadate,ioid"
  - "optionm.secnmd:secid,effect_date"
  - "optionm.secprd:secid,date"
  - "optionm.opprcdYYYY:secid,optionid,date"
  - "optionm.borrateYYYY:securityid,date,expirationdate"
  - "optionm.zerocd:date,days"
  - "optionm.distrd:secid,record_date,seq_num"
  - "ibes.det_epsus:ticker,estimator,analys,fpedats,actdats,revdats,measure,fpi"
  - "ibes.statsum_epsus:ticker,statpers,fpedats,measure,fpi"
  - "ibes.act_epsus:ticker,pends,measure,anndats,actdats"
  - "ciq.ciqindexconstituent:indexid,constituentid,fromdate,todate,tradingitemid"
  - "ciq.ciqkeydev:keydevid"
  - "taqmsec.complete_nbbo_YYYYMMDD:sym_root,sym_suffix,date,time_m,time_m_nano,qu_source"
  - "taqmsec.ctm_YYYYMMDD:sym_root,sym_suffix,date,time_m,time_m_nano,tr_seqnum,tr_id"
  - "tr_13f.s34:mgrno,cusip,fdate,rdate"
  - "tfn.s12:fundno,cusip,fdate,rdate"
  - "tfn.table1:insider_or_filing_key,secdate"
  - "audit.feed39_financial_restatements:company_or_filing_key,file_date"
  - "boardex.na_wrds_company_names:boardid,companyid"
  - "execcomp.anncomp:gvkey,execid,year"
  - "trace.trace:cusip,trd_exctn_dt,trade_identifier_or_control_number"
  - "msrb.msrb:rtrs_control_number"
```
