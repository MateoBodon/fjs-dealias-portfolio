<!--
Provenance
- Ingested: 2026-02-16
- Source: uploaded Analysis.md (`docs/Analysis.md`)
- Method: verbatim paste; body below is unchanged from source text.
-->

## **1\) Project snapshot (what you *actually* have right now)**

* **Core idea:** apply Fan–Johnstone–Sun (FJS) “de‑aliasing” for **spiked MANOVA / random‑effects covariance structure** as a **gated overlay** on top of mainstream covariance estimators, then test whether this improves **out‑of‑sample risk forecasting** and **min‑variance portfolio behavior** (your original framing is in `One_Pager.pdf`, `Extended_Proposal.pdf`, `Paper_Meeting_Notes.pdf`).  
* **Repo has grown into a serious research codebase**: clear modular separation (`project_state/ARCHITECTURE.md`, `project_state/MODULE_SUMMARIES.md`), explicit “stop-the-line” rules (`AGENTS.md`, `docs/PLAN_OF_RECORD.md`), synthetic calibration harnesses, rolling evaluation runners, and a real testing surface (`project_state/TEST_COVERAGE.md`).  
* **Two evaluation “front-ends”:**  
  * Daily rolling evaluator: `experiments/eval/run.py` (targets: `make rc-*`, `rc-lite`, etc; see `project_state/PIPELINE_FLOW.md` / `EXPERIMENTS.md`).  
  * Weekly panel runner: `experiments/equity_panel/run.py` (nested/oneway/DoW/vol designs).  
* **You have strong instrumentation for validity**, which is uncommon for student research:  
  * “No silent fallbacks,” cap discipline, window intersection discipline, explicit skip reasons, etc (`AGENTS.md`, `docs/PLAN_OF_RECORD.md`, and the eval validity work referenced throughout `PROGRESS.md`).  
* **But the research signal is not established**: your own logs repeatedly show that **overlay triggers are scarce or brittle** (weekly/nested often zero), and when it triggers, **benefits are mixed and small** (`project_state/CURRENT_RESULTS.md`, `project_state/KNOWN_ISSUES.md`, `project_state/RESEARCH_NOTES.md`, and the relevant entries in `PROGRESS.md`).  
* **Most important meta‑observation:** you’ve successfully built a *credible experimental platform*, but you have not yet locked in a *paper‑tight scientific claim* that survives (i) correct synthetic calibration \+ (ii) real‑data robustness \+ (iii) strong baselines.

That’s not “nothing” — it’s a real project — but the next step is to stop expanding the platform and force the project into a crisp, falsifiable paper.

---

## **2\) Research framing and goals (is there a publishable question?)**

### **What the paper can honestly be “about”**

Right now, the strongest coherent framing is basically what `docs/PLAN_OF_RECORD.md` already says:

* **Estimand:** out‑of‑sample portfolio risk forecast quality (variance forecast losses like **MSE** and **QLIKE**), plus portfolio operational metrics (turnover, skip rates, etc).  
* **Baseline:** standard covariance estimators (LW/OAS/RIE and/or factor baselines; `docs/PLAN_OF_RECORD.md` explicitly calls out the “minimum defensible baseline set”).  
* **Treatment:** a **gated** FJS/MANOVA de‑aliasing overlay applied as a *surgical spectral correction* on top of a base estimator (the “overlay operator” in `src/fjs/overlay.py` and gating in `src/fjs/gating.py` / `src/fjs/mp.py`, as described in `project_state/ARCHITECTURE.md` and `docs/PLAN_OF_RECORD.md`).

### **Is the idea valid?**

**Mathematically:** yes, the underlying phenomenon is real: in high‑dimensional random‑effects designs, MANOVA outliers can be “aliased,” and FJS provides a de‑aliasing procedure (your one‑pager / proposal summarizes the exact mechanism).

**As a finance application:** it is *plausible but not guaranteed to matter*, because the finance mapping is an approximation:

* FJS theory assumes a random‑effects structure with (near) balanced replicates and high‑dimensional asymptotics.  
* Daily returns are time‑dependent, heavy‑tailed, heteroskedastic, and factor‑structured. You’ve partially addressed this via **factor prewhitening** and **robust scatter / robust MP edge** toggles (`project_state/ARCHITECTURE.md`, `project_state/EXPERIMENTS.md`, `project_state/CONFIG_REFERENCE.md`).  
* The real question isn’t “is FJS true?” — it’s: **does this overlay ever improve forecasting/optimization beyond modern shrinkage/factors once you control false positives?**

That’s a legitimate research question.

### **Where it becomes “real research” (vs a code exercise)**

It becomes real research if you can answer something like:

“Under which panel designs and market regimes does a calibrated FJS de‑aliasing overlay measurably improve min‑variance portfolio risk (or variance forecast losses), and when is it neutral/harmful?”

That’s quant‑adjacent, falsifiable, and not obviously already published.

---

## **3\) Theory ↔ implementation alignment (where you’re strong, where you’re shaky)**

### **Where you’re well aligned**

* **Balanced design obsession** is correct: you explicitly build balanced windows (`src/eval/balance.py` \+ the balancing path described in `project_state/DATAFLOW.md`), because FJS closed forms and outlier logic are fragile when balance breaks.  
* **Synthetic null/power calibration exists** and is treated as a first‑class artifact (`experiments/synthetic/*`, calibration JSONs under `calibration/`, and reporting in `project_state/CURRENT_RESULTS.md`).  
* **Gating/guardrails are taken seriously** (stability in an a‑neighborhood, MP edge buffer, isolated spike requirement, etc). That’s exactly what a referee will grill you on.

### **Where alignment is likely violated (and you have to own it)**

1. **“What is the variance component I’m de‑aliasing?” is still not paper‑tight.**  
   FJS de‑aliasing is about isolating spikes of a specific component Σ\_r. In finance, you ultimately care about the total covariance relevant to forecasting. Your plan-of-record *suggests* an overlay that modifies a base covariance estimator, but your actual narrative must clearly state:  
   * Which component you treat as “target” under each design (week / DoW / vol)?  
   * Why the corrected spike eigenvalue should improve the *forecasted* Σ used in portfolio risk?  
2. **Design choice matters a lot more than you’re treating it.**  
   `docs/PLAN_OF_RECORD.md` bluntly says daily `week` (or weekly `oneway`) is closer to the “balanced random effects” intuition, and daily `dow` is secondary. Yet your headline run in `project_state/CURRENT_RESULTS.md` / `PROGRESS.md` is **daily DoW paper-v1**.  
   If DoW ends up being your main result, you need to explain why 5 groups is enough and why the components correspond to anything meaningful.  
3. **Heavy tails and temporal dependence are not solved by “robust MP edge” alone.**  
   Tyler/Huber help, but they don’t magically restore FJS assumptions. Your paper will need either:  
   * explicit robustness checks (block bootstrap / crisis slicing / subsampling), or  
   * a clear “this is a heuristic overlay; we calibrate FPR in synthetic regimes and stress test empirically.”

The good news: your infra is already built to support that “calibrate \+ stress test” style paper.

---

## **4\) Data, designs, and evaluation setup (what you have, what’s missing)**

### **What you have implemented**

* **Daily returns panel:** `data/returns_daily.csv` with 300 assets (2010‑01‑05 → 2024‑12‑31) tracked in `data/registry.json` (`project_state/DATAFLOW.md`).  
* **Factor data:** `data/factors/ff5mom_daily.csv` tracked in `data/factors/registry.json` (`project_state/DATAFLOW.md`).  
* **Group designs available:** `dow`, `week`, `vol`, `dowxvol` for daily (`experiments/eval/run.py`) and multiple designs for weekly runner (`project_state/EXPERIMENTS.md`).  
* **Portfolios:** EW and constrained MV with solver fail‑loud/skip discipline (documented in `project_state/ARCHITECTURE.md` and `project_state/CONFIG_REFERENCE.md`).

### **What’s missing / underdeveloped for a paper**

* **A single “primary design” that you commit to** and fully validate end‑to‑end.  
* **A stable interpretation of metrics:** right now you report absolute ΔMSE values like `+2.64e-11` and `-6.65e-13` (`project_state/CURRENT_RESULTS.md`). Without normalization or baseline magnitude, readers can’t tell if this is meaningful or rounding error.  
* **Economic significance framing:** even if your goal is “risk forecasting,” you still want to report things like realized variance reduction, turnover changes, constraint binding rates, etc.

---

## **5\) Codebase \+ infrastructure assessment (this matters for “is it worth it?”)**

### **The good**

* **This is not a toy repo.**  
  The combination of:  
  * tests across math/gating/calibration/eval/reporting (`project_state/TEST_COVERAGE.md`),  
  * run metadata discipline,  
  * window intersection validity and cap discipline,  
  * reproducible configs and registries  
    …is exactly what makes a project “real” in a research sense.  
* **Your “stop‑the‑line” rules are unusually mature** (`AGENTS.md`, `docs/PLAN_OF_RECORD.md`). This is a real differentiator for a resume.

### **The bad (and it’s fixable)**

* **Your “project\_state” docs are stale relative to current reality.**  
  Many core docs show “generated 2025‑12‑22” headers, while `PROGRESS.md` has 2026‑02 entries. That creates reviewer distrust: “what is the current code state?”  
* **You have at least one blatant reporting inconsistency that undermines trust.**  
  In both `project_state/CURRENT_RESULTS.md` and `PROGRESS.md`, the daily DoW paper-v1 entry says detection\_rate\_mean ≈ 0.04162 (4.162%) but then writes “(1751/1774 windows)” — which is \~98.7%, not 4.16%. That’s either a typo or a logic bug in how you’re logging counts. Either way: **this must be cleaned up before a paper.**

---

## **6\) Current results and what they actually imply**

I’m going to interpret your latest “validated” claims as what’s written in:

* `project_state/CURRENT_RESULTS.md`  
* `project_state/RESEARCH_NOTES.md`  
* the corresponding entries in `PROGRESS.md`

### **What the results currently say (plain English)**

1. **Daily DoW paper-v1 (uncapped, factor prewhitened) shows:**  
   * Overlay triggers on the order of a few percent of windows (≈4% detection rate is consistent with your other RC-lite notes).  
   * **QLIKE improves** for both EW and MV (ΔQLIKE negative for both).  
   * **MSE is mixed:** EW ΔMSE is harmful; MV ΔMSE is slightly beneficial (`project_state/CURRENT_RESULTS.md` lines 17–21).  
2. Interpretation: *if* this is real (and not a reporting artifact), it suggests:  
   * the overlay may help more for **optimization-sensitive** quantities (MV) than fixed portfolios (EW), which is plausible in high‑dim covariance land.  
   * but effect sizes look tiny in absolute scale; you need relative scale and statistical testing.  
3. **RC-lite sanity / earlier RC-lite runs show “overlay can be harmful.”**  
   * `project_state/CURRENT_RESULTS.md` notes rc-lite sanity where overlay effect is harmful (ΔMSE \> 0).  
   * This is actually good news scientifically: it means the method isn’t trivially “always helps,” so gating/calibration matters. But it increases the burden of proof.  
4. **Weekly / nested designs are still effectively “overlay off” in many cases.**  
   * `project_state/KNOWN_ISSUES.md` explicitly says weekly detection scarcity persists.  
   * If weekly/nested is your “theory-aligned” setup, that’s a problem.  
5. **The injection sensitivity harness is a major red flag right now.**  
   * `project_state/RESEARCH_NOTES.md` and `PROGRESS.md` show repeated “inject spikes of increasing µ” experiments where detection/acceptance stays at **zero**.  
   * Worse, the failure reasons are mostly in the t-vector / admissible root logic (`tvec_off_component`, `tvec_no_real_root`, `tvec_no_admissible_root`), meaning you’re not even reaching the “it found candidates but gated them out” stage.  
6. Interpretation: you currently do **not** have a reliable “unit test on reality” that your detector responds to known injected structure. That undermines credibility of any real-data performance claims, because your detector might be underpowered or mis-specified.

### **The brutal takeaway**

* You have **one plausible positive-ish signal** (DoW daily paper-v1 QLIKE improvements, slight MV improvement), but:  
  * it’s not yet demonstrated to be statistically or economically meaningful,  
  * the “counts vs rate” logging inconsistency hurts credibility,  
  * and your injection harness suggests the detection logic may not behave as expected.

So the project is real — but your current “scientific conclusion” is still unstable.

---

## **7\) Gaps, risks, and possible pivots (this is the decision point)**

### **The highest-leverage scientific risk**

**This could be a “beautiful method that rarely triggers or doesn’t matter in finance.”**  
That is *very plausible*. Modern shrinkage \+ factor modeling is hard to beat, and finance covariance isn’t a clean random-effects MANOVA world.

If that’s the outcome, the project can still succeed — but only if you frame it correctly and finish it as a rigorous negative/neutral result with strong diagnostics.

### **The highest-leverage engineering risk**

**Detector correctness is not convincingly established on realistic data.**  
Your injection harness staying flat-zero across µ suggests:

* target component mismatch,  
* numerical root/admissibility selection issues,  
* gating too strict,  
* or injection implementation not aligned with detection.

Until that’s resolved, your “paper v1” result is on shaky ground.

### **Potential pivots (ranked by “likely to produce a resume-worthy paper”)**

1. **Stay on this project, but narrow the claim to something you can actually prove.**  
   Example claim:  
   “A calibrated gated FJS-style spectral overlay triggers in X% of windows and is neutral-to-slightly-beneficial for min-variance risk forecasting under design Y; it is not beneficial for equal-weight; we characterize failure modes and when gating prevents harm.”  
   This is publishable as an arXiv preprint / student paper if executed cleanly.  
2. **Reframe the overlay as a *diagnostic tool* rather than a performance booster.**  
   If performance gains are tiny, you can still publish:  
   * “when does the detector fire,”  
   * what market regimes,  
   * what factor structure,  
   * how it correlates with portfolio instability / condition number / eigenvalue separation,  
   * whether it predicts when shrinkage underestimates risk.  
3. This is *more likely* to produce a coherent paper even if deltas are small.  
4. **Hard pivot away from FJS de-aliasing into “robust \+ factor \+ shrinkage benchmarking.”**  
   This is the safer path to “some improvement exists,” but it is less unique. You already have the infra for it; you could write a paper about robust covariance forecasting pipelines. It’ll be impressive engineering, but the novelty is weaker.

---

## **8\) Roadmap (what I would do if I were your coauthor)**

### **Short-term priorities (the “stop wasting time” list)**

1. **Fix the credibility breakers**  
   * Resolve the detection-rate/count inconsistency in `project_state/CURRENT_RESULTS.md` and the corresponding `PROGRESS.md` entry.  
   * Refresh `project_state/*` so the snapshot corresponds to current HEAD and current results.  
2. **Make injection sensitivity a hard gate**  
   * The inject-spike harness (`experiments/eval/inject_spike.py`) should show monotone detection/acceptance as µ increases under at least one realistic configuration.  
   * If it doesn’t, you do not proceed to more real-data grid runs. This is the single highest leverage debugging task.  
3. **Pick one primary design and commit**  
   * Based on your own `docs/PLAN_OF_RECORD.md`: make **daily `week`** (or weekly `oneway`) the primary.  
   * Treat DoW as ablation only unless you can justify it.  
4. **Force interpretable effect size reporting**  
   * Report relative improvements (percent) and baseline scale.  
   * Report conditional effects on windows where overlay actually changes Σ / weights.

### **Medium-term (turn into “paper v1”)**

1. **One clean experimental grid (small but defensible)**  
   * Designs: primary \+ one ablation  
   * Baselines: LW, OAS, RIE/QuEST, factor prewhitening on/off  
   * Edge modes: SCM vs Tyler  
   * Portfolios: EW \+ constrained MV  
   * Metrics: QLIKE \+ MSE \+ DM tests \+ turnover  
2. **Regime slicing**  
   * Predefine regimes (crisis windows like 2020 and 2022 are already referenced in configs and roadmap).  
   * Report safety: “overlay never catastrophically worsens risk in crisis windows” (or admit it does).  
3. **Write as you go**  
   * Start the paper document once injection is fixed and paper-v1 grid is pinned.  
   * Don’t wait for “perfect results” — you need a coherent narrative.

### **Longer-term (if you want “publishable beyond resume”)**

To get into a peer-reviewed venue, you probably need at least one of:

* a new theoretical extension (e.g., unbalanced designs, heavy-tail robustness theory),  
* or a clearly differentiated practical method (e.g., a new calibrated gating procedure with provable FPR control under realistic noise),  
* plus strong empirical evidence across datasets.

That’s a lot. For a resume paper, I’d aim for **arXiv-level rigor**, not journal-level novelty.

---

## **9\) Advisor-facing discussion points (what to take to Prof. Fan)**

Bring these as concrete questions / updates:

1. **Design choice sanity:**  
   “For finance mapping, is `week` (many groups, small replicates) the right primary design, and what component should we treat as ‘target’ Σ\_r for the de-aliasing map µ=λ/t\_r(λ,a)?”  
2. **Injection harness failure:**  
   “Our inject-spike on realistic windows is flat-zero detection even for large µ; failure reasons are mostly t-vector/off-component/admissible-root. This suggests a mismatch between injection mode and target component or a solver/admissibility issue. Can you sanity check what should happen theoretically under injection?”  
3. **What’s a defensible baseline set for a short paper?**  
   “We’re planning LW/OAS/RIE and factor prewhitening ablation. Is that enough, or should we also include a simple observed-factor covariance baseline?”  
4. **Paper claim negotiation:**  
   “If improvements are small, is a diagnostic/characterization paper (when does aliasing detector fire; how does it correlate with portfolio instability) still worthwhile?”

---

# **My blunt opinion: is this worth your time?**

### **Yes, there is real research here.**

It’s not fluff. You’re implementing a nontrivial high‑dimensional statistical idea, building synthetic calibration, and evaluating it in a leakage-aware rolling framework. That’s research.

### **The idea is valid — but finance relevance is not guaranteed.**

The key risk is not “the math is wrong,” it’s “the mechanism rarely matters once shrinkage/factors/robustness are applied,” or “it triggers too rarely / too unreliably.” Your own logs already show hints of that.

### **You can absolutely complete this into a resume-worthy paper *if you narrow scope hard*.**

A strong resume deliverable would be:

* an arXiv-style PDF (8–15 pages),  
* repo reproducibility,  
* synthetic replication \+ calibration,  
* one clean real-data grid with honest outcomes (positive, neutral, or negative),  
* and clear diagnostics about when it triggers and why.

### **If you keep going, impose a stop condition.**

If you cannot make **inject-spike sensitivity** behave (monotone detection vs µ) under at least one realistic configuration, then you do **not** have a stable experimental handle on the method. At that point, continuing to run more panels is just burning time. Either pivot the method or pivot the project.

### **If your goal is “impressive quant-adjacent project,” this is still one of the better bets — *but only if you stop expanding the platform and ship the paper*.**

Right now, your biggest enemy is “infinite infrastructure \+ no final claim.”

If you want, I can also propose a concrete “paper v1 outline” (section-by-section with the exact figures/tables to produce from your existing summary outputs) based on `docs/PLAN_OF_RECORD.md` and the runners you already have.
