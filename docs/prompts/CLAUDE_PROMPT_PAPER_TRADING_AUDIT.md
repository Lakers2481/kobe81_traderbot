# Final Pre-Flight Audit for Paper Trading: kobe81_traderbot

**Objective:** Conduct a final, intensive audit of the `kobe81_traderbot` codebase to ensure it is 1000% ready for paper trading deployment by tomorrow. The primary goal is to identify and flag any critical blockers, inconsistencies, or configuration errors that would jeopardize a safe and successful paper trading launch.

**Background & Context:**
A preliminary analysis has been completed, revealing a sophisticated, well-documented, and highly configurable trading bot. The architecture is layered, with a standout "cognitive" module that enables advanced, "System 2" decision-making.

However, a significant potential risk was identified: the existence of two separate execution paths for backtesting.
1.  A simplified, standalone script: `scripts/backtest_dual_strategy.py`
2.  A full-featured, integrated engine: `backtest/engine.py`

This divergence is the single biggest threat to deployment, as the backtested results may not accurately reflect how the strategy will perform within the full application framework.

**Your Core Task & Priority Checklist:**
You are to act as the final quality assurance gate before deployment. Your mission is to meticulously review the codebase and configuration to answer the following questions. Focus on verification and identifying discrepancies.

**1. Strategy Logic & Execution Path Verification:**
   - **Crucial Question:** Does the core trading logic (signal generation, position sizing, entry/exit conditions) in `scripts/backtest_dual_strategy.py` perfectly match the logic implemented within the full application stack that will be used for paper trading?
   - **Action:** Compare the strategy implementation in the simplified script against the components it would interact with in a live run (e.g., `cognitive_brain`, `oms`, `portfolio`, `risk` modules).
   - **Deliverable:** State definitively whether they are identical. If not, pinpoint every single discrepancy.

**2. Configuration Audit for Paper Trading:**
   - **Crucial Question:** Are all configuration files (`config/*.yaml`, `config/settings.json`, etc.) set correctly for a *paper trading* environment?
   - **Action:** Review all configuration files. Pay special attention to:
     - `base.yaml`: Ensure features like `paper_trading_mode` are enabled and backtesting-specific features are disabled.
     - API endpoints, keys, and secrets (check that they are being loaded from a secure environment, not hardcoded).
     - File paths for logs, data, and state to ensure they point to the correct production locations.
     - Brokerage-specific settings.
   - **Deliverable:** Confirm that all settings are correctly configured for paper trading. List any settings that are incorrect or ambiguous.

**3. Risk & Compliance System Check:**
   - **Crucial Question:** Are all risk management and compliance systems enabled and configured with sane, protective values?
   - **Action:** Review the configurations for modules like `risk`, `compliance`, and `core`. Specifically check:
     - The `kill_switch` mechanism (`core/kill_switch.py`). Is it active?
     - Drawdown limits, position size limits, and other parameters in the `risk` module.
     - The `prohibited_list` in the `compliance` module.
     - The `SymbolicReasoner` rules (`config/symbolic_rules.yaml`) and `DynamicPolicyGenerator` (`config/trading_policies.yaml`). Are they set to a safe, baseline state for the initial launch (e.g., not "crisis mode")?
   - **Deliverable:** Provide a summary of the key active risk parameters and confirm they are set to reasonable, protective levels for an initial paper trading run.

**4. Data Pipeline Integrity:**
   - **Crucial Question:** Is the data pipeline configured to receive live or near-live market data appropriate for paper trading?
   - **Action:** Examine the `data` providers and related configurations. Verify that the system is not pointing to static, historical data files used for backtesting.
   - **Deliverable:** Confirm the data pipeline is configured for a live environment.

**Final Deliverable:**
Produce a concise, clear "Pre-Flight Check Report." For each of the four points above, give a "GO" or "NO-GO" status. For any "NO-GO," you must provide a precise, actionable list of the changes required to achieve a "GO" status. The report should conclude with a final, unambiguous recommendation: "DEPLOY" or "DO NOT DEPLOY."
