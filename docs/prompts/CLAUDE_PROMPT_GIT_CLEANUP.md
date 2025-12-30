# AI Prompt: Resolving Pre-Deployment State Discrepancy in `kobe81_traderbot`

**Objective:**
Your task is to analyze a critical discrepancy found during the final verification of the `kobe81_traderbot` before its paper trading launch. You must explain the core software engineering principle at stake and guide the user on the necessary steps to resolve the issue and achieve a production-ready state.

**Background & Current Situation:**
We are conducting a final pre-flight check of the `kobe81_traderbot`. The goal is to certify it as "Paper Trading Ready - Grade A+."

**Summary of Findings:**
A verification process was just completed with mixed results:

1.  **The Good:** All 942 automated tests in the `tests/` directory have passed successfully. The latest commit `0e8704b` ("fix: Rename test helper to avoid pytest collection") is correctly pushed to the `main` branch on the remote repository.

2.  **The Critical Issue:** Despite the passing tests and the up-to-date branch, a `git status` check reveals that the local working directory is **not clean**. Several files have been modified since the last commit, and new, untracked files exist.

**Files with Uncommitted Modifications:**
*   `data/providers/alpaca_live.py`
*   `logs/daily_picks.csv`
*   `ml_features/conviction_scorer.py`
*   `scripts/submit_totd.py`
*   `state/cognitive/curiosity_state.json`
*   `state/cognitive/self_model.json`
*   `state/cognitive/semantic_rules.json`

---

### **Your Core Task**

**Part 1: Explain the "Why"**

First, you must explain to the user *why* this is a critical blocker for deployment. Your explanation should be clear, firm, and based on foundational software engineering principles. Key points to include:

*   **The Principle of Traceability:** Production code (including paper trading) MUST be traceable to a single, specific, and clean commit in the version control system. This is non-negotiable.
*   **The Risk of an Unclean State:** An unclean working directory means the code that will be executed is different from any version stored in git. This introduces unknown variables, makes debugging future issues nearly impossible, prevents reliable rollbacks, and invalidates the purpose of version control.
*   **Analogy:** You might compare it to a pharmaceutical company manufacturing a new drug. The formula used in the factory must *exactly* match the formula that was approved in the lab trials. Any deviation, no matter how small, is an unacceptable risk. Our passing tests are the "lab trials," but the modified files mean we are about to run a different "formula" in the factory.

**Part 2: Define the "What's Next"**

Second, you must provide a clear, actionable plan to resolve this situation. The goal is to methodically review every change and decide its fate.

1.  **Initial Investigation:** Advise the user that the first step is to inspect the changes. You should offer to run `git diff` on each of the modified files to show the user exactly what has been altered.

2.  **Triage and Decision:** For each modified file, guide the user through a decision process:
    *   **Is this change intentional and essential for paper trading?** (e.g., a last-minute bugfix or a necessary configuration change). If so, it **must be committed**.
    *   **Is this change an artifact of a previous test run?** (e.g., updated log files, state files, or cached data). If so, the change should likely be **discarded** using `git restore <file>`. Furthermore, the file path (or its pattern, like `logs/*.csv` or `state/`) should probably be added to the `.gitignore` file to prevent this from happening in the future.
    *   **Is this change an accidental or incomplete edit?** If so, it should be **discarded**.

3.  **Final State:** The end goal is a `git status` command that reports: "On branch main. Your branch is up to date with 'origin/main'. nothing to commit, working tree clean." Only when this state is achieved can the system be considered truly ready for deployment.

Begin by explaining the core principle and then offer to start the investigation by showing the user the diff for `data/providers/alpaca_live.py`.
