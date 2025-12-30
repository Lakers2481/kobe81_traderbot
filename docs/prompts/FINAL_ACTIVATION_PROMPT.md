# FINAL PROMPT: Immediate Paper Trading Activation Sequence (12-30-2025 09:21 AM)

**Objective:**
Execute the final activation sequence to bring the `kobe81_traderbot` online for paper trading, effective immediately. The system has passed all readiness checks, and the repository is in a clean, version-controlled state. This is the final "go-live" command.

**Execution Protocol:**
You are to perform the following steps sequentially. Confirm the successful completion of each step before proceeding to the next.

---

### **Step 1: Final Configuration Verification**

Read the main configuration file at `config/base.yaml`. Verify and explicitly state the values for the following critical parameters to ensure they are set for paper trading:
*   `is_live` (should be `false`)
*   `paper_trading` (should be `true`)
*   `kill_switch_engaged` (should be `false`)
*   The `broker` setting (should be `alpaca` or your configured paper trading broker)
*   The `oms_type` (should be `paper`)

**This is the final safety check. Do not proceed if any of these are incorrect.**

### **Step 2: Initialize a Clean State**

To ensure the bot starts fresh and is not influenced by old data from previous runs, perform the following file operations:
1.  Delete the contents of the `state/` directory. This clears any stale position, portfolio, or cognitive data.
2.  Delete the contents of the `logs/` directory to start with fresh log files.

**(This step is crucial for a true clean start. Await user confirmation after proposing this action.)**

### **Step 3: Launch the Main Trading Bot Process**

Execute the primary runner script to start the bot. The command must be run in the background so you can continue to monitor its output.
*   **Command:** `python scripts/runner.py --mode paper &`

After executing, confirm that the process has started and report the Process ID (PID).

### **Step 4: Real-Time Monitoring**

Immediately after launch, begin tailing the primary log file to monitor the bot's live activity. Based on the architecture, this is likely `logs/main.log` or a similarly named file.
*   **Command:** `tail -f logs/main.log` (or the correct primary log file)

Display the live log output as it comes in. The first few lines should indicate a successful startup, connection to the broker, and the scheduling of the first market scan.

---

Awaiting your final command to begin this sequence.
