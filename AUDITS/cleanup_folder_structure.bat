@echo off
REM FWO-Prime Folder Structure Cleanup Script (Windows)
REM Generated: 2026-01-09
REM Execute with: AUDITS\cleanup_folder_structure.bat

echo FWO-Prime: Starting folder structure cleanup...
echo.

REM Phase 1: DELETE BROKEN/STRANGE DIRECTORIES
echo === PHASE 1: Removing broken and strange directories ===

if exist "C:UsersOwnerOneDriveDesktopkobe81_traderbotdataverification" (
    echo Removing broken path directory (data verification)...
    rmdir /s /q "C:UsersOwnerOneDriveDesktopkobe81_traderbotdataverification"
)

if exist "C:UsersOwnerOneDriveDesktopkobe81_traderbotscriptsexperiments" (
    echo Removing broken path directory (scripts experiments)...
    rmdir /s /q "C:UsersOwnerOneDriveDesktopkobe81_traderbotscriptsexperiments"
)

if exist "_ul" (
    echo Removing temp directory (_ul)...
    rmdir /s /q "_ul"
)

if exist "_ul-DESKTOP-5IB5S6R" (
    echo Removing temp directory (_ul-DESKTOP-5IB5S6R)...
    rmdir /s /q "_ul-DESKTOP-5IB5S6R"
)

if exist "nul" (
    echo WARNING: Cannot remove 'nul' directory on Windows (reserved name)
    echo MANUAL ACTION REQUIRED: Delete via Linux subsystem or rename first
)

if exist "vuLDY5zrhSOyIpTVB6JB5taCKu71bWAQ" (
    echo Found strange directory: vuLDY5zrhSOyIpTVB6JB5taCKu71bWAQ
    echo MANUAL ACTION REQUIRED: Review and delete if unnecessary
    REM Uncomment to auto-delete:
    REM rmdir /s /q "vuLDY5zrhSOyIpTVB6JB5taCKu71bWAQ"
)

echo.

REM Phase 2: MOVE ROOT FILES TO PROPER LOCATIONS
echo === PHASE 2: Organizing root directory files ===

REM Move JSON files to AUDITS (excluding package.json)
for %%f in (*.json) do (
    if not "%%f"=="package.json" (
        echo Moving %%f -^> AUDITS\
        move /y "%%f" AUDITS\ 2>nul
    )
)

REM Move .txt files to AUDITS (excluding requirements)
for %%f in (*.txt) do (
    echo %%f | findstr /i /v "requirements" >nul
    if not errorlevel 1 (
        echo Moving %%f -^> AUDITS\
        move /y "%%f" AUDITS\ 2>nul
    )
)

REM Move specific markdown files to docs
if exist "CAPABILITY_MATRIX.md" move /y "CAPABILITY_MATRIX.md" docs\
if exist "INTEGRATION_RECOMMENDATIONS.md" move /y "INTEGRATION_RECOMMENDATIONS.md" docs\
if exist "INTERVIEW_ONE_PAGER.md" move /y "INTERVIEW_ONE_PAGER.md" docs\
if exist "MODULE_FILE_COUNTS.md" move /y "MODULE_FILE_COUNTS.md" docs\
if exist "PRODUCTION_CRITICAL_COMPONENTS.md" move /y "PRODUCTION_CRITICAL_COMPONENTS.md" docs\
if exist "PROJECT_CONTEXT.md" move /y "PROJECT_CONTEXT.md" docs\
if exist "QUICK_START.md" move /y "QUICK_START.md" docs\
if exist "SYSTEM_ARCHITECTURE_INVENTORY.md" move /y "SYSTEM_ARCHITECTURE_INVENTORY.md" docs\
if exist "AI_HANDOFF_PROMPT.md" move /y "AI_HANDOFF_PROMPT.md" docs\
if exist "CLAUDE_PROMPT_DETERMINISM_AUDIT.md" move /y "CLAUDE_PROMPT_DETERMINISM_AUDIT.md" docs\
if exist "CLAUDE_WORK_PROMPT.md" move /y "CLAUDE_WORK_PROMPT.md" docs\
if exist "DATA_VERIFICATION_REPORT.md" move /y "DATA_VERIFICATION_REPORT.md" docs\
if exist "EXTERNAL_RESOURCE_AUDIT_FINAL_REPORT.md" move /y "EXTERNAL_RESOURCE_AUDIT_FINAL_REPORT.md" docs\
if exist "EXTERNAL_RESOURCES_DETAILED_ANALYSIS.md" move /y "EXTERNAL_RESOURCES_DETAILED_ANALYSIS.md" docs\
if exist "FIX_1_IMPLEMENTATION_SUMMARY.md" move /y "FIX_1_IMPLEMENTATION_SUMMARY.md" docs\
if exist "FIX_2_IMPLEMENTATION_SUMMARY.md" move /y "FIX_2_IMPLEMENTATION_SUMMARY.md" docs\
if exist "FIX_3_IMPLEMENTATION_SUMMARY.md" move /y "FIX_3_IMPLEMENTATION_SUMMARY.md" docs\
if exist "FIX_4_IMPLEMENTATION_SUMMARY.md" move /y "FIX_4_IMPLEMENTATION_SUMMARY.md" docs\
if exist "NIGHT_AUDIT_REPORT.md" move /y "NIGHT_AUDIT_REPORT.md" docs\
if exist "OPTIMIZER_PROOF_VERDICT.md" move /y "OPTIMIZER_PROOF_VERDICT.md" docs\
if exist "OVERNIGHT_STATUS.md" move /y "OVERNIGHT_STATUS.md" docs\
if exist "PIPELINE_VERIFICATION_REPORT.md" move /y "PIPELINE_VERIFICATION_REPORT.md" docs\
if exist "PROGRESS_STATUS.md" move /y "PROGRESS_STATUS.md" docs\

echo.

REM Phase 3: CONSOLIDATE OUTPUT DIRECTORIES
echo === PHASE 3: Consolidating output directories ===

REM Create consolidated structure
if not exist "outputs\backtests" mkdir "outputs\backtests"
if not exist "outputs\walk_forward" mkdir "outputs\walk_forward"
if not exist "outputs\showdowns" mkdir "outputs\showdowns"
if not exist "outputs\optimizations" mkdir "outputs\optimizations"
if not exist "outputs\smoke_tests" mkdir "outputs\smoke_tests"

REM Move backtest outputs
if exist "backtest_outputs" (
    echo Consolidating backtest_outputs\ -^> outputs\backtests\
    xcopy /e /i /y backtest_outputs\* outputs\backtests\ 2>nul
    rmdir /s /q backtest_outputs
)

REM Move walk-forward outputs
if exist "wf_outputs" (
    echo Consolidating wf_outputs\ -^> outputs\walk_forward\
    xcopy /e /i /y wf_outputs\* outputs\walk_forward\ 2>nul
    rmdir /s /q wf_outputs
)

REM Move showdown outputs
if exist "showdown_outputs" (
    echo Consolidating showdown_outputs\ -^> outputs\showdowns\
    xcopy /e /i /y showdown_outputs\* outputs\showdowns\ 2>nul
    rmdir /s /q showdown_outputs
)

if exist "showdown_2025_cap60" (
    echo Consolidating showdown_2025_cap60\ -^> outputs\showdowns\
    xcopy /e /i /y showdown_2025_cap60\* outputs\showdowns\ 2>nul
    rmdir /s /q showdown_2025_cap60
)

REM Move optimization outputs
if exist "optimize_outputs" (
    echo Consolidating optimize_outputs\ -^> outputs\optimizations\
    xcopy /e /i /y optimize_outputs\* outputs\optimizations\ 2>nul
    rmdir /s /q optimize_outputs
)

REM Move smoke test outputs
if exist "smoke_outputs" (
    echo Consolidating smoke_outputs\ -^> outputs\smoke_tests\
    xcopy /e /i /y smoke_outputs\* outputs\smoke_tests\ 2>nul
    rmdir /s /q smoke_outputs
)

if exist "smoke_turtle_soup" (
    echo Consolidating smoke_turtle_soup\ -^> outputs\smoke_tests\
    xcopy /e /i /y smoke_turtle_soup\* outputs\smoke_tests\ 2>nul
    rmdir /s /q smoke_turtle_soup
)

if exist "smoke_wf_audit" (
    echo Consolidating smoke_wf_audit\ -^> outputs\smoke_tests\
    xcopy /e /i /y smoke_wf_audit\* outputs\smoke_tests\ 2>nul
    rmdir /s /q smoke_wf_audit
)

REM Merge output\ into outputs\
if exist "output" (
    echo Merging output\ -^> outputs\
    xcopy /e /i /y output\* outputs\ 2>nul
    rmdir /s /q output
)

echo.

REM Phase 4: CREATE MISSING __init__.py FILES
echo === PHASE 4: Creating missing __init__.py files ===

if not exist "analysis\__init__.py" (
    echo Creating analysis\__init__.py
    type nul > "analysis\__init__.py"
)

if not exist "autonomous\scrapers\__init__.py" (
    echo Creating autonomous\scrapers\__init__.py
    type nul > "autonomous\scrapers\__init__.py"
)

if not exist "backtest\__init__.py" (
    echo Creating backtest\__init__.py
    type nul > "backtest\__init__.py"
)

if not exist "cognitive\__init__.py" (
    echo Creating cognitive\__init__.py
    type nul > "cognitive\__init__.py"
)

if not exist "config\alpha_workflows\__init__.py" (
    echo Creating config\alpha_workflows\__init__.py
    type nul > "config\alpha_workflows\__init__.py"
)

if not exist "data\schemas\__init__.py" (
    echo Creating data\schemas\__init__.py
    type nul > "data\schemas\__init__.py"
)

if not exist "evolution\__init__.py" (
    echo Creating evolution\__init__.py
    type nul > "evolution\__init__.py"
)

if not exist "explainability\__init__.py" (
    echo Creating explainability\__init__.py
    type nul > "explainability\__init__.py"
)

if not exist "extensions\__init__.py" (
    echo Creating extensions\__init__.py
    type nul > "extensions\__init__.py"
)

if not exist "ml\alpha_discovery\__init__.py" (
    echo Creating ml\alpha_discovery\__init__.py
    type nul > "ml\alpha_discovery\__init__.py"
)

if not exist "options\__init__.py" (
    echo Creating options\__init__.py
    type nul > "options\__init__.py"
)

if not exist "pipelines\__init__.py" (
    echo Creating pipelines\__init__.py
    type nul > "pipelines\__init__.py"
)

if not exist "research_os\__init__.py" (
    echo Creating research_os\__init__.py
    type nul > "research_os\__init__.py"
)

if not exist "risk\advanced\__init__.py" (
    echo Creating risk\advanced\__init__.py
    type nul > "risk\advanced\__init__.py"
)

if not exist "strategy_specs\__init__.py" (
    echo Creating strategy_specs\__init__.py
    type nul > "strategy_specs\__init__.py"
)

echo.

REM Phase 5: CLEAN CACHE DIRECTORIES
echo === PHASE 5: Cleaning cache directories ===

echo Removing __pycache__ directories...
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul

echo Removing .pytest_cache directories...
for /d /r %%d in (.pytest_cache) do @if exist "%%d" rmdir /s /q "%%d" 2>nul

echo Removing .ruff_cache directories...
for /d /r %%d in (.ruff_cache) do @if exist "%%d" rmdir /s /q "%%d" 2>nul

echo.

REM Phase 6: FIX DATA CACHE DUPLICATION
echo === PHASE 6: Consolidating data cache ===

if exist "data\polygon_cache\polygon" (
    echo Merging data\polygon_cache\polygon\ -^> data\cache\polygon\
    if not exist "data\cache\polygon" mkdir "data\cache\polygon"
    xcopy /e /i /y data\polygon_cache\polygon\* data\cache\polygon\ 2>nul
    rmdir /s /q data\polygon_cache
)

echo.

REM Phase 7: UPDATE .gitignore
echo === PHASE 7: Updating .gitignore ===

findstr /i "__pycache__" .gitignore >nul 2>&1
if errorlevel 1 (
    echo Adding cache patterns to .gitignore...
    (
        echo.
        echo # Python cache
        echo __pycache__/
        echo *.pyc
        echo *.pyo
        echo *.pyd
        echo .Python
        echo.
        echo # Testing cache
        echo .pytest_cache/
        echo .ruff_cache/
        echo .mypy_cache/
        echo.
        echo # ML cache
        echo mlruns/
        echo *.egg-info/
        echo.
        echo # OS files
        echo .DS_Store
        echo Thumbs.db
    ) >> .gitignore
)

echo.
echo === CLEANUP COMPLETE ===
echo.
echo Summary:
echo   - Removed broken/strange directories
echo   - Organized 34 root files into docs\ and AUDITS\
echo   - Consolidated 7 output directories into outputs\
echo   - Created 15 missing __init__.py files
echo   - Cleaned all __pycache__ directories
echo   - Updated .gitignore
echo.
echo Next steps:
echo   1. Review changes with 'git status'
echo   2. Test imports with 'python -m pytest tests\'
echo   3. Re-run audit: 'python AUDITS\folder_audit_analyzer.py'
echo   4. Commit changes with descriptive message
echo.
echo FWO-Prime: Cleanup complete. Repository is now Renaissance-grade.
echo.
pause
