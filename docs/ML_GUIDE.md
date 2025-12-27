Kobe81 Traderbot â€” ML + Sentiment Guide

Daily (Topâ€‘3 + TOTD with ML)
- Update sentiment cache:
  python scripts/update_sentiment_cache.py --universe data/universe/optionable_liquid_900.csv --date 2025-01-03 --dotenv ./.env
- Scan, score and write Topâ€‘3 + TOTD:
  python scripts/scan.py --dotenv ./.env --strategy all --cap 900 --top3 --ml --ensure-top3 --min-conf 0.60 --date 2025-01-03
- Submit Trade of the Day (confidenceâ€‘gated):
  python scripts/submit_totd.py --dotenv ./.env

Oneâ€‘shot daily pipeline:
  python scripts/run_daily_pipeline.py --dotenv ./.env --cap 900 --ensure-top3

Weekly retrain + promote
- Build dataset from WF trades:
  python scripts/build_signal_dataset.py --wfdir wf_outputs --dotenv ./.env
- Train candidate models:
  python scripts/train_meta.py
- Promote if better:
  python scripts/promote_models.py --min-delta 0.01 --min-test 100

Endâ€‘toâ€‘end weekly pipeline:
  python scripts/run_weekly_training.py --wfdir wf_outputs --dotenv ./.env

24/7 Master Scheduler (2K28 schedule times)
- Runs ET schedule (preâ€‘game, news, first scan, midâ€‘day scans, swing, EOD learning):
  python scripts/scheduler_kobe.py --dotenv ./.env --universe data/universe/optionable_liquid_900.csv --cap 900 --min-conf 0.60 --tick-seconds 20

Notes
- Deployed models: state/models/deployed/meta_*.pkl (used by scanner)
- Candidates: state/models/candidates/meta_*.pkl (awaiting promotion)
- Confidence blend: conf_score = 0.8Ã—ML + 0.2Ã—sentiment (if available)
- TOTD threshold: --min-conf (default 0.55)

