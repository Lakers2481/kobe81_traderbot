#!/usr/bin/env python3
"""
Research OS CLI - Command-line interface for the DISCOVER -> RESEARCH -> ENGINEER workflow.

This is the HUMAN INTERFACE for the Research OS.

Commands:
    discoveries   - View discovered knowledge cards
    proposals     - View research proposals
    approvals     - View pending approvals
    approve       - Approve a proposal (HUMAN ACTION)
    reject        - Reject a proposal (HUMAN ACTION)
    discover      - Run discovery cycle
    research      - Run research on a knowledge card
    propose       - Propose engineering change
    status        - Show full system status

Usage:
    python scripts/research_os_cli.py status
    python scripts/research_os_cli.py discoveries --status validated
    python scripts/research_os_cli.py approvals --pending
    python scripts/research_os_cli.py approve --id <request_id> --approver "John Doe"
    python scripts/research_os_cli.py reject --id <request_id> --reason "Overfitting suspected"
    python scripts/research_os_cli.py discover --cycle
    python scripts/research_os_cli.py research --card <card_id>

CRITICAL: approve and reject are HUMAN ACTIONS only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research_os import (
    ResearchOSOrchestrator,
    KnowledgeCard,
    CardStatus,
    ResearchProposal,
    ProposalStatus,
    ApprovalGate,
    APPROVE_LIVE_ACTION,
)
from research_os.orchestrator import get_research_os


def cmd_status(args):
    """Show full system status."""
    os = get_research_os()
    print(os.status())


def cmd_discoveries(args):
    """List knowledge cards."""
    os = get_research_os()
    cards = os.card_store.list_all()

    if args.status:
        try:
            status = CardStatus(args.status)
            cards = [c for c in cards if c.status == status]
        except ValueError:
            print(f"Invalid status: {args.status}")
            print(f"Valid: {', '.join(s.value for s in CardStatus)}")
            return

    if not cards:
        print("No knowledge cards found.")
        return

    print(f"Found {len(cards)} knowledge cards:\n")
    for card in cards:
        print(card.summary())
        print()


def cmd_proposals(args):
    """List research proposals."""
    os = get_research_os()
    proposals = os.proposal_store.list_all()

    if args.status:
        try:
            status = ProposalStatus(args.status)
            proposals = [p for p in proposals if p.status == status]
        except ValueError:
            print(f"Invalid status: {args.status}")
            print(f"Valid: {', '.join(s.value for s in ProposalStatus)}")
            return

    if not proposals:
        print("No proposals found.")
        return

    print(f"Found {len(proposals)} proposals:\n")
    for proposal in proposals:
        print(proposal.summary())
        print()


def cmd_approvals(args):
    """List pending approvals."""
    gate = ApprovalGate()

    if args.pending:
        pending = gate.get_pending()
        if not pending:
            print("No pending approvals.")
            return

        print(f"=== {len(pending)} Pending Approvals ===\n")
        for p in pending:
            print(f"Request ID: {p['request_id']}")
            print(f"  Title: {p.get('title', 'N/A')}")
            print(f"  Change: {p.get('change_type', 'N/A')} in {p.get('target_file', 'N/A')}")
            print(f"  Submitted: {p.get('submitted_at', 'N/A')}")
            print()
    elif args.approved:
        approved = gate.get_approved()
        if not approved:
            print("No approved changes awaiting implementation.")
            return

        print(f"=== {len(approved)} Approved (Awaiting Implementation) ===\n")
        print(f"NOTE: APPROVE_LIVE_ACTION = {APPROVE_LIVE_ACTION}")
        if not APPROVE_LIVE_ACTION:
            print("       To implement, manually set APPROVE_LIVE_ACTION = True\n")
        for a in approved:
            print(f"Request ID: {a['request_id']}")
            print(f"  Title: {a.get('title', 'N/A')}")
            print(f"  Approved by: {a.get('approved_by', 'N/A')}")
            print(f"  Approved at: {a.get('approved_at', 'N/A')}")
            print()
    else:
        print(gate.summary())


def cmd_approve(args):
    """
    Approve a proposal.

    THIS IS A HUMAN ACTION.
    """
    if not args.id:
        print("ERROR: --id is required")
        return
    if not args.approver:
        print("ERROR: --approver is required (your name)")
        return

    gate = ApprovalGate()
    success = gate.approve(args.id, args.approver, args.notes or "")

    if success:
        print(f"APPROVED: {args.id}")
        print(f"Approved by: {args.approver}")
        if args.notes:
            print(f"Notes: {args.notes}")
        print()
        print("To implement this change:")
        print("  1. Manually set APPROVE_LIVE_ACTION = True in research_os/approval_gate.py")
        print("  2. Run: python scripts/research_os_cli.py implement --id {args.id} --implementer <name>")
    else:
        print(f"FAILED: Could not approve {args.id}")
        print("Check that the request exists in pending approvals.")


def cmd_reject(args):
    """
    Reject a proposal.

    THIS IS A HUMAN ACTION.
    """
    if not args.id:
        print("ERROR: --id is required")
        return
    if not args.reason:
        print("ERROR: --reason is required")
        return

    gate = ApprovalGate()
    rejector = args.rejector or "Unknown"
    success = gate.reject(args.id, rejector, args.reason)

    if success:
        print(f"REJECTED: {args.id}")
        print(f"Rejected by: {rejector}")
        print(f"Reason: {args.reason}")
    else:
        print(f"FAILED: Could not reject {args.id}")
        print("Check that the request exists in pending approvals.")


def cmd_discover(args):
    """Run discovery cycle."""
    if not args.cycle:
        print("Use --cycle to run a discovery cycle")
        return

    os = get_research_os()
    print("Running discovery cycle...")
    print()

    result = os.run_discovery_cycle()

    print("=== Discovery Cycle Results ===")
    print(f"Hypotheses generated: {result.hypotheses_generated}")
    print(f"Parameters explored: {result.parameters_explored}")
    print(f"Patterns found: {result.patterns_found}")
    print(f"External sources checked: {result.external_sources_checked}")
    print(f"Knowledge cards created: {result.knowledge_cards_created}")

    if result.errors:
        print()
        print("Errors:")
        for e in result.errors:
            print(f"  - {e}")


def cmd_research(args):
    """Run research on a knowledge card."""
    if not args.card:
        print("Use --card <card_id> to specify which card to research")
        return

    os = get_research_os()
    print(f"Running research cycle for: {args.card}")
    print()

    result = os.run_research_cycle(args.card)

    print("=== Research Cycle Results ===")
    print(f"Experiments run: {result.experiments_run}")
    print(f"Validations passed: {result.validations_passed}")
    print(f"Validations failed: {result.validations_failed}")
    print(f"Knowledge cards updated: {result.knowledge_cards_updated}")
    print(f"Proposals created: {result.proposals_created}")

    if result.errors:
        print()
        print("Errors:")
        for e in result.errors:
            print(f"  - {e}")


def cmd_propose(args):
    """Propose engineering change from validated card."""
    if not args.card:
        print("Use --card <card_id> to specify which validated card to propose")
        return

    os = get_research_os()
    print(f"Creating proposal for: {args.card}")

    request_id = os.propose_engineering_change(args.card)

    if request_id:
        print(f"Proposal submitted: {request_id}")
        print()
        print("Next steps:")
        print("  1. Review: python scripts/research_os_cli.py approvals --pending")
        print(f"  2. Approve: python scripts/research_os_cli.py approve --id {request_id} --approver <name>")
    else:
        print("Failed to create proposal. Check that the card exists and is validated.")


def cmd_implement(args):
    """Implement an approved change."""
    if not args.id:
        print("ERROR: --id is required")
        return
    if not args.implementer:
        print("ERROR: --implementer is required (your name)")
        return

    os = get_research_os()

    try:
        success = os.implement_approved_change(args.id, args.implementer)
        if success:
            print(f"IMPLEMENTED: {args.id}")
            print(f"Implemented by: {args.implementer}")
        else:
            print(f"FAILED: Could not implement {args.id}")
    except Exception as e:
        print(f"SAFETY ERROR: {e}")
        print()
        print("To implement changes:")
        print("  1. Ensure the proposal is APPROVED")
        print("  2. Manually set APPROVE_LIVE_ACTION = True in research_os/approval_gate.py")
        print("  3. Re-run this command")


def main():
    parser = argparse.ArgumentParser(
        description="Research OS CLI - DISCOVER -> RESEARCH -> ENGINEER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/research_os_cli.py status
  python scripts/research_os_cli.py discoveries --status validated
  python scripts/research_os_cli.py approvals --pending
  python scripts/research_os_cli.py approve --id req_abc123 --approver "John Doe"
  python scripts/research_os_cli.py reject --id req_abc123 --reason "Overfitting"
  python scripts/research_os_cli.py discover --cycle
  python scripts/research_os_cli.py research --card kc_xyz789

CRITICAL: 'approve' and 'reject' are HUMAN ACTIONS only.
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    subparsers.add_parser("status", help="Show system status")

    # discoveries
    disc_parser = subparsers.add_parser("discoveries", help="List knowledge cards")
    disc_parser.add_argument("--status", help="Filter by status")

    # proposals
    prop_parser = subparsers.add_parser("proposals", help="List proposals")
    prop_parser.add_argument("--status", help="Filter by status")

    # approvals
    appr_parser = subparsers.add_parser("approvals", help="View approval status")
    appr_parser.add_argument("--pending", action="store_true", help="Show pending")
    appr_parser.add_argument("--approved", action="store_true", help="Show approved")

    # approve (HUMAN ACTION)
    approve_parser = subparsers.add_parser("approve", help="Approve a proposal (HUMAN)")
    approve_parser.add_argument("--id", required=True, help="Request ID")
    approve_parser.add_argument("--approver", required=True, help="Your name")
    approve_parser.add_argument("--notes", help="Approval notes")

    # reject (HUMAN ACTION)
    reject_parser = subparsers.add_parser("reject", help="Reject a proposal (HUMAN)")
    reject_parser.add_argument("--id", required=True, help="Request ID")
    reject_parser.add_argument("--reason", required=True, help="Rejection reason")
    reject_parser.add_argument("--rejector", help="Your name")

    # discover
    disc_cycle_parser = subparsers.add_parser("discover", help="Run discovery")
    disc_cycle_parser.add_argument("--cycle", action="store_true", help="Run cycle")

    # research
    research_parser = subparsers.add_parser("research", help="Research a card")
    research_parser.add_argument("--card", help="Knowledge card ID")

    # propose
    propose_parser = subparsers.add_parser("propose", help="Propose change")
    propose_parser.add_argument("--card", help="Knowledge card ID")

    # implement
    impl_parser = subparsers.add_parser("implement", help="Implement approved")
    impl_parser.add_argument("--id", required=True, help="Request ID")
    impl_parser.add_argument("--implementer", required=True, help="Your name")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "discoveries":
        cmd_discoveries(args)
    elif args.command == "proposals":
        cmd_proposals(args)
    elif args.command == "approvals":
        cmd_approvals(args)
    elif args.command == "approve":
        cmd_approve(args)
    elif args.command == "reject":
        cmd_reject(args)
    elif args.command == "discover":
        cmd_discover(args)
    elif args.command == "research":
        cmd_research(args)
    elif args.command == "propose":
        cmd_propose(args)
    elif args.command == "implement":
        cmd_implement(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
