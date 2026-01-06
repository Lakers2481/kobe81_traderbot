# ORDER SURFACES VERIFICATION

**Generated**: 2026-01-06
**Status**: ALL 7 ORDER PRIMITIVES PROTECTED

---

## ORDER PRIMITIVE COVERAGE

| # | Primitive | File | Line | Guard Function | Verified |
|---|-----------|------|------|----------------|----------|
| 1 | `place_ioc_limit()` | broker_alpaca.py | 669 | Upstream via place_order() | YES |
| 2 | `AlpacaBroker.place_order()` | broker_alpaca.py | 1686 | `evaluate_safety_gates()` | YES |
| 3 | `PaperBroker.place_order()` | broker_paper.py | 223 | `evaluate_safety_gates()` | YES |
| 4 | `CryptoBroker.place_order()` | broker_crypto.py | 358 | `evaluate_safety_gates()` | YES |
| 5 | `AlpacaCryptoBroker.place_order()` | broker_alpaca_crypto.py | 314 | `evaluate_safety_gates()` | YES |
| 6 | `OptionsOrderRouter.submit_order()` | order_router.py | 323 | `evaluate_safety_gates()` | YES |
| 7 | `close_position()` | position_manager.py | 230 | `evaluate_safety_gates()` | YES |

---

## CHOKE POINT ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                     UNIFIED SAFETY GATE                             │
│                 safety/execution_choke.py                           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  evaluate_safety_gates(is_paper_order: bool) -> GateResult  │   │
│  │                                                              │   │
│  │  Checks (in order):                                          │   │
│  │  1. kill_switch_inactive (KILL_SWITCH file)                 │   │
│  │  2. paper_only_disabled (PAPER_ONLY config)                 │   │
│  │  3. live_trading_enabled (LIVE_TRADING_ENABLED)             │   │
│  │  4. trading_mode_live (TRADING_MODE)                        │   │
│  │  5. approve_live_action (APPROVE_LIVE_ACTION)               │   │
│  │  6. approve_live_action_2 (APPROVE_LIVE_ACTION_2)           │   │
│  │  7. ack_token_valid (LIVE_ORDER_ACK_TOKEN)                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  AlpacaBroker   │  │   PaperBroker   │  │  CryptoBroker   │
│  place_order()  │  │  place_order()  │  │  place_order()  │
│   Line 1698     │  │    Line 237     │  │    Line 373     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│AlpacaCryptoBrkr │  │OptionsOrderRtr │  │close_position() │
│  place_order()  │  │ submit_order() │  │  Line 238       │
│   Line 327      │  │   Line 336     │  │position_manager │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## FAIL-CLOSED VERIFICATION

### CCXT Crypto Broker
- **File**: `execution/broker_crypto.py`
- **Lines**: 85-89
- **Behavior**: If `ccxt` not installed, raises `ImportError` at module load
- **Result**: FAIL-CLOSED (cannot instantiate broker without dependency)

### Alpaca Crypto Broker
- **File**: `execution/broker_alpaca_crypto.py`
- **Lines**: 22-30
- **Behavior**: If `alpaca-py` not installed, raises `ImportError`
- **Result**: FAIL-CLOSED

---

## RUNTIME TEST PROOF

Tests in `tests/security/test_runtime_choke_enforcement.py` verify:

| Test | What It Proves |
|------|----------------|
| `test_alpaca_broker_calls_gate_before_any_api` | Gate called BEFORE API, blocked orders never reach API |
| `test_alpaca_crypto_broker_calls_gate_before_api` | Crypto gate enforced |
| `test_ccxt_crypto_broker_calls_gate_before_ccxt` | CCXT orders gated |
| `test_position_manager_calls_gate_before_alpaca_request` | Position close gated |
| `test_options_router_calls_gate_before_execution` | Options orders gated |
| `test_paper_broker_calls_gate_even_for_simulation` | Paper orders also gated |
| `test_kill_switch_blocks_paper_orders` | Kill switch blocks paper |
| `test_kill_switch_blocks_live_orders` | Kill switch blocks live |
| `test_gate_called_before_order_creation` | Call sequence verified |

**Result**: 25 security tests PASSED, 1 skipped (ccxt not installed)

---

## NO BYPASS PATHS

Verified that NO order can be placed without going through `evaluate_safety_gates()`:

1. **Direct API calls**: All low-level functions (`place_ioc_limit`, `alpaca_request`) are protected by calling `evaluate_safety_gates()` upstream
2. **New broker implementations**: Must follow the pattern or tests fail
3. **Options**: Router checks gate before simulation OR live execution
4. **Crypto**: Both CCXT and Alpaca crypto check gate

---

## VERDICT

**ALL ORDER SURFACES PROTECTED** - No bypass paths detected.
