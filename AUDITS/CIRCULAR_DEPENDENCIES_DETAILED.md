# CIRCULAR DEPENDENCY ANALYSIS - DETAILED REPORT

---

## Module: `cognitive`

**Total modules:** 33

**Circular dependencies found:** 1

### Circular Import Chains

#### Cycle 1

```
cognitive
  |
  v
cognitive
  |
  v
cognitive (back to start)
```

**Import details:**

- `cognitive` imports `cognitive` at L38, L39, L40
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\cognitive\__init__.py`
- `cognitive` imports `cognitive` at L38, L39, L40
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\cognitive\__init__.py`

**Recommended fix:**

1. Extract common interface/protocol from `cognitive` and `cognitive`
2. Move shared types to a new `cognitive.types` module
3. Use TYPE_CHECKING for type hints

```python
# Example fix using TYPE_CHECKING:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognitive import SomeClass

# Use 'SomeClass' as string in type hints
def func(obj: 'SomeClass') -> None:
    ...
```

---

## Module: `altdata`

**Total modules:** 10

**Circular dependencies found:** 1

### Circular Import Chains

#### Cycle 1

```
altdata
  |
  v
altdata
  |
  v
altdata (back to start)
```

**Import details:**

- `altdata` imports `altdata` at L4, L7, L15
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\altdata\__init__.py`
- `altdata` imports `altdata` at L4, L7, L15
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\altdata\__init__.py`

**Recommended fix:**

1. Extract common interface/protocol from `altdata` and `altdata`
2. Move shared types to a new `altdata.types` module
3. Use TYPE_CHECKING for type hints

```python
# Example fix using TYPE_CHECKING:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from altdata import SomeClass

# Use 'SomeClass' as string in type hints
def func(obj: 'SomeClass') -> None:
    ...
```

---

## Module: `analytics`

**Total modules:** 12

**Circular dependencies found:** 1

### Circular Import Chains

#### Cycle 1

```
analytics
  |
  v
analytics
  |
  v
analytics (back to start)
```

**Import details:**

- `analytics` imports `analytics` at L11, L16, L21
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\analytics\__init__.py`
- `analytics` imports `analytics` at L11, L16, L21
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\analytics\__init__.py`

**Recommended fix:**

1. Extract common interface/protocol from `analytics` and `analytics`
2. Move shared types to a new `analytics.types` module
3. Use TYPE_CHECKING for type hints

```python
# Example fix using TYPE_CHECKING:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from analytics import SomeClass

# Use 'SomeClass' as string in type hints
def func(obj: 'SomeClass') -> None:
    ...
```

---

## Module: `research`

**Total modules:** 13

**Circular dependencies found:** 1

### Circular Import Chains

#### Cycle 1

```
research
  |
  v
research
  |
  v
research (back to start)
```

**Import details:**

- `research` imports `research` at L20, L24, L32
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\research\__init__.py`
- `research` imports `research` at L20, L24, L32
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\research\__init__.py`

**Recommended fix:**

1. Extract common interface/protocol from `research` and `research`
2. Move shared types to a new `research.types` module
3. Use TYPE_CHECKING for type hints

```python
# Example fix using TYPE_CHECKING:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research import SomeClass

# Use 'SomeClass' as string in type hints
def func(obj: 'SomeClass') -> None:
    ...
```

---

## Module: `safety`

**Total modules:** 3

**Circular dependencies found:** 1

### Circular Import Chains

#### Cycle 1

```
safety
  |
  v
safety
  |
  v
safety (back to start)
```

**Import details:**

- `safety` imports `safety` at L22
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\safety\__init__.py`
- `safety` imports `safety` at L22
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\safety\__init__.py`

**Recommended fix:**

1. Extract common interface/protocol from `safety` and `safety`
2. Move shared types to a new `safety.types` module
3. Use TYPE_CHECKING for type hints

```python
# Example fix using TYPE_CHECKING:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safety import SomeClass

# Use 'SomeClass' as string in type hints
def func(obj: 'SomeClass') -> None:
    ...
```

---

## Module: `bounce`

**Total modules:** 5

**Circular dependencies found:** 1

### Circular Import Chains

#### Cycle 1

```
bounce
  |
  v
bounce
  |
  v
bounce (back to start)
```

**Import details:**

- `bounce` imports `bounce` at L11, L16, L22
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\bounce\__init__.py`
- `bounce` imports `bounce` at L11, L16, L22
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\bounce\__init__.py`

**Recommended fix:**

1. Extract common interface/protocol from `bounce` and `bounce`
2. Move shared types to a new `bounce.types` module
3. Use TYPE_CHECKING for type hints

```python
# Example fix using TYPE_CHECKING:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bounce import SomeClass

# Use 'SomeClass' as string in type hints
def func(obj: 'SomeClass') -> None:
    ...
```

---

## Module: `trade_logging`

**Total modules:** 2

**Circular dependencies found:** 1

### Circular Import Chains

#### Cycle 1

```
trade_logging
  |
  v
trade_logging
  |
  v
trade_logging (back to start)
```

**Import details:**

- `trade_logging` imports `trade_logging` at L39, L52
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\trade_logging\__init__.py`
- `trade_logging` imports `trade_logging` at L39, L52
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\trade_logging\__init__.py`

**Recommended fix:**

1. Extract common interface/protocol from `trade_logging` and `trade_logging`
2. Move shared types to a new `trade_logging.types` module
3. Use TYPE_CHECKING for type hints

```python
# Example fix using TYPE_CHECKING:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trade_logging import SomeClass

# Use 'SomeClass' as string in type hints
def func(obj: 'SomeClass') -> None:
    ...
```

---

## Module: `guardian`

**Total modules:** 10

**Circular dependencies found:** 1

### Circular Import Chains

#### Cycle 1

```
guardian
  |
  v
guardian
  |
  v
guardian (back to start)
```

**Import details:**

- `guardian` imports `guardian` at L59
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\guardian\__init__.py`
- `guardian` imports `guardian` at L59
  - File: `C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\guardian\__init__.py`

**Recommended fix:**

1. Extract common interface/protocol from `guardian` and `guardian`
2. Move shared types to a new `guardian.types` module
3. Use TYPE_CHECKING for type hints

```python
# Example fix using TYPE_CHECKING:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from guardian import SomeClass

# Use 'SomeClass' as string in type hints
def func(obj: 'SomeClass') -> None:
    ...
```

---

## Summary

**Total circular dependencies:** 8

### Priority Actions

1. Use `TYPE_CHECKING` guards for type-only imports (fastest fix)
2. Extract shared types to `.types` modules
3. Refactor to dependency injection where appropriate

---

**Note:** Self-referencing modules (e.g., `cognitive -> cognitive`) are often due to `__init__.py` re-exports and are generally acceptable for public API design.
