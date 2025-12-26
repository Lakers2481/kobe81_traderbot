# /idempotency

View and manage the idempotency store (duplicate order prevention).

## Usage
```
/idempotency [--show|--stats|--clear-old DAYS]
```

## What it does
1. Shows recent idempotency keys
2. Displays store statistics
3. Clears old entries (with safety checks)
4. Validates store integrity

## Commands
```bash
# Show idempotency store stats
python -c "
import sqlite3
from pathlib import Path
from datetime import datetime

db_path = Path('state/idempotency.sqlite')
if not db_path.exists():
    print('No idempotency store found (first run?)')
    exit()

con = sqlite3.connect(db_path)
cur = con.cursor()

# Count entries
cur.execute('SELECT COUNT(*) FROM idempotency')
total = cur.fetchone()[0]

# Get date range
cur.execute('SELECT MIN(created_at), MAX(created_at) FROM idempotency')
row = cur.fetchone()

print('=== IDEMPOTENCY STORE ===')
print(f'Location: {db_path}')
print(f'Total entries: {total}')
if row[0]:
    print(f'Oldest: {row[0][:19]}')
    print(f'Newest: {row[1][:19]}')

# Size
size = db_path.stat().st_size
print(f'Database size: {size / 1024:.1f} KB')
con.close()
"

# Show recent entries
python -c "
import sqlite3
from pathlib import Path

db_path = Path('state/idempotency.sqlite')
if not db_path.exists():
    print('No idempotency store found')
    exit()

con = sqlite3.connect(db_path)
cur = con.cursor()

print('=== RECENT IDEMPOTENCY KEYS ===')
cur.execute('SELECT decision_id, idempotency_key, created_at FROM idempotency ORDER BY created_at DESC LIMIT 20')
for row in cur.fetchall():
    print(f'{row[2][:19]} | {row[0][:30]}...')
con.close()
"

# Check for duplicates (should be none)
python -c "
import sqlite3
from pathlib import Path

db_path = Path('state/idempotency.sqlite')
if not db_path.exists():
    print('No idempotency store found')
    exit()

con = sqlite3.connect(db_path)
cur = con.cursor()

cur.execute('''
    SELECT decision_id, COUNT(*) as cnt
    FROM idempotency
    GROUP BY decision_id
    HAVING cnt > 1
''')
dups = cur.fetchall()

if dups:
    print('⚠️ DUPLICATE ENTRIES FOUND:')
    for row in dups:
        print(f'  {row[0]}: {row[1]} copies')
else:
    print('✅ No duplicates found')
con.close()
"

# Clear entries older than N days (CAREFUL!)
python -c "
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

DAYS = 30  # Adjust as needed

db_path = Path('state/idempotency.sqlite')
if not db_path.exists():
    print('No idempotency store found')
    exit()

cutoff = (datetime.utcnow() - timedelta(days=DAYS)).isoformat()

con = sqlite3.connect(db_path)
cur = con.cursor()

# Count first
cur.execute('SELECT COUNT(*) FROM idempotency WHERE created_at < ?', (cutoff,))
count = cur.fetchone()[0]

print(f'Entries older than {DAYS} days: {count}')
print('To delete, uncomment the DELETE line in the script')
# cur.execute('DELETE FROM idempotency WHERE created_at < ?', (cutoff,))
# con.commit()
# print(f'Deleted {count} entries')

con.close()
"
```

## How Idempotency Works
```
Signal generated → decision_id created
                     ↓
              IdempotencyStore.exists()?
                   /     \
                 YES      NO
                  ↓        ↓
              SKIP     place_ioc_limit()
                              ↓
                     IdempotencyStore.put()
```

## Store Schema
| Column | Type | Description |
|--------|------|-------------|
| decision_id | TEXT PK | Unique decision identifier |
| idempotency_key | TEXT | Key sent to broker |
| created_at | TEXT | ISO timestamp |

## Maintenance
- **Daily**: No action needed
- **Weekly**: Review stats
- **Monthly**: Clear entries >30 days old
- **Never**: Clear during active trading
