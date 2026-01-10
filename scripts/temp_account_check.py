#!/usr/bin/env python3
"""Temporary script to check Alpaca account."""
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import OrderStatus, QueryOrderStatus

    api_key = os.getenv('ALPACA_API_KEY_ID')
    secret = os.getenv('ALPACA_API_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL', '')

    print(f'Base URL: {base_url}')
    print(f'Is Paper: {"paper" in base_url.lower()}')
    print()

    client = TradingClient(api_key, secret, paper=('paper' in base_url.lower()))
    account = client.get_account()

    print('=== ALPACA ACCOUNT (REAL DATA) ===')
    print(f'Equity: ${float(account.equity):,.2f}')
    print(f'Cash: ${float(account.cash):,.2f}')
    print(f'Buying Power: ${float(account.buying_power):,.2f}')
    print(f'Portfolio Value: ${float(account.portfolio_value):,.2f}')
    print(f'Long Market Value: ${float(account.long_market_value):,.2f}')
    print(f'Last Equity: ${float(account.last_equity):,.2f}')
    print()

    # Get closed trades to see actual history
    request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=50)
    orders = client.get_orders(request)

    filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]

    print(f'=== RECENT FILLED ORDERS ({len(filled_orders)}) ===')
    for o in filled_orders[:20]:
        date_str = o.filled_at.strftime("%Y-%m-%d") if o.filled_at else "N/A"
        price = float(o.filled_avg_price) if o.filled_avg_price else 0
        print(f'{date_str} | {o.symbol:5} | {o.side.value:4} | {o.filled_qty:>4} @ ${price:>8.2f} | {o.order_type.value}')

except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc()
