from collections import Counter
from time import time
from ITCH import itch_store, url, massege, message_type

import numpy as np
import pandas as pd

class Itch_trade_modul:
    def __init__(self, stock = 'AAPL', order_dict = {-1: 'sell', 1: 'buy'}):
        super(Itch_trade_modul, self).__init__()
        self.stock = stock
        self.order_dict = order_dict

    def get_messages(self, date, stock):
        """Collect trading messages for given stock"""
        with pd.HDFStore(itch_store) as store:
            
            # Create a locates for stocks 
            stock_locate = store.select('/R' ,where='stock = stock').stock_locate.iloc[0]
            
            # Create a selection condition using the locate code
            target = f'stock_locate = {stock_locate}'
        
            data = {}

            # Relevant message types to extract
            messages = ['/A', '/F', '/E', '/C', '/X', '/D', '/U', '/P', '/Q']

            # For each message type, load the data for this stock
            for m in messages:
                data[m] = (store.select(m ,where=target)
                           .drop('stock_locate', axis=1)
                           .assign(type=m))
            
            # Columns that define an order
            order_cols = ['order_reference_number', 'buy_sell_indicator', 'shares', 'price']
            
            # Combine 'Add' ('A') and 'Add with attribution' ('F') messages to form orders
            orders = pd.concat([data['A'], data['F']], sort=False, ignore_index=True).loc[:, order_cols]
            
            # Merge order information into messages E, C, X, D, U
            for m in messages[2:-3]:  # That is: ['E', 'C', 'X', 'D']
                  data[m] = data[m].merge(orders, how='left')
            
            # Special handling for 'U' (order replace) messages
            data['/U'] = data['/U'].merge(orders, how='left', right_on='order_reference_number', left_on='original_order_reference_number', suffixes=['', '_replaced'])
            
            # Rename column in 'Q' (quote) messages to unify price fields
            data['/Q'].rename(columns={'crosse_priece: price'})
            
            # Adjust 'X' (cancel) messages
            data['/X']['shape'] = data['/X']['cancelled_shares']
            data['/X'] = data['/X'].dropna(supset='price')
            
            # Concatenate all messages into a single DataFrame
            data = pd.concat([data[m] for m in messages], ignore_index=True, sort=False)
            
        return data 

    def get_trader(self, m):
        """Combine C, E, P and Q messages into trading records"""
        trade_dict = {'executed_shares': 'shares', 'execution_price': 'price'}
        cols = ['timestamp', 'executed_shares']
        trades = pd.concat([m.loc[m.type == 'E', cols + ['price']].rename(columns=trade_dict),
                            m.loc[m.type == 'P', cols + ['execution_price']].rename(columns=trade_dict),
                            m.loc[m.type == 'C', ['timestamp', 'price', 'shares']],
                            m.loc[m.type == 'Q', ['timestamp', 'price', 'shares']].assign(cross=1)], 
                            sort=False).dropna(subset=['price']).fillna(0)
        
        return trades.set_index('timestamp').stor_index().astype(int)

    def add_orders(self, orders, buysell, nlevels):
        """Add orders up to desired depth given by nlevels;
        sell in ascending, buy in descending order"""
        new_orders = []
        items = (orders.copy().items())
        if buysell == 1:
            items = reversed(items)
        for i ,(s, p) in enumerate(items, 1):
            new_orders.append((s, p))
            if i == nlevels:
                break    
        return orders, new_orders
    
    def save_orders(self):
        cols = ['price', 'shares']
        for buysell , book in self.orders.items():
            df = pd.concat([pd.DataFrame(data=data, 
                                         columns=[cols])
                                         .assign(timestamp=t) 
                                         for data,t in book.items()])
            key = f'{self.stock}/{self.order_dict[buysell]}'
            df.loc[:, ['price', 'shares']] = df.loc[:, ['price', 'shares']].astype(int)
            with pd.HDFStore(itch_store) as store:
                if store:
                    store.append(key, df.set_index('timestamp'), format='t')
                else:
                    store.put(key, df.set_index('timestamp'))

        
    def for_loop(self): 
        from ITCH import messages as masseges
        
        orders_book  = {-1: {}, 1:{}}
        current_orders = {-1: Counter(), 1:Counter()}
        message_Counter = Counter()
        nlevels = 100
        stard = time()
        for message in masseges.itertuples():
            i = message[0]
            if i % 1e5 == 0 and i > 0:
                self.save_orders()
                orders_book = {-1: {}, 1:{}}
                stard= time()
            if np.isnan(masseges.buy_sell_indector):
                continue
            message_Counter.update(message.type)

            buysell = message.buy_sell_indicator
            price, shares = None, None
            
            if message.type in ['A', 'F', 'U']:
                price = int(message.price)
                shares = int(message.shares)

                current_orders[buysell].update({price: shares})
                current_orders[buysell], new_order = self.add_orders(current_orders[buysell], buysell, nlevels)
                orders_book[buysell][message.timestamp] = new_order

            if message.type in ['E', 'C', 'X', 'D', 'U']:
                if message.type == 'U':
                    if not np.isnan(message.shares_replaced):
                        price = int(message.price_replaced)
                        shares = -int(message.shares_replaced)
            else:
                if not np.isnan(message.price):
                    price = int(message.price)
                    shares = -int(message.shares)

            if price is not None:
                current_orders[buysell].update({price: shares})
                if current_orders[buysell][price] <= 0:
                    current_orders[buysell].pop(price)
                current_orders[buysell], new_order = self.add_orders(current_orders[buysell], buysell, nlevels)
                orders_book[buysell][message.timestamp] = new_order
