import encodings
import struct
import gzip
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urljoin
from clint.textui import progress
from datetime import datetime
from collections import namedtuple, Counter
from matplotlib.ticker import FuncFormatter
from datetime import timedelta
from time import time

# download data_itch from https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/
from ITCH_DOWNLOAD import may_be_download, URL_ITCH, FIELD_ICTH
from urllib.parse import urljoin

#field_name = may_be_download(urljoin(URL_ITCH, FIELD_ICTH))
#field_name = field_name.name.split('.')[0]

url = r'C:\Users\User\Documents\clever-trade-bot-ai-main44\clever-trade-bot-ai-main\P_project_with_python\Data_sources\GIT_DATASETS\tvagg.h5'

# git Path for all fields project
itch_path = Path('/clever-trade-bot-ai-main44/clever-trade-bot-ai-main/P_project_with_python/Data_sources/GIT_DATASETS')
itch_store = itch_path / 'tvagg.h5'
open_new_order = itch_path / 'message_types.h5'

event_codes = {'O': 'Start of Messages',
               'S': 'Start of System Hours',
               'Q': 'Start of Market Hours',
               'M': 'End of Market Hours',
               'E': 'End of System Hours',
               'C': 'End of Messages'}

message_type = [['A', 'message_type', 'char', 1, 'Type of the message (A = Add Order)'],
    ['A', 'stock_locate', 'uint16', 2, 'Unique stock identifier'],
    ['A', 'tracking_number', 'uint32', 4, 'Tracking number for message tracing'],
    ['A', 'timestamp', 'uint48', 6, 'Timestamp in nanoseconds since midnight'],
    ['A', 'order_reference_number', 'uint64', 8, 'Unique ID for this order'],
    ['A', 'buy_sell_indicator', 'char', 1, "'B' for Buy, 'S' for Sell"],
    ['A', 'shares', 'uint32', 4, 'Number of shares in the order'],
    ['A', 'stock', 'alpha', 8, 'Stock symbol (e.g. AAPL, MSFT)'],
    ['A', 'price', 'uint32', 4, 'Order price in 1/10,000 USD'],
    ['F', 'message_type', 'char', 1, 'Type of the message (F = Add w/ MPID)'],
    ['F', 'stock_locate', 'uint16', 2, 'Stock ID'],
    ['F', 'tracking_number', 'uint32', 4, 'Message tracking'],
    ['F', 'timestamp', 'uint48', 6, 'Time of order'],
    ['F', 'order_reference_number', 'uint64', 8, 'Reference ID'],
    ['F', 'buy_sell_indicator', 'char', 1, 'Buy or sell indicator'],
    ['F', 'shares', 'uint32', 4, 'Quantity'],
    ['F', 'stock', 'alpha', 8, 'Stock symbol'],
    ['F', 'price', 'uint32', 4, 'Price'],
    ['F', 'attribution', 'alpha', 4, 'Market participant ID'],
    ['P', 'message_type', 'char', 1, 'Type of the message (P = Trade)'],
    ['P', 'stock_locate', 'uint16', 2, 'Stock ID'],
    ['P', 'timestamp', 'uint48', 6, 'Time of trade'],
    ['P', 'order_reference_number', 'uint64', 8, 'ID of the order being executed'],
    ['P', 'match_number', 'uint64', 8, 'Unique match ID'],
    ['P', 'shares', 'uint32', 4, 'Executed share quantity'],
    ['P', 'price', 'uint32', 4, 'Execution price'],
    ['P', 'attribution', 'alpha', 4, 'Participant ID'],
    ['E', 'message_type', 'char', 1, 'Execution Message'],
    ['E', 'stock_locate', 'uint16', 2, 'Stock ID'],
    ['E', 'timestamp', 'uint48', 6, 'Execution time'],
    ['E', 'order_reference_number', 'uint64', 8, 'Order ID'],
    ['E', 'executed_shares', 'uint32', 4, 'Number of shares executed'],
    ['E', 'match_number', 'uint64', 8, 'Trade match number'],
    ['E', 'printable', 'char', 1, 'Indicates if trade should be printed'],
    ['E', 'execution_price', 'uint32', 4, 'Price of execution'],
    ['D', 'message_type', 'char', 1, 'Type of the message (D = Delete Order)'],
    ['D', 'stock_locate', 'uint16', 2, 'Stock ID'],
    ['D', 'tracking_number', 'uint32', 4, 'Tracking number'],
    ['D', 'timestamp', 'uint48', 6, 'Timestamp'],
    ['D', 'order_reference_number', 'uint64', 8, 'Order ID to delete'],
]

encoding = {'primary_market_maker': {'Y': 1, 'N': 0},
            'printable'           : {'Y': 1, 'N': 0},
            'buy_sell_indicator'  : {'B': 1, 'S': -1},
            'cross_type'          : {'O': 0, 'C': 1, 'H': 2},
            'imbalance_direction' : {'B': 0, 'S': 1, 'N': 0, 'O': -1}
            }

formats = {
    ('integer', 2): 'H',
    ('integer', 4): 'I',
    ('integer', 6): '6s',
    ('integer', 8): 'Q',
    ('alpha', 1)  : 's',
    ('alpha', 2)  : '2s',
    ('alpha', 4)  : '4s',
    ('alpha', 8)  : '8s',
    ('price_4', 4): 'I',
    ('price_8', 8): 'Q',
}

# excel path
excel_path = r'clever-trade-bot-ai-main\P_project_with_python\Data_sources\GIT_DATASETS\message_types.xlsx'

# make message_type excel field
os.makedirs(os.path.basename(excel_path), exist_ok=True)
specs = pd.DataFrame(message_type, columns=['message_type', 'name', 'value', 'length', 'notes'])
specs.to_excel(excel_path, sheet_name='message_types',index=False)

# 
def clean_message_type(df):
    df.columns = [c.lower().strip() for c in df.columns]
    df.value = df.value.str.strip()
    df.name = (df.name
               .str.strip() # remove whitespace
               .str.lower()
               .str.replace(' ', '_')
               .str.replace('-', '_')
               .str.replace('/', '_'))
    df.notes = df.notes.str.strip()
    df['message_type'] = df.loc[df.name == 'message_type', 'value']
    return df
print()
specs = clean_message_type(specs)

specs['formats'] = specs[['value', 'length']].apply(tuple, axis=1).map(formats)

alpha_fields = specs[specs.value == 'alpha'].set_index('name')
alpha_msg = alpha_fields.groupby('message_type')
alpha_formats = {k: c.to_dict() for k,c in alpha_msg.formats} 
alpha_lenghts = {k: c.add(5).to_dict() for k,c in alpha_msg.length}

fstring = {
    "S": ">cHH6scc",
    "R": ">cHH6s8sccIcIcc",
    "H": ">cHH6s8sc10s",
    "A": ">cHH6sQcI8sI",
    "F": ">cHH6sQcI8sI4s",
    "D": ">cHH6sQ",
    "E": ">cHH6sQI8scI",
    "X": ">cHH6sQI",
    "P": ">cHH6sQcI8sI"
}

message_fields = {
    'S': namedtuple('SystemEvent', ('message_type', 'stock_locate', 'tracking_number', 'timestamp', 'event_code_1', 'event_code_2')),
    'R': namedtuple('StockDirectory', ('message_type', 'stock_locate', 'tracking_number', 'timestamp', 'stock', 'market_category', 'financial_status_indicator', 'round_lot_size', 'round_lots_only', 'issue_classification', 'issue_sub_type', 'authenticity', 'short_sale_threshold')),
    'H': namedtuple('StockTradingAction', ('message_type', 'stock_locate', 'tracking_number', 'timestamp', 'stock', 'trading_state', 'reason')),
    'A': namedtuple('AddOrder', ('message_type', 'stock_locate', 'tracking_number', 'timestamp', 'order_ref', 'buy_sell', 'shares', 'stock', 'price')),
    'F': namedtuple('AddOrderMPID', ('message_type', 'stock_locate', 'tracking_number', 'timestamp', 'order_ref', 'buy_sell', 'shares', 'stock', 'price', 'attribution')),
    'D': namedtuple('OrderDelete', ('message_type', 'stock_locate', 'tracking_number', 'timestamp', 'order_ref')),
    'E': namedtuple('OrderExecutedPrice', ('message_type', 'stock_locate', 'tracking_number', 'timestamp', 'order_ref', 'executed_shares', 'match_number', 'printable', 'execution_price')),
    'X': namedtuple('OrderCancel', ('message_type', 'stock_locate', 'tracking_number', 'timestamp', 'order_ref', 'cancelled_shares')),
    'P': namedtuple('Trade', ('message_type', 'stock_locate', 'tracking_number', 'timestamp', 'order_ref', 'buy_sell', 'shares', 'stock', 'price', 'match_number')),
}

# Generate message classes as named tuples and format strings
for t, massege in specs.groupby('message_type'):
    message_field = massege['name'].tolist()
    message_field = list(dict.fromkeys(message_field))
    message_fields[t] = namedtuple(typename=t, field_names=message_field)
    fstring[t] = '>' + ''.join((str(f) for f in massege['formats'].tolist()))

def get_alpha(mtype, data):
    """Process byte strings of type alpha"""
    for col in alpha_formats.get(mtype).keys():
        if col == 'stocks' and mtype == "R":
            data = data.drop(col, axis=3)
            continue
        data.loc[:, col] = (data.loc[:, col].str.decode('utf-8').str.strip())
        if encoding.get(col):
            data.loc[:, col] = data.loc[:, data].map(encoding.get(col))
    return data

def store_message(m):
    with pd.HDFStore(itch_store) as store:
        for mtype, data in m.items:
            # make datasets as DataFrome use pandas
            data = pd.DataFrame(data)
            
            # timestamp
            data.timestamp = data.timestamp.apply(int.from_bytes, byteorder='big')
            data.timestamp = data.to_timedelta(data.timestamp)   

            if mtype in alpha_formats.keys():
                data = get_alpha(mtype, data)
            s = alpha_lenghts.get(mtype)
            if s:
               s = {c: s.get(c) for c in data.columns}
            dc = ['stock_locate']
            if m == '/R':
               dc.append('stock')
            store.append(mtype, data, format='t', data_columns=dc, min_itemsize=s)    

messages = {}
message_type_counter = Counter()
message_count = 0
start = time()

with open(url, 'rb') as data:
    while True:
        # determine message size in bytes
        size_bytes = data.read(2)
        if not size_bytes:
                break
        message_size = int.from_bytes(size_bytes, byteorder='big')

        message_type_byte = data.read(1)
        if not message_type_byte:
                break
        message_type = message_type_byte.decode('ascii')

        if message_type not in fstring:
                data.read(message_size - 1)
                continue

        record = data.read(message_size - 1)
        fmt = fstring[message_type]
        expected_size = struct.calcsize(fmt)

        if len(record) != expected_size:
                print(f"Skipping {message_type}: expected {expected_size}, got {len(record)}")
                continue

        fields = struct.unpack(fmt, record)
        field_names = message_fields[message_type]._fields
        message = dict(zip(field_names, fields))
        
                    
        if message_type not in messages:
              messages[message_type] = []

        messages[message_type].append(message)
        message_type_counter.update([message_type])
        message_count += 1

        # deal with system events
        if message_type == 'S':
            timestamp = int.from_bytes(message.timestamp, byteorder='big')
            print('\n', event_codes.get(message.event_code.decode('ascii'), 'Error'))
            print('\t{0}\t{1:,.0f}'.format(timedelta(seconds=timestamp * 1e-9),
                                         message_count))
            if message.event_code.decode('ascii') == 'C':
                store_message(messages)
                break

        message_count += 1
        if message_count % 2.5e7 == 0:
            timestamp = int.from_bytes(message.timestamp, byteorder='big')
            print('\t{0}\t{1:,.0f}\t{2}'.format(timedelta(seconds=timestamp * 1e-9),
                                                message_count,
                                                timedelta(seconds=time() - start)))
            store_message(messages)
            messages = {}
        
print(timedelta(seconds=time() - start))

#with pd.HDFStore(url) as store:
#    stocks = store['/Q'].loc[:, ['stock_locate', 'shares']]
#    trades = store['/P'].append(store['/Q'].rename(columns={'cross_price': 'price'}), sort=False).merge(stocks)
#trades['value'] = trades.shares.mul(trades.price)
#trades['value_share'] = trades.value.div(trades.value.sum())
#trade_summary = trades.groupby('stock').value_share.sum().sort_values(ascending=False)
#trade_summary.iloc[:50].plot.bar(figsize=(14, 6), color='darkblue', title='Share of Traded Value')
#plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

if __name__ == "__main__":
    print("function of may_be_download()...")

