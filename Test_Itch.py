from sklearn.model_selection import train_test_split
from ITCH_PROSSERCE_ML import Itch_trade_modul
from ITCH import message, messages
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, SVR, NuSVC, NuSVR
from sklearn.kernel_approximation import KERNEL_PARAMS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def extract_features(trades_df, n_log=5):
    trades_df = trades_df.sort_index()
    df = trades_df.copy()
    for i in range(1, n_log + 1):
        df[f'lag_price_{i}'] = df['price'].shift(i)
        df[f'lag_shares_{i}'] = df['shares'].shift(i)
    
    df['target_price'] = df['price'].shift(-1)
    
    return df.dropna()

def train_model(df):
    X = df.drop(columns=['target_price', 'price', 'shares'])
    y = df['target_price']
    
    (x_train, x_test), (y_train, y_test) = train_test_split(X, y, shuffle=False,train_size=0.8)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(x_train, y_train)
    
    y_kyes = model.predict(x_test)
    print(f'MES:{mean_squared_error(y_test, y_kyes)}')
    return model


module = Itch_trade_modul(stock='AAPL')

messages = module.get_messages(date='2023-01-01', stock='AAPL')

trades = module.get_trader(messages)

features_df = extract_features(trades)

model = train_model(features_df)

