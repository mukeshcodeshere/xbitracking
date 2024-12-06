# backtest.py
import backtrader as bt

def prepare_backtrader_data(df_xbi, df_spy):
    # Convert DataFrame to Backtrader data feed
    xbi_data = bt.feeds.PandasData(
        dataname=df_xbi,
        datetime='date',
        open='open',
        high='high',
        low='low',
        close='adjclose',
        volume='volume'
    )
    
    spy_data = bt.feeds.PandasData(
        dataname=df_spy,
        datetime='date',
        open='open',
        high='high',
        low='low',
        close='adjclose',
        volume='volume'
    )
    
    return xbi_data, spy_data

def run_top_bottom_strategy(df_ticker_daily, strategy_class):
    # Filter XBI and SPY data
    df_xbi = df_ticker_daily[df_ticker_daily.symbol == "XBI"].sort_values('date')
    df_spy = df_ticker_daily[df_ticker_daily.symbol == "SPY"].sort_values('date')

    # Prepare Cerebro engine
    cerebro = bt.Cerebro()

    # Prepare data feeds
    xbi_data, spy_data = prepare_backtrader_data(df_xbi, df_spy)

    # Add data to Cerebro
    cerebro.adddata(xbi_data)
    cerebro.adddata(spy_data)

    # Add strategy
    cerebro.addstrategy(strategy_class)

    # Run the strategy
    cerebro.run()
