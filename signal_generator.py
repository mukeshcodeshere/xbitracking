# signal_generator.py
import backtrader as bt

class TopBottomStrategy(bt.Strategy):
    params = (
        ('top_window', 3),  # Number of periods to identify tops
        ('bottom_window', 3),  # Number of periods to identify bottoms
        ('signal_threshold', 0.02)  # Percentage threshold for significant moves
    )

    def __init__(self):
        # Store data series
        self.xbi_data = self.datas[0]  # First data series is XBI
        self.spy_data = self.datas[1]  # Second data series is SPY

        # Initialize signal tracking
        self.top_signals = []
        self.bottom_signals = []

    def next(self):
        # Detect tops with a more robust method
        if (self.xbi_data.close[0] > self.xbi_data.close[-1] and 
            self.xbi_data.close[-1] > self.xbi_data.close[-2]):
            # Calculate percentage change
            pct_change = (self.xbi_data.close[0] / self.xbi_data.close[-2] - 1) * 100
            
            # Check if the top is significant
            if pct_change > self.p.signal_threshold:
                self.top_signals.append({
                    'date': self.xbi_data.datetime.date(0),
                    'price': self.xbi_data.close[0],
                    'pct_change': pct_change
                })
                print(f'Top detected: {self.xbi_data.close[0]} on {self.xbi_data.datetime.date(0)}')

        # Detect bottoms with a more robust method
        if (self.xbi_data.close[0] < self.xbi_data.close[-1] and 
            self.xbi_data.close[-1] < self.xbi_data.close[-2]):
            # Calculate percentage change
            pct_change = (self.xbi_data.close[0] / self.xbi_data.close[-2] - 1) * 100
            
            # Check if the bottom is significant
            if pct_change < -self.p.signal_threshold:
                self.bottom_signals.append({
                    'date': self.xbi_data.datetime.date(0),
                    'price': self.xbi_data.close[0],
                    'pct_change': pct_change
                })
                print(f'Bottom detected: {self.xbi_data.close[0]} on {self.xbi_data.datetime.date(0)}')
