# Organic Traffic Prophet

A clean, production-ready Prophet-based forecasting tool for organic traffic prediction. Uses data from Google Search Console (Dates.csv) to predict future month performance.

## Features

- **Prophet-based forecasting** with automatic seasonality detection
- **Comprehensive traffic analysis** including weekly and monthly patterns
- **Anomaly detection** to identify unusual traffic spikes or drops
- **Beautiful visualizations** for insights and presentations
- **Modular architecture** for easy customization and extension

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/organic-traffic-prophet.git
cd organic-traffic-prophet

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Basic usage
python main.py your_traffic_data.csv

# With custom options
python main.py your_traffic_data.csv --output results --days 180
```

## Data Format

Your CSV file should contain the following columns:
- `Date`: Date in YYYY-MM-DD format
- `Clicks`: Daily click count
- `Impressions`: Daily impressions
- `CTR`: Click-through rate (e.g., "2.5%")
- `Position`: Average position

Example:
```csv
Date,Clicks,Impressions,CTR,Position
2024-01-01,523,25000,2.09%,15.3
2024-01-02,498,24500,2.03%,15.8
```

## Output

The tool generates:
- `predictions.csv`: Future traffic predictions with confidence intervals
- `forecast.png`: Visual forecast with historical data
- `components.png`: Breakdown of trend, weekly, and seasonal patterns
- `analysis.png`: Comprehensive traffic analysis visualizations

## Project Structure

```
organic-traffic-prophet/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md              # This file
└── src/
    ├── __init__.py
    ├── config.py          # Configuration settings
    ├── data_loader.py     # Data loading utilities
    ├── preprocessor.py    # Data preprocessing
    ├── model.py           # Prophet model implementation
    ├── analyzer.py        # Traffic analysis
    └── visualizer.py      # Visualization utilities
```

## Configuration

Modify `src/config.py` to customize model parameters:

```python
@dataclass
class Config:
    growth: str = 'linear'  # or 'logistic' for capped growth
    changepoint_prior_scale: float = 0.05  # Flexibility of trend
    seasonality_mode: str = 'multiplicative'  # or 'additive'
    # ... more options
```

## Advanced Usage

### Programmatic API

```python
from src.data_loader import TrafficDataLoader
from src.preprocessor import TrafficPreprocessor
from src.model import TrafficProphetModel
from src.config import Config

# Load and preprocess data
loader = TrafficDataLoader('your_data.csv')
df = loader.load()

preprocessor = TrafficPreprocessor()
df_processed = preprocessor.process(df)

# Train model
model = TrafficProphetModel(Config())
model.fit(df_processed)

# Generate predictions
forecast = model.predict(periods=90)
```

### Cross-Validation

```python
# Perform time series cross-validation
cv_results = model.cross_validate(
    initial_days=365,
    period_days=30,
    horizon_days=30
)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Prophet](https://facebook.github.io/prophet/) by Meta
- Inspired by real-world SEO analytics needs