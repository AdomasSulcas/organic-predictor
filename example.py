"""Example usage of Organic Traffic Prophet."""

from pathlib import Path
import warnings
import pandas as pd

# Suppress the plotly warning from Prophet
warnings.filterwarnings('ignore', message='Importing plotly failed')

from src import (
    Config, 
    TrafficDataLoader, 
    TrafficPreprocessor,
    TrafficProphetModel,
    TrafficAnalyzer,
    TrafficVisualizer
)


def create_example_data(filename: str = 'example_data.csv') -> None:
    """Create example traffic data for demonstration."""
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(500)]
    
    base_traffic = 500
    trend = np.linspace(0, 300, 500)
    weekly_seasonality = 50 * np.sin(2 * np.pi * np.arange(500) / 7)
    yearly_seasonality = 100 * np.sin(2 * np.pi * np.arange(500) / 365.25)
    noise = np.random.normal(0, 30, 500)
    
    clicks = base_traffic + trend + weekly_seasonality + yearly_seasonality + noise
    clicks = np.maximum(clicks, 100).astype(int)
    
    impressions = clicks * np.random.uniform(40, 60, 500)
    ctr = (clicks / impressions * 100)
    position = np.random.uniform(10, 20, 500)
    
    df = pd.DataFrame({
        'Date': dates,
        'Clicks': clicks,
        'Impressions': impressions.astype(int),
        'CTR': [f'{x:.2f}%' for x in ctr],
        'Position': np.round(position, 2)
    })
    
    df.to_csv(filename, index=False)
    print(f"Example data created: {filename}")


def run_example_analysis() -> None:
    """Run a complete example analysis."""
    data_file = 'example_data.csv'
    
    if not Path(data_file).exists():
        print("Creating example data...")
        create_example_data(data_file)
    
    print("\n1. Loading data...")
    loader = TrafficDataLoader(data_file)
    df = loader.load()
    
    print("\n2. Preprocessing data...")
    preprocessor = TrafficPreprocessor()
    df_processed = preprocessor.process(df)
    
    print("\n3. Analyzing traffic patterns...")
    analyzer = TrafficAnalyzer()
    analysis = analyzer.analyze(df_processed)
    
    print("\nWeekly Pattern:")
    print(analysis['weekly_pattern'][['day', 'relative_strength']])
    
    print(f"\nGrowth Metrics:")
    print(f"- Total growth: {analysis['growth_metrics']['total_growth_pct']:.1f}%")
    print(f"- Monthly average: {analysis['growth_metrics']['avg_monthly_growth_pct']:.1f}%")
    
    print("\n4. Training Prophet model...")
    config = Config()
    model = TrafficProphetModel(config)
    model.fit(df_processed)
    
    print("\n5. Generating 90-day forecast...")
    forecast = model.predict(periods=90)
    future_predictions = model.get_future_predictions(forecast, df_processed)
    
    print("\nNext 7 days forecast:")
    print(future_predictions.head(7))
    
    print("\n6. Creating visualizations...")
    output_dir = Path('example_output')
    visualizer = TrafficVisualizer(output_dir)
    visualizer.plot_forecast(df_processed, forecast, model.model)
    visualizer.plot_components(model.model, forecast)
    visualizer.plot_analysis(df_processed, analysis)
    
    print(f"\n✓ Analysis complete! Check the '{output_dir}' directory for results.")


if __name__ == "__main__":
    run_example_analysis()