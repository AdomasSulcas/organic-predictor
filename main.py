"""
Organic Traffic Prophet - Main Entry Point

A Prophet-based forecasting tool for organic traffic prediction.
"""

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd


from src.data_loader import TrafficDataLoader
from src.preprocessor import TrafficPreprocessor
from src.model import TrafficProphetModel
from src.analyzer import TrafficAnalyzer
from src.visualizer import TrafficVisualizer
from src.config import Config


def main(data_path: str, output_dir: str, forecast_days: int = 90) -> None:
    """
    Run the complete traffic prediction pipeline.
    
    Args:
        data_path: Path to the CSV file containing traffic data
        output_dir: Directory to save outputs
        forecast_days: Number of days to forecast
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    loader = TrafficDataLoader(data_path)
    df = loader.load()
    
    print("Preprocessing data...")
    preprocessor = TrafficPreprocessor()
    df_processed = preprocessor.process(df)
    
    print("Analyzing traffic patterns...")
    analyzer = TrafficAnalyzer()
    analysis_results = analyzer.analyze(df_processed)
    
    print("Training Prophet model...")
    model = TrafficProphetModel(config=Config())
    model.fit(df_processed)
    
    print(f"Generating {forecast_days}-day forecast...")
    forecast = model.predict(periods=forecast_days)
    
    print("Creating visualizations...")
    visualizer = TrafficVisualizer(output_dir=output_path, config=Config())
    visualizer.plot_forecast(df_processed, forecast, model.model)
    visualizer.plot_components(model.model, forecast)
    visualizer.plot_analysis(df_processed, analysis_results)
    
    if visualizer.config.create_dashboard:
        print("Creating interactive dashboard...")
        visualizer.create_interactive_dashboard(df_processed, forecast, analysis_results)
    
    print("Saving predictions...")
    predictions_df = model.get_future_predictions(forecast, df_processed)
    predictions_df.to_csv(output_path / 'predictions.csv', index=False)
    
    print("\nAnalysis Summary:")
    print(f"- Historical data: {len(df_processed)} days")
    print(f"- Date range: {df_processed['ds'].min().date()} to {df_processed['ds'].max().date()}")
    print(f"- Average daily traffic: {df_processed['y'].mean():.0f}")
    print(f"- Predicted average (next {forecast_days} days): {predictions_df['predicted'].mean():.0f}")
    
    print(f"\nAnalysis complete! Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organic Traffic Prophet Forecasting')
    parser.add_argument('data_path', type=str, help='Path to traffic data CSV file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--days', type=int, default=90, help='Days to forecast')
    
    args = parser.parse_args()
    main(args.data_path, args.output, args.days)