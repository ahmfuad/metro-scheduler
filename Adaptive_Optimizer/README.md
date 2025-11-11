# Metro Adaptive Optimization System

A comprehensive metro headway optimization system that uses genetic algorithms for initial optimization, then monitors real-time passenger loads to trigger adaptive re-optimization when needed.

## System Overview

The system implements a multi-stage optimization approach:

1. **Initial Optimization**: Uses genetic algorithm with historical data to generate optimal headway schedule for the entire day
2. **Real-Time Monitoring**: Continuously monitors passenger loads at all stations and calculates Z-scores against historical patterns
3. **Adaptive Re-optimization**: Triggers genetic algorithm re-optimization when Z-score thresholds are exceeded
4. **Demand Adaptation**: Scales demand up/down for the next 30 minutes based on current Z-scores, while keeping the rest of the day based on historical data
5. **Comprehensive Reporting**: Provides detailed visualization and analysis of system performance

## Architecture

The system consists of several modular components:

### Core Modules

- **`data_processor.py`**: Handles historical data loading, processing, and Z-score calculations
- **`real_time_monitor.py`**: Simulates real-time passenger monitoring and threshold detection
- **`adaptive_ga.py`**: Implements adaptive genetic algorithm with demand scaling
- **`visualization.py`**: Comprehensive visualization and reporting system
- **`main.py`**: Main orchestration application that integrates all components

### Dependencies

- **Existing GA Infrastructure**: Uses the genetic algorithm and simulator from `../Simple Optimizer/`
- **Data**: Processes CSV data from `../data/simulation_log.csv`

## Features

### Data Processing
- Loads and processes historical trip data from CSV
- Calculates hourly arrival rates per station
- Builds origin-destination (OD) matrices
- Computes baseline demand patterns
- Implements Z-score calculations for deviation detection

### Real-Time Monitoring
- Simulates real-time passenger data with configurable noise
- Calculates station-wise and aggregated Z-scores
- Tracks threshold exceedances
- Provides demand scaling factors for adaptation

### Adaptive Optimization
- Extends base genetic algorithm with adaptive capabilities
- Implements 30-minute demand scaling windows
- Optimizes headway schedules based on real-time conditions
- Tracks optimization performance and improvements

### Visualization and Reporting
- Real-time monitoring dashboards
- Optimization results analysis
- Performance comparison plots
- Animated monitoring visualizations
- Comprehensive PDF reports

## Quick Start

### 1. Test the System
Run a quick validation test with reduced parameters:

```bash
cd Adaptive_Optimizer
python test_system.py
```

This will:
- Use a small data sample (1,000 trips)
- Run a 2-hour simulation
- Test all system components
- Generate test results in `test_results/`

### 2. Run Full Simulation
Run the complete optimization system:

```bash
python main.py
```

Default parameters:
- Sample size: 10,000 trips
- Simulation time: 8 hours
- Z-score threshold: 2.0
- Monitoring interval: 5 minutes

### 3. Custom Configuration
Run with custom parameters:

```bash
python main.py --sample-size 15000 --simulation-hours 12 --z-threshold 1.8 --output-dir my_results
```

## Configuration Options

### Command Line Arguments

- `--sample-size`: Number of data rows to process (default: 10000)
- `--simulation-hours`: Simulation duration in hours (default: 8.0)
- `--z-threshold`: Z-score threshold for triggering adaptation (default: 2.0)
- `--output-dir`: Output directory for results (default: optimization_results)
- `--no-viz`: Disable visualization generation

### Configuration Class

The `MetroOptimizationConfig` class provides detailed configuration options:

```python
config = MetroOptimizationConfig()
config.csv_file_path = "../data/simulation_log.csv"
config.sample_size = 10000
config.z_score_threshold = 2.0
config.monitoring_interval_minutes = 5.0
config.total_simulation_time_hours = 8.0
config.adaptation_duration_minutes = 30.0
```

## System Parameters

### Genetic Algorithm Parameters

**Initial Optimization** (comprehensive):
- Population size: 25
- Generations: 30
- Mutation rate: 0.15
- Crossover rate: 0.6

**Adaptive Optimization** (faster):
- Population size: 20
- Generations: 20
- Mutation rate: 0.20
- Crossover rate: 0.6

### Fitness Function Weights
- α (waiting time weight): 3.5
- β (leftover passengers weight): 0.25
- γ (number of trains weight): 50.0

### Station Configuration
- 16 stations: S01UN through S16MJ
- Train capacity: 1,200 passengers
- Headway range: 3-10 minutes

## Output

The system generates comprehensive output in the specified directory:

### Files Generated
- `final_metrics.json`: Complete performance metrics
- `monitoring_history.json`: Real-time monitoring data
- `adaptation_history.json`: Adaptation events and results
- `metro_optimization.log`: Detailed system logs

### Visualizations (in `plots/` subdirectory)
- `monitoring_dashboard.png`: Real-time monitoring overview
- `optimization_results.png`: GA performance analysis
- `comparison_analysis.png`: Static vs adaptive comparison
- `fitness_evolution.png`: Fitness improvement over time
- `animated_monitoring.gif`: Animated monitoring visualization
- `comprehensive_report.png`: Summary dashboard

## Performance Metrics

The system tracks and reports:

### Optimization Metrics
- Initial fitness score
- Final adaptive fitness score
- Overall improvement percentage
- Average waiting time
- Passengers served vs. left behind

### Monitoring Metrics
- Total monitoring points
- Threshold exceedances
- Exceedance rate
- Average and maximum Z-scores

### System Performance
- Total adaptation events
- Average adaptation time
- Optimization efficiency
- System responsiveness

## Example Output

```
METRO ADAPTIVE OPTIMIZATION SYSTEM - FINAL REPORT
================================================================================

SIMULATION PARAMETERS:
  Total simulation time: 8.0 hours
  Monitoring interval: 5.0 minutes
  Z-score threshold: 2.0
  Data sample size: 10000 trips

OPTIMIZATION RESULTS:
  Initial (Static) Fitness: 1250.75
  Final (Adaptive) Fitness: 1180.42
  Overall Improvement: 5.6%

PASSENGER SERVICE:
  Initial Passengers Served: 8543
  Initial Passengers Left: 267
  Initial Avg Waiting Time: 3.24 minutes

ADAPTIVE OPTIMIZATION:
  Total Adaptations: 7
  Average Improvement per Adaptation: 8.2%
  Average Adaptation Time: 12.5 seconds

MONITORING PERFORMANCE:
  Total Monitoring Points: 96
  Threshold Exceedances: 12
  Exceedance Rate: 12.5%
  Average Z-score: 1.45
  Maximum Z-score: 3.78

FINAL HEADWAY SCHEDULE:
  [6, 4, 5, 7, 4, 6, 5, 8]
```

## Data Requirements

The system expects CSV data with the following columns:
- `Entry`: Entry timestamp
- `Exit`: Exit timestamp  
- `Origin`: Origin station ID
- `Dest`: Destination station ID
- `ID`: Trip/passenger ID

Station IDs should follow the format: S01UN, S02, ..., S16MJ

## Dependencies

### Python Packages
- `numpy`: Numerical computations
- `pandas`: Data processing
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualization
- `scipy`: Statistical functions

### System Dependencies
- Existing GA implementation in `../Simple Optimizer/`
- Metro simulation data in `../data/simulation_log.csv`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the `Simple Optimizer` directory is accessible
2. **Data Not Found**: Check CSV file path in configuration
3. **Memory Issues**: Reduce sample size for large datasets
4. **Slow Performance**: Decrease GA population size and generations

### Debug Mode
Enable detailed logging by setting log level to DEBUG:

```python
logging.getLogger().setLevel(logging.DEBUG)
```

## Extensions

The modular architecture allows for easy extensions:

- **Different Optimization Algorithms**: Replace GA with other optimization methods
- **Additional Monitoring Metrics**: Extend Z-score calculations
- **Advanced Visualization**: Add interactive plots and dashboards
- **Real-Time Data Integration**: Connect to actual metro sensors
- **Multi-Line Support**: Extend to multiple metro lines

## License

This project is part of the Metro Scheduler research system.