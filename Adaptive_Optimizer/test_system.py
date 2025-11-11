"""
Test script for Metro Adaptive Optimization System.

This script runs a quick proof-of-concept test with a small dataset
to validate that all components work together correctly.

Created: November 2025
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from main import MetroAdaptiveOptimizationSystem, MetroOptimizationConfig

# Configure logging for test
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_config() -> MetroOptimizationConfig:
    """
    Create a test configuration with reduced parameters for quick validation.
    
    Returns
    -------
    MetroOptimizationConfig
        Test configuration
    """
    config = MetroOptimizationConfig()
    
    # Reduce dataset size for faster testing
    config.sample_size = 1000  # Small sample for quick test
    
    # Reduce simulation time
    config.total_simulation_time_hours = 2.0  # 2 hours only
    config.monitoring_interval_minutes = 10.0  # Less frequent monitoring
    
    # Lower threshold for more adaptation events
    config.z_score_threshold = 1.5
    
    # Faster GA parameters for testing
    config.ga_params.update({
        'pop_size': 15,
        'generations': 10,  # Very fast for testing
        'headway_min': 5,
        'headway_max': 15,
    })
    
    config.adaptive_ga_params.update({
        'pop_size': 10,
        'generations': 8,  # Very fast for testing
        'headway_min': 5,
        'headway_max': 15,
    })
    
    # Test output directory
    config.output_directory = "test_results"
    config.generate_visualizations = True
    config.save_detailed_logs = True
    
    return config


def run_component_tests(system: MetroAdaptiveOptimizationSystem) -> bool:
    """
    Run individual component tests.
    
    Parameters
    ----------
    system : MetroAdaptiveOptimizationSystem
        The system to test
        
    Returns
    -------
    bool
        True if all tests pass
    """
    logger.info("Running component tests...")
    
    try:
        # Test 1: Data processor
        logger.info("Testing data processor...")
        if not hasattr(system.data_processor, 'data') or len(system.data_processor.data) == 0:
            logger.error("Data processor test failed: No data loaded")
            return False
        logger.info(f"✓ Data processor test passed: {len(system.data_processor.data)} trips loaded")
        
        # Test 2: Monitor
        logger.info("Testing real-time monitor...")
        test_monitoring_result = system.monitor.monitor_step(0.0)
        if not hasattr(test_monitoring_result, 'aggregated_z_score'):
            logger.error("Monitor test failed: Invalid monitoring result")
            return False
        logger.info(f"✓ Monitor test passed: Z-score = {test_monitoring_result.aggregated_z_score:.2f}")
        
        # Test 3: Adaptive GA
        logger.info("Testing adaptive GA...")
        if not hasattr(system.adaptive_ga, 'data_processor'):
            logger.error("Adaptive GA test failed: Not properly initialized")
            return False
        logger.info("✓ Adaptive GA test passed: Properly initialized")
        
        # Test 4: Visualization system
        if system.visualization_system:
            logger.info("Testing visualization system...")
            if not hasattr(system.visualization_system, 'config'):
                logger.error("Visualization test failed: Not properly initialized")
                return False
            logger.info("✓ Visualization system test passed")
        
        logger.info("All component tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Component test failed: {e}")
        return False


def run_integration_test() -> bool:
    """
    Run full integration test with reduced parameters.
    
    Returns
    -------
    bool
        True if test passes
    """
    logger.info("Starting integration test...")
    
    try:
        # Create test system
        config = create_test_config()
        system = MetroAdaptiveOptimizationSystem(config)
        
        # Initialize
        logger.info("Initializing system for test...")
        if not system.initialize_system():
            logger.error("System initialization failed")
            return False
        
        # Run component tests
        if not run_component_tests(system):
            logger.error("Component tests failed")
            return False
        
        # Run initial optimization
        logger.info("Testing initial optimization...")
        if not system.run_initial_optimization():
            logger.error("Initial optimization test failed")
            return False
        
        logger.info(f"✓ Initial optimization completed: fitness = {system.initial_optimization_result.fitness:.2f}")
        
        # Run a few monitoring steps
        logger.info("Testing monitoring and adaptation...")
        for i in range(3):  # Test 3 monitoring steps
            test_time = i * config.monitoring_interval_minutes
            logger.info(f"  Running monitoring step {i+1}/3 at time {test_time:.1f} minutes...")
            monitoring_result = system.monitor.monitor_step(test_time)
            system.monitoring_history.append(monitoring_result)
            
            logger.info(f"  Step {i+1}: Time={test_time:.1f}min, Z-score={monitoring_result.aggregated_z_score:.2f}")
            
            # Test adaptation if threshold exceeded
            if monitoring_result.threshold_exceeded and system.monitor.should_trigger_adaptive_optimization(monitoring_result):
                logger.info("  Threshold exceeded - testing adaptive optimization...")
                adaptation_result = system.adaptive_ga.adaptive_optimization(monitoring_result)
                system.adaptation_history.append(adaptation_result)
                logger.info(f"  ✓ Adaptation completed: improvement = {adaptation_result.improvement_percentage:.1f}%")
        
        # Test report generation
        logger.info("Testing report generation...")
        final_metrics = system.generate_final_report()
        
        if not final_metrics or 'static_optimization' not in final_metrics:
            logger.error("Report generation test failed")
            return False
        
        logger.info("✓ Report generation test passed")
        
        # Print test summary
        print_test_summary(final_metrics, system)
        
        logger.info("Integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


def print_test_summary(final_metrics: dict, system: MetroAdaptiveOptimizationSystem) -> None:
    """
    Print test summary results.
    
    Parameters
    ----------
    final_metrics : dict
        Final metrics from test
    system : MetroAdaptiveOptimizationSystem
        The tested system
    """
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    print(f"\\nTEST CONFIGURATION:")
    print(f"  Sample size: {system.config.sample_size}")
    print(f"  Simulation time: {system.config.total_simulation_time_hours} hours")
    print(f"  Z-score threshold: {system.config.z_score_threshold}")
    print(f"  Monitoring interval: {system.config.monitoring_interval_minutes} minutes")
    
    static = final_metrics['static_optimization']
    print(f"\\nOPTIMIZATION RESULTS:")
    print(f"  Initial fitness: {static['fitness']:.2f}")
    print(f"  Initial waiting time: {static['avg_waiting_time']:.2f} min")
    print(f"  Passengers served: {static['total_passengers_served']}")
    print(f"  Passengers left: {static['total_passengers_left']}")
    
    print(f"\\nMONITORING RESULTS:")
    print(f"  Monitoring points: {len(system.monitoring_history)}")
    print(f"  Adaptations: {len(system.adaptation_history)}")
    print(f"  Threshold exceedances: {sum(1 for r in system.monitoring_history if r.threshold_exceeded)}")
    
    if system.adaptation_history:
        avg_improvement = sum(r.improvement_percentage for r in system.adaptation_history) / len(system.adaptation_history)
        print(f"  Average improvement: {avg_improvement:.1f}%")
    
    print(f"\\nOUTPUT:")
    print(f"  Results saved to: {system.config.output_directory}/")
    
    print("\\n" + "="*60)


def main():
    """Main test function."""
    print("Metro Adaptive Optimization System - Integration Test")
    print("="*60)
    
    start_time = time.time()
    
    try:
        success = run_integration_test()
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        if success:
            print(f"\\n✓ All tests PASSED in {test_duration:.1f} seconds!")
            print("\\nThe system is ready for full simulation.")
            print("Run 'python main.py' for complete optimization with full dataset.")
            return 0
        else:
            print(f"\\n✗ Tests FAILED after {test_duration:.1f} seconds.")
            print("Please check the logs for error details.")
            return 1
            
    except KeyboardInterrupt:
        print("\\n\\nTest interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Test failed with unexpected error: {e}")
        print(f"\\n✗ Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())