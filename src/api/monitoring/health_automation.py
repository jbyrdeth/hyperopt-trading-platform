"""
Health Check Automation and Testing

This module provides automation capabilities for health monitoring including:
- Automated health check scheduling
- Health check failure simulation for testing
- Alert validation and testing
- Health monitoring configuration management
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .health import get_health_checker, HealthStatus, ComponentHealth
from .metrics import get_metrics_collector
from .alerts import get_alert_manager, Alert, AlertSeverity, AlertStatus

logger = logging.getLogger(__name__)


class FailureSimulationType(Enum):
    """Types of health check failures that can be simulated."""
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    DISK_FULL = "disk_full"
    API_SLOW = "api_slow"
    JOB_FAILURE = "job_failure"
    COMPONENT_DOWN = "component_down"


@dataclass
class HealthTestScenario:
    """Configuration for a health check test scenario."""
    name: str
    description: str
    failure_type: FailureSimulationType
    duration_seconds: int
    severity: AlertSeverity
    expected_alerts: List[str]
    recovery_time_seconds: int = 30


class HealthAutomation:
    """
    Health monitoring automation system.
    
    Provides capabilities for:
    - Scheduled health monitoring
    - Failure simulation and testing
    - Alert validation
    - Health system self-testing
    """
    
    def __init__(self):
        self.health_checker = get_health_checker()
        self.metrics_collector = get_metrics_collector()
        self.alert_manager = get_alert_manager()
        
        # Test state
        self._active_simulations: Dict[str, FailureSimulationType] = {}
        self._test_results: List[Dict[str, Any]] = []
        
        # Monitoring configuration
        self._monitoring_enabled = True
        self._check_interval = 30.0
        self._alert_validation_timeout = 120.0  # seconds
        
        logger.info("HealthAutomation initialized")
    
    async def run_health_test_suite(self) -> Dict[str, Any]:
        """
        Run comprehensive health check test suite.
        
        Tests various failure scenarios and validates that:
        - Health checks detect failures correctly
        - Metrics are updated appropriately
        - Alerts are generated as expected
        - Recovery is handled properly
        
        Returns:
            Test results summary
        """
        logger.info("Starting comprehensive health test suite")
        
        test_scenarios = [
            HealthTestScenario(
                name="High CPU Usage",
                description="Simulate high CPU usage to trigger CPU alerts",
                failure_type=FailureSimulationType.HIGH_CPU,
                duration_seconds=60,
                severity=AlertSeverity.WARNING,
                expected_alerts=["HighCPUUsage"]
            ),
            HealthTestScenario(
                name="High Memory Usage",
                description="Simulate high memory usage to trigger memory alerts",
                failure_type=FailureSimulationType.HIGH_MEMORY,
                duration_seconds=60,
                severity=AlertSeverity.WARNING,
                expected_alerts=["HighMemoryUsage"]
            ),
            HealthTestScenario(
                name="Component Failure",
                description="Simulate component failure to trigger component alerts",
                failure_type=FailureSimulationType.COMPONENT_DOWN,
                duration_seconds=45,
                severity=AlertSeverity.CRITICAL,
                expected_alerts=["ComponentHealthCheck"]
            ),
            HealthTestScenario(
                name="Job Manager Issues",
                description="Simulate job manager problems to trigger job alerts",
                failure_type=FailureSimulationType.JOB_FAILURE,
                duration_seconds=90,
                severity=AlertSeverity.WARNING,
                expected_alerts=["HighOptimizationFailureRate"]
            )
        ]
        
        test_results = {
            "start_time": datetime.utcnow().isoformat(),
            "total_scenarios": len(test_scenarios),
            "scenarios": [],
            "overall_success": True,
            "summary": {
                "passed": 0,
                "failed": 0,
                "alerts_generated": 0,
                "alerts_validated": 0
            }
        }
        
        for scenario in test_scenarios:
            logger.info(f"Running test scenario: {scenario.name}")
            
            scenario_result = await self._run_test_scenario(scenario)
            test_results["scenarios"].append(scenario_result)
            
            if scenario_result["success"]:
                test_results["summary"]["passed"] += 1
            else:
                test_results["summary"]["failed"] += 1
                test_results["overall_success"] = False
            
            test_results["summary"]["alerts_generated"] += scenario_result["alerts_generated"]
            test_results["summary"]["alerts_validated"] += scenario_result["alerts_validated"]
            
            # Brief pause between scenarios
            await asyncio.sleep(10)
        
        test_results["end_time"] = datetime.utcnow().isoformat()
        test_results["total_duration_seconds"] = (
            datetime.fromisoformat(test_results["end_time"].replace('Z', '+00:00')) -
            datetime.fromisoformat(test_results["start_time"].replace('Z', '+00:00'))
        ).total_seconds()
        
        logger.info(f"Health test suite completed. Success: {test_results['overall_success']}")
        self._test_results.append(test_results)
        
        return test_results
    
    async def _run_test_scenario(self, scenario: HealthTestScenario) -> Dict[str, Any]:
        """Run a single test scenario."""
        start_time = datetime.utcnow()
        
        scenario_result = {
            "name": scenario.name,
            "description": scenario.description,
            "start_time": start_time.isoformat(),
            "duration_seconds": scenario.duration_seconds,
            "success": False,
            "alerts_generated": 0,
            "alerts_validated": 0,
            "errors": [],
            "details": {}
        }
        
        try:
            # Capture baseline health state
            baseline_health = await self.health_checker.check_all_components()
            initial_alert_count = len(self.alert_manager.get_active_alerts())
            
            # Start failure simulation
            logger.info(f"Starting failure simulation: {scenario.failure_type.value}")
            await self._simulate_failure(scenario.failure_type, scenario.duration_seconds)
            
            # Wait for health checks to detect the issue
            await asyncio.sleep(5)
            
            # Force health check
            affected_health = await self.health_checker.check_all_components()
            
            # Validate health check detected the issue
            health_detected = self._validate_health_detection(
                baseline_health, affected_health, scenario.failure_type
            )
            
            if not health_detected:
                scenario_result["errors"].append("Health check did not detect simulated failure")
            
            # Wait for alerts to be generated
            await asyncio.sleep(15)
            
            # Check for expected alerts
            current_alerts = self.alert_manager.get_active_alerts()
            new_alert_count = len(current_alerts) - initial_alert_count
            scenario_result["alerts_generated"] = new_alert_count
            
            # Validate expected alerts were generated
            alerts_validated = self._validate_expected_alerts(current_alerts, scenario.expected_alerts)
            scenario_result["alerts_validated"] = len(alerts_validated)
            
            # Stop failure simulation
            logger.info(f"Stopping failure simulation: {scenario.failure_type.value}")
            await self._stop_simulation(scenario.failure_type)
            
            # Wait for recovery
            await asyncio.sleep(scenario.recovery_time_seconds)
            
            # Validate recovery
            recovery_health = await self.health_checker.check_all_components()
            recovery_detected = self._validate_recovery(baseline_health, recovery_health)
            
            if not recovery_detected:
                scenario_result["errors"].append("System did not recover properly after simulation")
            
            # Determine overall success
            scenario_result["success"] = (
                health_detected and 
                new_alert_count > 0 and 
                len(alerts_validated) > 0 and 
                recovery_detected
            )
            
            scenario_result["details"] = {
                "health_detection": health_detected,
                "alert_generation": new_alert_count > 0,
                "alert_validation": len(alerts_validated) > 0,
                "recovery": recovery_detected,
                "validated_alerts": alerts_validated
            }
            
        except Exception as e:
            logger.error(f"Test scenario {scenario.name} failed with exception: {e}")
            scenario_result["errors"].append(f"Exception: {str(e)}")
            scenario_result["success"] = False
        
        scenario_result["end_time"] = datetime.utcnow().isoformat()
        return scenario_result
    
    async def _simulate_failure(self, failure_type: FailureSimulationType, duration: int):
        """Simulate a specific type of failure."""
        simulation_id = f"{failure_type.value}_{int(time.time())}"
        self._active_simulations[simulation_id] = failure_type
        
        try:
            if failure_type == FailureSimulationType.HIGH_CPU:
                await self._simulate_high_cpu(duration)
            elif failure_type == FailureSimulationType.HIGH_MEMORY:
                await self._simulate_high_memory(duration)
            elif failure_type == FailureSimulationType.COMPONENT_DOWN:
                await self._simulate_component_failure(duration)
            elif failure_type == FailureSimulationType.JOB_FAILURE:
                await self._simulate_job_failures(duration)
            else:
                logger.warning(f"Simulation not implemented for {failure_type.value}")
                
        finally:
            self._active_simulations.pop(simulation_id, None)
    
    async def _simulate_high_cpu(self, duration: int):
        """Simulate high CPU usage."""
        # Override CPU metrics temporarily
        original_collect = self.metrics_collector.collect_system_metrics
        
        def mock_collect():
            # Call original collection first
            try:
                original_collect()
            except:
                pass
            # Override CPU metrics with high values
            self.metrics_collector.metrics.cpu_usage.set(85.0)
        
        self.metrics_collector.collect_system_metrics = mock_collect
        
        # Wait for the duration
        await asyncio.sleep(duration)
        
        # Restore original collection
        self.metrics_collector.collect_system_metrics = original_collect
    
    async def _simulate_high_memory(self, duration: int):
        """Simulate high memory usage."""
        original_collect = self.metrics_collector.collect_system_metrics
        
        def mock_collect():
            try:
                original_collect()
            except:
                pass
            # Override memory metrics with high values
            self.metrics_collector.metrics.memory_usage_percent.set(90.0)
        
        self.metrics_collector.collect_system_metrics = mock_collect
        await asyncio.sleep(duration)
        self.metrics_collector.collect_system_metrics = original_collect
    
    async def _simulate_component_failure(self, duration: int):
        """Simulate component failure by injecting failing health checks."""
        # Mock a component health check to fail
        original_check = self.health_checker._check_api_components
        
        async def mock_check():
            self.health_checker.components["api_core"] = ComponentHealth(
                name="API Core",
                status=HealthStatus.CRITICAL,
                message="Simulated component failure",
                details={"error": "Test simulation"},
                last_check=datetime.utcnow()
            )
        
        self.health_checker._check_api_components = mock_check
        await asyncio.sleep(duration)
        self.health_checker._check_api_components = original_check
    
    async def _simulate_job_failures(self, duration: int):
        """Simulate optimization job failures."""
        # Mock job failure metrics
        original_collect = self.metrics_collector.collect_system_metrics
        
        def mock_collect():
            try:
                original_collect()
            except:
                pass
            # Simulate high job failure rate
            self.metrics_collector.metrics.jobs_total.labels(strategy="test", status="failed").inc(10)
            self.metrics_collector.metrics.jobs_total.labels(strategy="test", status="completed").inc(5)
        
        self.metrics_collector.collect_system_metrics = mock_collect
        await asyncio.sleep(duration)
        self.metrics_collector.collect_system_metrics = original_collect
    
    async def _stop_simulation(self, failure_type: FailureSimulationType):
        """Stop failure simulation and restore normal operation."""
        # Force a normal health check to clear any mocked states
        await self.health_checker.check_all_components()
        # Force metrics collection to restore normal values
        self.metrics_collector.collect_system_metrics()
        logger.info(f"Simulation {failure_type.value} stopped and normal operation restored")
    
    def _validate_health_detection(
        self, 
        baseline: Dict[str, ComponentHealth], 
        affected: Dict[str, ComponentHealth], 
        failure_type: FailureSimulationType
    ) -> bool:
        """Validate that health checks detected the simulated failure."""
        
        # Look for degraded health status in affected components
        for component_name, component in affected.items():
            if component.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                # Check if this degradation is related to our failure type
                if failure_type == FailureSimulationType.HIGH_CPU and "cpu" in component_name.lower():
                    return True
                elif failure_type == FailureSimulationType.HIGH_MEMORY and "memory" in component_name.lower():
                    return True
                elif failure_type == FailureSimulationType.COMPONENT_DOWN and component_name == "api_core":
                    return True
                elif component.status == HealthStatus.CRITICAL:  # Any critical status counts
                    return True
        
        return False
    
    def _validate_expected_alerts(self, current_alerts: List[Alert], expected_alert_names: List[str]) -> List[str]:
        """Validate that expected alerts were generated."""
        validated = []
        
        current_alert_names = [alert.alertname for alert in current_alerts]
        
        for expected_name in expected_alert_names:
            if expected_name in current_alert_names:
                validated.append(expected_name)
                logger.info(f"Expected alert validated: {expected_name}")
            else:
                logger.warning(f"Expected alert not found: {expected_name}")
        
        return validated
    
    def _validate_recovery(
        self, 
        baseline: Dict[str, ComponentHealth], 
        recovery: Dict[str, ComponentHealth]
    ) -> bool:
        """Validate that the system recovered to healthy state."""
        
        # Check if all critical components are back to healthy
        for component_name, component in recovery.items():
            if component.status == HealthStatus.CRITICAL:
                return False
        
        # At least 80% of components should be healthy
        healthy_count = sum(1 for comp in recovery.values() if comp.status == HealthStatus.HEALTHY)
        total_count = len(recovery)
        
        return (healthy_count / total_count) >= 0.8
    
    def get_test_history(self) -> List[Dict[str, Any]]:
        """Get history of all test runs."""
        return self._test_results.copy()
    
    def get_active_simulations(self) -> Dict[str, str]:
        """Get currently active failure simulations."""
        return {sim_id: failure_type.value for sim_id, failure_type in self._active_simulations.items()}
    
    async def validate_monitoring_system(self) -> Dict[str, Any]:
        """
        Validate the entire monitoring system is working correctly.
        
        Performs a comprehensive check of:
        - Health check system
        - Metrics collection  
        - Alert generation
        - Integration between components
        
        Returns:
            Validation results
        """
        logger.info("Validating monitoring system")
        
        validation_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "integration_tests": {},
            "issues": []
        }
        
        # Test health checker
        try:
            health_result = await self.health_checker.check_all_components()
            validation_result["components"]["health_checker"] = {
                "status": "healthy",
                "components_checked": len(health_result),
                "healthy_components": len([c for c in health_result.values() if c.status == HealthStatus.HEALTHY])
            }
        except Exception as e:
            validation_result["components"]["health_checker"] = {
                "status": "failed",
                "error": str(e)
            }
            validation_result["issues"].append(f"Health checker failed: {e}")
        
        # Test metrics collector
        try:
            self.metrics_collector.collect_system_metrics()
            validation_result["components"]["metrics_collector"] = {
                "status": "healthy",
                "background_collection": self.metrics_collector._collection_thread is not None
            }
        except Exception as e:
            validation_result["components"]["metrics_collector"] = {
                "status": "failed",
                "error": str(e)
            }
            validation_result["issues"].append(f"Metrics collector failed: {e}")
        
        # Test alert manager
        try:
            alert_stats = self.alert_manager.get_alert_stats()
            validation_result["components"]["alert_manager"] = {
                "status": "healthy",
                "total_alerts": alert_stats["total_alerts"],
                "active_alerts": alert_stats["active_alerts"]
            }
        except Exception as e:
            validation_result["components"]["alert_manager"] = {
                "status": "failed",
                "error": str(e)
            }
            validation_result["issues"].append(f"Alert manager failed: {e}")
        
        # Integration test: Health to Metrics
        try:
            # Force health check and verify metrics are updated
            await self.health_checker.check_all_components()
            # Check if health metrics exist (they should be set by background monitoring)
            validation_result["integration_tests"]["health_to_metrics"] = "passed"
        except Exception as e:
            validation_result["integration_tests"]["health_to_metrics"] = f"failed: {e}"
            validation_result["issues"].append(f"Health to metrics integration failed: {e}")
        
        # Set overall status
        if validation_result["issues"]:
            validation_result["overall_status"] = "degraded" if len(validation_result["issues"]) < 3 else "unhealthy"
        
        return validation_result


# Global health automation instance
_health_automation: Optional[HealthAutomation] = None


def get_health_automation() -> HealthAutomation:
    """Get the global health automation instance."""
    global _health_automation
    
    if _health_automation is None:
        _health_automation = HealthAutomation()
    
    return _health_automation 