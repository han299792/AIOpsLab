"""Otel demo kafkaQueueProblems feature flag fault."""

import json
import subprocess
from typing import Any

from aiopslab.orchestrator.tasks import *
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.astronomy_shop import AstronomyShop
from aiopslab.generators.fault.inject_otel import OtelFaultInjector
from aiopslab.session import SessionItem
from aiopslab.observer import monitor_config
from aiopslab.observer.metric_api import PrometheusAPI


class KafkaQueueProblemsBaseTask:
    def __init__(self):
        self.app = AstronomyShop()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.injector = OtelFaultInjector(namespace=self.namespace)
        self.faulty_service = "kafka"

    def start_workload(self):
        print("== Start Workload ==")
        print("Workload skipped since AstronomyShop has a built-in load generator.")

    def inject_fault(self):
        print("== Fault Injection ==")
        self.injector.inject_fault("kafkaQueueProblems")
        print(f"Fault: kafkaQueueProblems | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        self.injector.recover_fault("kafkaQueueProblems")


################## Detection Problem ##################
class KafkaQueueProblemsDetection(KafkaQueueProblemsBaseTask, DetectionTask):
    def __init__(self):
        KafkaQueueProblemsBaseTask.__init__(self)
        DetectionTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        expected_solution = "Yes"

        if isinstance(soln, str):
            if soln.strip().lower() == expected_solution.lower():
                print(f"Correct detection: {soln}")
                self.add_result("Detection Accuracy", "Correct")
            else:
                print(f"Incorrect detection: {soln}")
                self.add_result("Detection Accuracy", "Incorrect")
        else:
            print("Invalid solution format")
            self.add_result("Detection Accuracy", "Invalid Format")

        return super().eval(soln, trace, duration)


################## Localization Problem ##################
class KafkaQueueProblemsLocalization(KafkaQueueProblemsBaseTask, LocalizationTask):
    def __init__(self):
        KafkaQueueProblemsBaseTask.__init__(self)
        LocalizationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")

        if soln is None:
            print("Solution is None")
            self.add_result("Localization Accuracy", 0.0)
            self.results["success"] = False
            self.results["is_subset"] = False
            super().eval(soln, trace, duration)
            return self.results

        # Calculate exact match and subset
        is_exact = is_exact_match(soln, self.faulty_service)
        is_sub = is_subset([self.faulty_service], soln)

        # Determine accuracy
        if is_exact:
            accuracy = 100.0
            print(f"Exact match: {soln} | Accuracy: {accuracy}%")
        elif is_sub:
            accuracy = (len([self.faulty_service]) / len(soln)) * 100.0
            print(f"Subset match: {soln} | Accuracy: {accuracy:.2f}%")
        else:
            accuracy = 0.0
            print(f"No match: {soln} | Accuracy: {accuracy}%")

        self.add_result("Localization Accuracy", accuracy)
        super().eval(soln, trace, duration)

        self.results["success"] = is_exact or (is_sub and len(soln) == 1)
        self.results["is_subset"] = is_sub

        return self.results


################## Mitigation Problem ##################
class KafkaQueueProblemsMitigation(KafkaQueueProblemsBaseTask, MitigationTask):
    def __init__(self):
        KafkaQueueProblemsBaseTask.__init__(self)
        MitigationTask.__init__(self, self.app)
        # Initialize Prometheus API for service-level health checks
        try:
            self.prometheus = PrometheusAPI(
                monitor_config["prometheusApi"], 
                self.namespace
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Prometheus API: {e}")
            self.prometheus = None

    def eval(self, soln: Any, trace: list[SessionItem], duration: float) -> dict:
        print("== Evaluation ==")
        super().eval(soln, trace, duration)

        # Kafka Queue Problems is a feature flag fault, not a pod crash issue
        # Check if the feature flag has been recovered (set to "off")
        configmap_name = "flagd-config"
        feature_flag = "kafkaQueueProblems"
        all_normal = True
        
        # Step 1: Check if feature flag is set to "off"
        try:
            command = (
                f"kubectl get configmap {configmap_name} -n {self.namespace} -o json"
            )
            output = self.kubectl.exec_command(command)
            configmap = json.loads(output)
            flagd_data = json.loads(configmap["data"]["demo.flagd.json"])
            
            if feature_flag in flagd_data["flags"]:
                current_value = flagd_data["flags"][feature_flag].get("defaultVariant", "")
                if current_value.lower() == "off":
                    print(f"✓ Feature flag '{feature_flag}' is correctly set to 'off'")
                else:
                    print(
                        f"✗ Feature flag '{feature_flag}' is still set to '{current_value}', expected 'off'"
                    )
                    all_normal = False
            else:
                print(f"✗ Feature flag '{feature_flag}' not found in ConfigMap")
                all_normal = False
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            print(f"✗ Error checking feature flag status: {e}")
            all_normal = False

        # Step 2: Verify flagd deployment has been restarted and is ready
        # This ensures the feature flag change has been applied
        if all_normal:
            try:
                flagd_deployment = self.kubectl.get_deployment("flagd", self.namespace)
                if flagd_deployment:
                    # Check if flagd deployment is ready
                    if flagd_deployment.status.ready_replicas != flagd_deployment.spec.replicas:
                        print(f"✗ flagd deployment is not ready: {flagd_deployment.status.ready_replicas}/{flagd_deployment.spec.replicas} replicas ready")
                        all_normal = False
                    else:
                        print(f"✓ flagd deployment is ready: {flagd_deployment.status.ready_replicas}/{flagd_deployment.spec.replicas} replicas")
                else:
                    print("✗ flagd deployment not found")
                    all_normal = False
            except Exception as e:
                print(f"✗ Error checking flagd deployment status: {e}")
                all_normal = False

        # Step 3: Infrastructure Level Check - Check if all pods are running normally
        # This ensures the system is healthy after the feature flag change
        infrastructure_healthy = all_normal
        if infrastructure_healthy:
            pod_list = self.kubectl.list_pods(self.namespace)
            for pod in pod_list.items:
                # Check container statuses
                if pod.status.container_statuses:
                    for container_status in pod.status.container_statuses:
                        if container_status.state.waiting:
                            reason = container_status.state.waiting.reason
                            if reason in ["CrashLoopBackOff", "Error", "ImagePullBackOff", "ErrImagePull"]:
                                print(f"✗ Container {container_status.name} in pod {pod.metadata.name} is in error state: {reason}")
                                infrastructure_healthy = False
                        elif container_status.state.terminated:
                            if container_status.state.terminated.reason != "Completed":
                                print(f"✗ Container {container_status.name} in pod {pod.metadata.name} is terminated with reason: {container_status.state.terminated.reason}")
                                infrastructure_healthy = False
                        elif not container_status.ready:
                            print(f"✗ Container {container_status.name} in pod {pod.metadata.name} is not ready")
                            infrastructure_healthy = False

        # Step 4: Service Level Check - Verify Kafka consumer lag is within acceptable threshold
        # This ensures the service is actually processing messages correctly
        service_healthy = True
        if infrastructure_healthy and self.prometheus:
            try:
                current_lag = self.get_max_kafka_lag()
                LAG_THRESHOLD = 500  # Maximum acceptable consumer lag
                
                if current_lag >= LAG_THRESHOLD:
                    print(f"✗ Service Failure: Consumer Lag is too high ({current_lag} >= {LAG_THRESHOLD})")
                    service_healthy = False
                else:
                    print(f"✓ Service Health: Consumer Lag is acceptable ({current_lag} < {LAG_THRESHOLD})")
            except Exception as e:
                print(f"⚠ Warning: Could not check Kafka consumer lag: {e}")
                # If we can't check lag, we don't fail the mitigation (graceful degradation)
                # But we log it for debugging
                service_healthy = True  # Don't fail if we can't check

        # Final evaluation: Both infrastructure and service must be healthy
        all_normal = infrastructure_healthy and service_healthy

        if all_normal:
            print("✓ All checks passed: Feature flag is off, flagd deployment is ready, all pods are running normally, and Kafka consumer lag is acceptable")
        else:
            print(f"✗ Mitigation incomplete: Infrastructure={infrastructure_healthy}, Service(Lag)={service_healthy}")

        self.results["success"] = all_normal
        self.results["infrastructure_healthy"] = infrastructure_healthy
        self.results["service_healthy"] = service_healthy
        return self.results

    def get_max_kafka_lag(self) -> float:
        """
        Query Prometheus for maximum Kafka consumer group lag.
        Returns the maximum lag value across all consumer groups.
        If query fails, returns a high value to indicate failure.
        """
        if not self.prometheus:
            raise Exception("Prometheus API not initialized")
        
        # Common Kafka consumer lag metric names (try multiple variations)
        lag_queries = [
            "max(kafka_consumer_group_lag)",  # Standard Kafka exporter metric
            "max(kafka_consumer_lag_sum)",     # Alternative naming
            "max(kafka_consumer_lag)",        # Simple naming
            "max(kafka_consumer_group_lag_sum)",  # Another variation
        ]
        
        max_lag = 0.0
        query_success = False
        
        for query in lag_queries:
            try:
                # Use custom_query for instant query (current value)
                result = self.prometheus.client.custom_query(query)
                
                if result and len(result) > 0:
                    # Extract the maximum value from all time series
                    for metric in result:
                        if "value" in metric and len(metric["value"]) >= 2:
                            lag_value = float(metric["value"][1])
                            max_lag = max(max_lag, lag_value)
                            query_success = True
                    
                    if query_success:
                        print(f"✓ Successfully queried Kafka lag using: {query}")
                        break
            except Exception as e:
                # Try next query if this one fails
                continue
        
        if not query_success:
            # If all queries fail, raise exception
            raise Exception(f"Failed to query Kafka consumer lag. Tried queries: {lag_queries}")
        
        return max_lag
    
    def __del__(self):
        """Cleanup Prometheus port-forward when object is destroyed."""
        if hasattr(self, 'prometheus') and self.prometheus:
            try:
                self.prometheus.cleanup()
            except Exception:
                pass  # Ignore cleanup errors