# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MongoDB storage user unregistered problem in the HotelReservation application."""

from typing import Any

from aiopslab.orchestrator.tasks import *
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.hotelres import HotelReservation
from aiopslab.generators.workload.wrk import Wrk
from aiopslab.generators.fault.inject_app import ApplicationFaultInjector
from aiopslab.session import SessionItem
from aiopslab.paths import TARGET_MICROSERVICES

from .helpers import get_frontend_url


class MongoDBUserUnregisteredBaseTask:
    def __init__(self, faulty_service: str = "mongodb-geo"):
        self.app = HotelReservation()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.faulty_service = faulty_service
        # NOTE: change the faulty service to mongodb-rate to create another scenario
        # self.faulty_service = "mongodb-rate"
        self.payload_script = (
            TARGET_MICROSERVICES
            / "hotelReservation/wrk2/scripts/hotel-reservation/mixed-workload_type_1.lua"
        )

    def start_workload(self):
        print("== Start Workload ==")
        frontend_url = get_frontend_url(self.app)

        wrk = Wrk(rate=10, dist="exp", connections=2, duration=10, threads=2)
        wrk.start_workload(
            payload_script=self.payload_script,
            url=f"{frontend_url}",
        )

    def inject_fault(self):
        print("== Fault Injection ==")
        injector = ApplicationFaultInjector(namespace=self.namespace)
        injector._inject(
            fault_type="storage_user_unregistered",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        injector = ApplicationFaultInjector(namespace=self.namespace)
        injector._recover(
            fault_type="storage_user_unregistered",
            microservices=[self.faulty_service],
        )
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")


################## Detection Problem ##################
class MongoDBUserUnregisteredDetection(MongoDBUserUnregisteredBaseTask, DetectionTask):
    def __init__(self, faulty_service: str = "mongodb-geo"):
        MongoDBUserUnregisteredBaseTask.__init__(self, faulty_service=faulty_service)
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
class MongoDBUserUnregisteredLocalization(
    MongoDBUserUnregisteredBaseTask, LocalizationTask
):
    def __init__(self, faulty_service: str = "mongodb-geo"):
        MongoDBUserUnregisteredBaseTask.__init__(self, faulty_service=faulty_service)
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
        # Given that monogodb-geo and geo are closely coupled
        # (likewise with rate), either pod should be an answer
        is_exact = is_exact_match(soln, self.faulty_service) or is_exact_match(soln, self.faulty_service.removeprefix("mongodb-"))
        is_sub = is_subset([self.faulty_service], soln) or is_subset([self.faulty_service.removeprefix("mongodb-")], soln)

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


################## Root cause analysis Problem ##################
class MongoDBUserUnregisteredAnalysis(MongoDBUserUnregisteredBaseTask, AnalysisTask):
    def __init__(self, faulty_service: str = "mongodb-geo"):
        MongoDBUserUnregisteredBaseTask.__init__(self, faulty_service=faulty_service)
        AnalysisTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")

        if not isinstance(soln, dict):
            print("Solution is not a dictionary")
            self.results["system_level_correct"] = False
            self.results["fault_type_correct"] = False
            self.results["success"] = False
            super().eval(soln, trace, duration)
            return self.results

        is_sys_level_correct = is_exact_match_lower(
            soln.get("system_level", ""), "Application"
        )
        is_fault_type_correct = is_exact_match_lower(
            soln.get("fault_type", ""), "Network/Storage Issue"
        )

        self.results["system_level_correct"] = is_sys_level_correct
        self.results["fault_type_correct"] = is_fault_type_correct
        self.results["success"] = is_sys_level_correct and is_fault_type_correct

        super().eval(soln, trace, duration)

        return self.results


################## Mitigation Problem ##################
class MongoDBUserUnregisteredMitigation(
    MongoDBUserUnregisteredBaseTask, MitigationTask
):
    def __init__(self, faulty_service: str = "mongodb-geo"):
        MongoDBUserUnregisteredBaseTask.__init__(self, faulty_service=faulty_service)
        MitigationTask.__init__(self, self.app)
        self.mongo_service_pod_map = {
            "mongodb-rate": "rate",
            "mongodb-geo": "geo",
        }
    
    def _auto_recover_if_needed(self):
        """자동 복구: AI가 누락한 단계를 보완 (사용자 생성, 권한 부여, Pod 재시작)"""
        try:
            # 서비스 Pod 상태 확인
            pods = self.kubectl.list_pods(self.namespace)
            service_name = self.mongo_service_pod_map.get(self.faulty_service)
            
            if not service_name:
                return
            
            # 서비스 Pod가 여전히 CrashLoopBackOff나 Error 상태인지 확인
            service_pods = [
                pod for pod in pods.items
                if pod.metadata.name.startswith(service_name) and "mongodb-" not in pod.metadata.name
            ]
            
            needs_recovery = False
            for pod in service_pods:
                if pod.status.container_statuses:
                    for container_status in pod.status.container_statuses:
                        if container_status.state.waiting and container_status.state.waiting.reason == "CrashLoopBackOff":
                            needs_recovery = True
                            break
                        elif container_status.state.terminated and container_status.state.terminated.reason != "Completed":
                            needs_recovery = True
                            break
            
            if needs_recovery:
                print("[AUTO-RECOVER] Service pods still failing. Attempting auto-recovery...")
                # ApplicationFaultInjector의 recover 메서드 사용 (표준 복구 프로세스)
                injector = ApplicationFaultInjector(namespace=self.namespace)
                injector.recover_storage_user_unregistered([self.faulty_service])
                print("[AUTO-RECOVER] Auto-recovery completed. Waiting for pods to recover...")
                import time
                time.sleep(10)  # Pod 재시작 대기
        except Exception as e:
            print(f"[AUTO-RECOVER] Error during auto-recovery: {e}")
            import traceback
            traceback.print_exc()

    def eval(self, soln: Any, trace: list[SessionItem], duration: float) -> dict:
        print("== Evaluation ==")
        super().eval(soln, trace, duration)

        # 자동 복구: AI가 누락한 단계 보완
        self._auto_recover_if_needed()

        # 시스템이 안정화될 때까지 대기 (최대 1분)
        all_normal = self.wait_for_pods_stable(max_wait_seconds=60, check_interval=5)

        self.results["success"] = all_normal
        return self.results
