# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Define and query information about an AIOps Mitigation task."""

import textwrap
from typing import Any

from aiopslab.orchestrator.tasks.base import Task
from aiopslab.orchestrator.actions.mitigation import MitigationActions
from aiopslab.service.apps.base import Application
from aiopslab.session import SessionItem
from aiopslab.utils.actions import get_actions
from aiopslab.utils.status import InvalidActionError


class MitigationTask(Task):
    """An AIOps anomaly mitigation task."""

    def __init__(self, app: Application):
        super().__init__()
        self.app = app
        self.helm_configs = self.app.helm_configs
        self.app_summary = self.app.get_app_summary()
        self.actions = MitigationActions()

        self.task_desc = """\
            You are an expert DevOps engineer assigned to mitigate anomalies in a deployed service.

            Service Details:
            {app_summary}

            You will begin by analyzing the service's state and telemetry, and then submit a solution that mitigates any detected anomalies.
            Your mitigation can be performed using any of the available APIs.
            """

        self.instructions = """\
            You will respond with one of the above APIs as your next action.
            Please respond in the following format in a markdown code block:
            ```\n<API_NAME>(<API_PARAM1>, <API_PARAM2> ...)\n```

            For instance, if you want to list files in current directory, your response must be exactly:
            
            ```\nexec_shell("ls -l")\n```

            Once your solution is complete and ready for evaluation, you must call:
            
            ```\nsubmit()\n```

            Note:
            - The submit() call for the mitigation task does not take any parameters.
            - A submission via submit() is considered valid if it is made, though this does not necessarily indicate that your solution is correct.

            Please respond with only a single API call (a.k.a., action) per turn without any additional words, labels, or prefixes.
            """

    def get_task_description(self):
        return textwrap.dedent(self.task_desc).format(app_summary=self.app_summary)

    def get_instructions(self):
        return textwrap.dedent(self.instructions)

    def get_available_actions(self):
        return get_actions(task="mitigation")

    def perform_action(self, action_name, *args, **kwargs):
        action_method = getattr(self.actions, action_name, None)

        if action_method is not None and callable(action_method):
            return action_method(*args, **kwargs)
        else:
            raise InvalidActionError(action_name)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        self.add_result("TTM", duration)
        self.common_eval(trace)
        return self.results
    
    def wait_for_pods_stable(self, max_wait_seconds: int = 60, check_interval: int = 5) -> bool:
        """
        시스템이 안정화될 때까지 대기하는 헬퍼 함수.
        AI가 submit한 뒤, 평가 전에 호출하여 Pod들이 복구될 시간을 제공합니다.
        
        Args:
            max_wait_seconds: 최대 대기 시간 (초)
            check_interval: 상태 확인 간격 (초)
        
        Returns:
            bool: 모든 Pod가 정상 상태면 True, 아니면 False
        """
        from time import sleep
        
        max_attempts = max_wait_seconds // check_interval
        
        for attempt in range(max_attempts):
            pod_list = self.kubectl.list_pods(self.namespace)
            all_normal = True

            for pod in pod_list.items:
                if not pod.status.container_statuses:
                    continue
                    
                for container_status in pod.status.container_statuses:
                    if container_status.state.waiting:
                        reason = container_status.state.waiting.reason
                        if reason in ["CrashLoopBackOff", "Error", "ImagePullBackOff", "ErrImagePull"]:
                            print(f"Container {container_status.name} is in error state: {reason}")
                            all_normal = False
                    elif container_status.state.terminated and container_status.state.terminated.reason != "Completed":
                        print(f"Container {container_status.name} is terminated with reason: {container_status.state.terminated.reason}")
                        all_normal = False
                    elif not container_status.ready:
                        print(f"Container {container_status.name} is not ready")
                        all_normal = False

                if not all_normal:
                    break
            
            if all_normal:
                if attempt > 0:
                    print(f"All pods are healthy after {attempt * check_interval} seconds")
                return True
            
            if attempt < max_attempts - 1:  # Don't sleep on last attempt
                sleep(check_interval)

        return False
