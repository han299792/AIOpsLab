# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Interface to K8S controller service."""

import json
import time
import subprocess
from rich.console import Console
from kubernetes import client, config
from kubernetes.client.rest import ApiException


class KubeCtl:

    def __init__(self):

        """Initialize the KubeCtl object and load the Kubernetes configuration."""

        import os
        from pathlib import Path
        
        # Support parallel execution via AIOPSLAB_CLUSTER environment variable
        # If AIOPSLAB_CLUSTER is set, use kind-{cluster} context
        cluster_env = os.environ.get('AIOPSLAB_CLUSTER')
        
        try:
            # 1. Pod 내부에서 실행 중인지 먼저 시도 (In-Cluster Config)
            # Pod 내부에서는 ServiceAccount의 권한(RoleBinding)을 사용하여 인증
            config.load_incluster_config()
        except config.ConfigException:
            # 2. 실패하면 로컬 설정 파일 시도 (개발/로컬 환경용)
            try:
                # Determine kubeconfig path
                kubeconfig_path = os.environ.get('KUBECONFIG')
                if not kubeconfig_path:
                    kubeconfig_path = '~/.kube/config'
                # Expand ~ to home directory
                kubeconfig_path = os.path.expanduser(kubeconfig_path)
                
                if cluster_env:
                    # For kind clusters, use kind-{cluster} context
                    context = f"kind-{cluster_env}"
                    try:
                        config.load_kube_config(context=context, config_file=kubeconfig_path)
                    except (config.ConfigException, FileNotFoundError):
                        # Fallback to default kubeconfig if context doesn't exist
                        config.load_kube_config(config_file=kubeconfig_path)
                else:
                    # For regular Kubernetes clusters, use explicit kubeconfig path
                    config.load_kube_config(config_file=kubeconfig_path)
            except (config.ConfigException, FileNotFoundError) as e:
                error_msg = (
                    f"Failed to load Kubernetes configuration: {e}\n"
                    f"Kubeconfig path: {kubeconfig_path if 'kubeconfig_path' in locals() else 'N/A'}\n"
                    f"Please check:\n"
                    f"  1. kubectl is working: try 'kubectl cluster-info'\n"
                    f"  2. KUBECONFIG environment variable is set correctly (if using custom path)\n"
                    f"  3. Kubernetes cluster is accessible\n"
                    f"  4. If running in a Pod, ensure ServiceAccount and RoleBinding are configured"
                )
                raise RuntimeError(error_msg)
        
        # Create API clients
        try:
            self.core_v1_api = client.CoreV1Api()
            self.apps_v1_api = client.AppsV1Api()
        except Exception as e:
            error_msg = (
                f"Failed to create Kubernetes API clients: {e}\n"
                f"This might indicate a connection issue. Please verify:\n"
                f"  1. kubectl cluster-info works\n"
                f"  2. Network connectivity to the API server\n"
                f"  3. Certificates are valid (for local config) or ServiceAccount permissions (for in-cluster config)"
            )
            raise RuntimeError(error_msg) from e
        
    def list_namespaces(self):
        """Return a list of all namespaces in the cluster."""
        try:
            return self.core_v1_api.list_namespace()
        except Exception as e:
            error_msg = (
                f"Failed to list namespaces: {e}\n"
                f"This might indicate a connection issue. Please verify:\n"
                f"  1. kubectl cluster-info works\n"
                f"  2. Network connectivity to the API server\n"
                f"  3. Certificates are valid"
            )
            raise RuntimeError(error_msg) from e

    def list_pods(self, namespace, debug=False):
        """Return a list of all pods within a specified namespace."""
        try:
            return self.core_v1_api.list_namespaced_pod(namespace)
        except Exception as e:
            error_msg = (
                f"Failed to list pods in namespace '{namespace}': {e}\n"
                f"This might indicate a connection issue. Please verify:\n"
                f"  1. kubectl cluster-info works\n"
                f"  2. Network connectivity to the API server\n"
                f"  3. Namespace '{namespace}' exists"
            )
            raise RuntimeError(error_msg) from e

    def list_services(self, namespace):
        """Return a list of all services within a specified namespace."""
        try:
            return self.core_v1_api.list_namespaced_service(namespace)
        except Exception as e:
            error_msg = (
                f"Failed to list services in namespace '{namespace}': {e}\n"
                f"This might indicate a connection issue. Please verify:\n"
                f"  1. kubectl cluster-info works\n"
                f"  2. Network connectivity to the API server\n"
                f"  3. Namespace '{namespace}' exists"
            )
            raise RuntimeError(error_msg) from e

    def get_cluster_ip(self, service_name, namespace):
        """Retrieve the cluster IP address of a specified service within a namespace."""
        service_info = self.core_v1_api.read_namespaced_service(service_name, namespace)
        return service_info.spec.cluster_ip  # type: ignore
    
    def get_container_runtime(self):
        """
            Retrieve the container runtime used by the cluster.
            If the cluster uses multiple container runtimes, the first one found will be returned.
        """
        try:
            nodes = self.core_v1_api.list_node()
            for node in nodes.items:
                for status in node.status.conditions:
                    if status.type == "Ready" and status.status == "True":
                        return node.status.node_info.container_runtime_version
        except Exception as e:
            error_msg = (
                f"Failed to get container runtime: {e}\n"
                f"This might indicate a connection issue. Please verify:\n"
                f"  1. kubectl cluster-info works\n"
                f"  2. Network connectivity to the API server\n"
                f"  3. Certificates are valid"
            )
            raise RuntimeError(error_msg) from e

    def get_pod_name(self, namespace, label_selector):
        """Get the name of the first pod in a namespace that matches a given label selector."""
        pod_info = self.core_v1_api.list_namespaced_pod(
            namespace, label_selector=label_selector
        )
        return pod_info.items[0].metadata.name

    def get_pod_logs(self, pod_name, namespace):
        """Retrieve the logs of a specified pod within a namespace."""
        return self.core_v1_api.read_namespaced_pod_log(pod_name, namespace)

    def get_service_json(self, service_name, namespace, deserialize=True):
        """Retrieve the JSON description of a specified service within a namespace."""
        command = f"kubectl get service {service_name} -n {namespace} -o json"
        result = self.exec_command(command)

        return json.loads(result) if deserialize else result

    def get_deployment(self, name: str, namespace: str):
        """Fetch the deployment configuration."""
        try:
            return self.apps_v1_api.read_namespaced_deployment(name, namespace)
        except Exception as e:
            error_msg = (
                f"Failed to get deployment '{name}' in namespace '{namespace}': {e}\n"
                f"This might indicate a connection issue. Please verify:\n"
                f"  1. kubectl cluster-info works\n"
                f"  2. Network connectivity to the API server\n"
                f"  3. Deployment '{name}' exists in namespace '{namespace}'"
            )
            raise RuntimeError(error_msg) from e

    def wait_for_ready(self, namespace, sleep=2, max_wait=300):
        """Wait for all pods in a namespace to be in a Ready state before proceeding."""

        console = Console()
        console.log(f"[bold green]Waiting for all pods in namespace '{namespace}' to be ready...")

        last_log_time = 0
        log_interval = 30  # Log pod status every 30 seconds

        with console.status("[bold green]Waiting for pods to be ready...") as status:
            wait = 0

            while wait < max_wait:
                try:
                    # wait_for_ready에서는 debug=False로 설정하여 로그 출력 최소화
                    pod_list = self.list_pods(namespace, debug=False)
                    
                    # Log pod status periodically for debugging
                    if wait - last_log_time >= log_interval or wait == 0:
                        if not pod_list.items:
                            console.log(f"[yellow]No pods found in namespace '{namespace}' yet (waited {wait}s)...")
                        else:
                            # Log detailed pod status
                            for pod in pod_list.items:
                                pod_name = pod.metadata.name
                                phase = pod.status.phase
                                ready = "Unknown"
                                container_statuses_info = []
                                
                                if pod.status.container_statuses:
                                    ready_count = sum(1 for cs in pod.status.container_statuses if cs.ready)
                                    total_count = len(pod.status.container_statuses)
                                    ready = f"{ready_count}/{total_count}"
                                    
                                    for cs in pod.status.container_statuses:
                                        if cs.state.waiting:
                                            container_statuses_info.append(f"{cs.name}: Waiting ({cs.state.waiting.reason})")
                                        elif cs.state.terminated:
                                            container_statuses_info.append(f"{cs.name}: Terminated ({cs.state.terminated.reason})")
                                        elif cs.ready:
                                            container_statuses_info.append(f"{cs.name}: Ready")
                                        else:
                                            container_statuses_info.append(f"{cs.name}: Not Ready")
                                
                                status_msg = f"  Pod {pod_name}: Phase={phase}, Ready={ready}"
                                if container_statuses_info:
                                    status_msg += f", Containers=[{', '.join(container_statuses_info)}]"
                                console.log(status_msg)
                        last_log_time = wait
                    
                    if pod_list.items:
                        ready_pods = [
                            pod for pod in pod_list.items
                            if pod.status.container_statuses and
                            all(cs.ready for cs in pod.status.container_statuses)
                        ]

                        if len(ready_pods) == len(pod_list.items):
                            console.log(f"[bold green]All pods in namespace '{namespace}' are ready.")
                            return
                    elif wait >= 60:  # If no pods after 60 seconds, warn
                        console.log(f"[yellow]Warning: No pods found in namespace '{namespace}' after {wait} seconds. This might indicate a deployment issue.")

                except Exception as e:
                    console.log(f"[red]Error checking pod statuses: {e}")

                time.sleep(sleep)
                wait += sleep

            # Final status check before raising exception
            try:
                pod_list = self.list_pods(namespace)
                if not pod_list.items:
                    # Check if deployments/statefulsets exist but pods don't
                    try:
                        deployments = self.apps_v1_api.list_namespaced_deployment(namespace)
                        statefulsets = self.apps_v1_api.list_namespaced_stateful_set(namespace)
                        
                        if deployments.items or statefulsets.items:
                            console.log(f"[red]Deployments/StatefulSets exist but no pods found. Checking status...")
                            for deploy in deployments.items:
                                console.log(f"  Deployment {deploy.metadata.name}:")
                                console.log(f"    Desired replicas: {deploy.spec.replicas}")
                                console.log(f"    Ready replicas: {deploy.status.ready_replicas if deploy.status.ready_replicas else 0}")
                                if deploy.status.conditions:
                                    for condition in deploy.status.conditions:
                                        if condition.status != "True":
                                            console.log(f"    {condition.type}: {condition.status} - {condition.reason}")
                                            if condition.message:
                                                console.log(f"      {condition.message}")
                            for sts in statefulsets.items:
                                console.log(f"  StatefulSet {sts.metadata.name}:")
                                console.log(f"    Desired replicas: {sts.spec.replicas}")
                                console.log(f"    Ready replicas: {sts.status.ready_replicas if sts.status.ready_replicas else 0}")
                    except:
                        pass
                    
                    raise Exception(f"[red]Timeout: No pods found in namespace '{namespace}' after {max_wait} seconds. Deployment may have failed. Check if Kubernetes resources were created successfully.")
                
                # Log final pod statuses for debugging
                console.log(f"[red]Final pod statuses in namespace '{namespace}':")
                for pod in pod_list.items:
                    pod_name = pod.metadata.name
                    phase = pod.status.phase
                    ready_count = 0
                    total_count = 0
                    container_details = []
                    
                    if pod.status.container_statuses:
                        ready_count = sum(1 for cs in pod.status.container_statuses if cs.ready)
                        total_count = len(pod.status.container_statuses)
                        
                        for cs in pod.status.container_statuses:
                            detail = f"{cs.name}: "
                            if cs.state.waiting:
                                detail += f"Waiting ({cs.state.waiting.reason})"
                                if cs.state.waiting.message:
                                    detail += f" - {cs.state.waiting.message}"
                            elif cs.state.terminated:
                                detail += f"Terminated ({cs.state.terminated.reason})"
                                if cs.state.terminated.message:
                                    detail += f" - {cs.state.terminated.message}"
                            elif cs.ready:
                                detail += "Ready"
                            else:
                                detail += "Not Ready"
                            container_details.append(detail)
                    
                    console.log(f"  {pod_name}: Phase={phase}, Ready={ready_count}/{total_count}")
                    for detail in container_details:
                        console.log(f"    {detail}")
            except Exception as e:
                if "No pods found" in str(e):
                    raise e
                pass  # Ignore other errors in final check
            
            raise Exception(f"[red]Timeout: Not all pods in namespace '{namespace}' reached the Ready state within {max_wait} seconds.")
    
    def wait_for_namespace_deletion(self, namespace, sleep=2, max_wait=300):
        """Wait for a namespace to be fully deleted before proceeding."""

        console = Console()
        console.log(f"[bold green]Waiting for namespace '{namespace}' to be deleted...")

        with console.status("[bold green]Waiting for namespace deletion...") as status:
            wait = 0

            while wait < max_wait:
                try:
                    self.core_v1_api.read_namespace(name=namespace)
                except Exception as e:
                    console.log(f"[bold green]Namespace '{namespace}' has been deleted.")
                    return

                time.sleep(sleep)
                wait += sleep

            raise Exception(f"[red]Timeout: Namespace '{namespace}' was not deleted within {max_wait} seconds.")

    def update_deployment(self, name: str, namespace: str, deployment):
        """Update the deployment configuration."""
        return self.apps_v1_api.replace_namespaced_deployment(
            name, namespace, deployment
        )

    def patch_service(self, name, namespace, body):
        """Patch a Kubernetes service in a specified namespace."""
        try:
            api_response = self.core_v1_api.patch_namespaced_service(
                name, namespace, body
            )
            return api_response
        except ApiException as e:
            print(f"Exception when patching service: {e}\n")
            return None

    def create_configmap(self, name, namespace, data):
        """Create or update a configmap from a dictionary of data."""
        try:
            api_response = self.update_configmap(name, namespace, data)
            return api_response
        except ApiException as e:
            if e.status == 404:
                return self.create_new_configmap(name, namespace, data)
            else:
                print(f"Exception when updating configmap: {e}\n")
                print(f"Exception status code: {e.status}\n")
                return None

    def create_new_configmap(self, name, namespace, data):
        """Create a new configmap."""
        config_map = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(name=name),
            data=data,
        )
        try:
            return self.core_v1_api.create_namespaced_config_map(namespace, config_map)
        except ApiException as e:
            print(f"Exception when creating configmap: {e}\n")
            return None

    def create_or_update_configmap(self, name: str, namespace: str, data: dict):
        """Create a configmap if it doesn't exist, or update it if it does."""
        try:
            existing_configmap = self.core_v1_api.read_namespaced_config_map(
                name, namespace
            )
            # ConfigMap exists, update it
            existing_configmap.data = data
            self.core_v1_api.replace_namespaced_config_map(
                name, namespace, existing_configmap
            )
            print(f"ConfigMap '{name}' updated in namespace '{namespace}'")
        except ApiException as e:
            if e.status == 404:
                # ConfigMap doesn't exist, create it
                body = client.V1ConfigMap(
                    metadata=client.V1ObjectMeta(name=name), data=data
                )
                self.core_v1_api.create_namespaced_config_map(namespace, body)
                print(f"ConfigMap '{name}' created in namespace '{namespace}'")
            else:
                print(f"Error creating/updating ConfigMap '{name}': {e}")

    def update_configmap(self, name, namespace, data):
        """Update existing configmap with the provided data."""
        config_map = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(name=name),
            data=data,
        )
        try:
            return self.core_v1_api.replace_namespaced_config_map(
                name, namespace, config_map
            )
        except ApiException as e:
            print(f"Exception when updating configmap: {e}\n")
            return

    def apply_configs(self, namespace: str, config_path: str):
        """Apply Kubernetes configurations from a specified path to a namespace."""
        import os
        from pathlib import Path
        
        console = Console()
        
        # Convert to Path object if it's a string
        if isinstance(config_path, Path):
            path_obj = config_path
        else:
            path_obj = Path(config_path)
        
        # Resolve absolute path
        try:
            path_obj = path_obj.resolve()
        except Exception as e:
            console.log(f"[yellow]Warning: Could not resolve path {config_path}: {e}")
        
        console.log(f"[blue]Checking Kubernetes config path: {path_obj}")
        
        # Check if path exists
        if not path_obj.exists():
            error_msg = f"[red]Error: Kubernetes config path does not exist: {path_obj}\n(Resolved from: {config_path})"
            console.log(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check if path is a directory and has files
        yaml_files = []
        if path_obj.is_dir():
            yaml_files = list(path_obj.rglob("*.yaml")) + list(path_obj.rglob("*.yml"))
            if not yaml_files:
                error_msg = f"[red]Error: No YAML files found in {path_obj}"
                console.log(error_msg)
                raise FileNotFoundError(error_msg)
            else:
                console.log(f"[green]Found {len(yaml_files)} YAML files in {path_obj}")
                # List first few files for debugging
                for yaml_file in yaml_files[:5]:
                    console.log(f"[dim]  - {yaml_file}")
                if len(yaml_files) > 5:
                    console.log(f"[dim]  ... and {len(yaml_files) - 5} more files")
        elif path_obj.is_file():
            console.log(f"[green]Applying single file: {path_obj}")
            yaml_files = [path_obj]
        else:
            error_msg = f"[red]Error: Path is neither a file nor directory: {path_obj}"
            console.log(error_msg)
            raise ValueError(error_msg)
        
        command = f"kubectl apply -Rf {path_obj} -n {namespace}"
        console.log(f"[blue]Executing: {command}")
        
        # Use raise_on_error to get proper exception
        try:
            result = self.exec_command(command, raise_on_error=True)
            
            # Check if result indicates success
            if result:
                result_lower = result.lower()
                # kubectl apply success indicators
                if any(keyword in result_lower for keyword in ["created", "configured", "unchanged"]):
                    console.log(f"[green]Successfully applied configurations from {path_obj}")
                    # Show summary of created resources
                    lines = result.strip().split('\n')
                    created_count = sum(1 for line in lines if 'created' in line.lower())
                    configured_count = sum(1 for line in lines if 'configured' in line.lower())
                    if created_count > 0 or configured_count > 0:
                        console.log(f"[dim]  Created: {created_count}, Configured: {configured_count}")
                else:
                    console.log(f"[yellow]Warning: Unexpected output from kubectl apply:\n{result[:500]}")
            
            return result
            
        except RuntimeError as e:
            console.log(f"[red]Error: kubectl apply failed:\n{e}")
            raise

    def delete_configs(self, namespace: str, config_path: str):
        """Delete Kubernetes configurations from a specified path in a namespace."""
        try:
            exists_resource = self.exec_command(
                f"kubectl get all -n {namespace} -o name"
            )
            if exists_resource:
                print(f"Deleting K8S configs in namespace: {namespace}")
                command = (
                    f"kubectl delete -Rf {config_path} -n {namespace} --timeout=10s"
                )
                self.exec_command(command)
            else:
                print(f"No resources found in: {namespace}. Skipping deletion.")
        except subprocess.CalledProcessError as e:
            print(f"Error deleting K8S configs: {e}")
            print(f"Command output: {e.output}")

    def delete_namespace(self, namespace: str):
        """Delete a specified namespace."""
        try:
            self.core_v1_api.delete_namespace(name=namespace)
            self.wait_for_namespace_deletion(namespace)
            print(f"Namespace '{namespace}' deleted successfully.")
        except ApiException as e:
            if e.status == 404:
                print(f"Namespace '{namespace}' not found.")
            else:
                print(f"Error deleting namespace '{namespace}': {e}")

    def create_namespace_if_not_exist(self, namespace: str):
        """Create a namespace if it doesn't exist."""
        try:
            self.core_v1_api.read_namespace(name=namespace)
            print(f"Namespace '{namespace}' already exists.")
        except ApiException as e:
            if e.status == 404:
                print(f"Namespace '{namespace}' not found. Creating namespace.")
                body = client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace))
                self.core_v1_api.create_namespace(body=body)
                print(f"Namespace '{namespace}' created successfully.")
            else:
                print(f"Error checking/creating namespace '{namespace}': {e}")

    def exec_command(self, command: str, input_data=None, raise_on_error=False):
        """Execute an arbitrary kubectl command.
        
        Args:
            command: Command to execute
            input_data: Input data for the command
            raise_on_error: If True, raise exception on error instead of returning stderr
        
        Returns:
            stdout if successful, stderr if failed (unless raise_on_error=True)
        """
        import os
        import shutil
        
        # Ensure kubectl is in PATH and get full path
        kubectl_path = shutil.which("kubectl")
        if not kubectl_path:
            raise RuntimeError("kubectl not found in PATH. Please ensure kubectl is installed and in your PATH.")
        
        # Preserve environment variables, especially KUBECONFIG
        env = os.environ.copy()
        
        if input_data is not None:
            input_data = input_data.encode("utf-8")
        
        try:
            out = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                input=input_data,
                env=env,
                timeout=300  # 5 minute timeout
            )
            return out.stdout.decode("utf-8")
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode("utf-8") if e.stderr else e.stdout.decode("utf-8") if e.stdout else str(e)
            if raise_on_error:
                raise RuntimeError(f"Command failed: {command}\nError: {error_output}")
            return error_output
        except subprocess.TimeoutExpired:
            if raise_on_error:
                raise RuntimeError(f"Command timed out after 300 seconds: {command}")
            return "Command timed out"

    def get_node_architectures(self):
        """Return a set of CPU architectures from all nodes in the cluster."""
        architectures = set()
        try:
            nodes = self.core_v1_api.list_node()
            for node in nodes.items:
                arch = node.status.node_info.architecture
                architectures.add(arch)
        except ApiException as e:
            print(f"Exception when retrieving node architectures: {e}\n")
        return architectures

# Example usage:
if __name__ == "__main__":
    kubectl = KubeCtl()
    namespace = "test-social-network"
    frontend_service = "nginx-thrift"
    user_service = "user-service"

    user_service_pod = kubectl.get_pod_name(namespace, f"app={user_service}")
    logs = kubectl.get_pod_logs(user_service_pod, namespace)
    print(logs)
