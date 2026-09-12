import json
from aiopslab.generators.fault.base import FaultInjector
from aiopslab.service.kubectl import KubeCtl


class ConfigMapMissing(Exception):
    """The flagd ConfigMap is not there -- usually the namespace is gone."""


class OtelFaultInjector(FaultInjector):
    #: Variant used when the caller does not name one. Preserves the
    #: previous hardcoded behaviour so existing problems are unaffected.
    DEFAULT_VARIANTS = {"paymentFailure": "100%", "imageSlowLoad": "10sec"}

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.kubectl = KubeCtl()
        self.configmap_name = "flagd-config"

    def _read_flags(self) -> dict:
        """Fetch and parse the flagd ConfigMap.

        ``exec_command`` returns *stderr as an ordinary string* when kubectl
        fails, so the original ``except subprocess.CalledProcessError`` here
        could never fire: a missing ConfigMap fell through to the JSON decode
        and was reported as "Error decoding JSON", naming the wrong cause.
        ``raise_on_error=True`` makes the two cases distinguishable.
        """
        command = (
            f"kubectl get configmap {self.configmap_name} -n {self.namespace} -o json"
        )
        try:
            output = self.kubectl.exec_command(command, raise_on_error=True)
        except RuntimeError as exc:
            raise ConfigMapMissing(
                f"ConfigMap '{self.configmap_name}' not readable in namespace "
                f"'{self.namespace}': {exc}"
            ) from exc
        try:
            configmap = json.loads(output)
            return json.loads(configmap["data"]["demo.flagd.json"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                f"ConfigMap '{self.configmap_name}' is present but its "
                f"demo.flagd.json is unreadable: {exc}"
            ) from exc

    def inject_fault(self, feature_flag: str, variant: str | None = None):
        """Enable a flagd feature flag.

        ``variant`` selects the severity where the flag offers a choice --
        ``paymentFailure`` ships 10%/25%/50%/75%/90%/100% and
        ``imageSlowLoad`` ships 5sec/10sec. Omitting it keeps the previous
        behaviour (the most severe variant), so existing callers are
        unaffected.

        The variant is validated against the ``variants`` map in the live
        ConfigMap rather than against an assumed ladder. The chart is
        pulled from a remote Helm repo and its variants have changed
        across versions; without this check an unrecognised name would be
        written to the ConfigMap and flagd would quietly serve the
        default, producing a no-op that still gets scored as a fault.
        """
        flagd_data = self._read_flags()

        if feature_flag not in flagd_data["flags"]:
            raise ValueError(
                f"Feature flag '{feature_flag}' not found in ConfigMap '{self.configmap_name}'."
            )

        chosen = variant or self.DEFAULT_VARIANTS.get(feature_flag, "on")
        available = flagd_data["flags"][feature_flag].get("variants", {})
        if chosen not in available:
            raise ValueError(
                f"Variant '{chosen}' is not defined for feature flag "
                f"'{feature_flag}' in ConfigMap '{self.configmap_name}'. "
                f"Available: {sorted(available)}"
            )
        flagd_data["flags"][feature_flag]["defaultVariant"] = chosen

        updated_data = {"demo.flagd.json": json.dumps(flagd_data, indent=2)}
        self.kubectl.create_or_update_configmap(
            self.configmap_name, self.namespace, updated_data
        )

        self.kubectl.exec_command(
            f"kubectl rollout restart deployment flagd -n {self.namespace}"
        )

        print(f"Fault injected: Feature flag '{feature_flag}' set to '{chosen}'.")

    def recover_fault(self, feature_flag: str):
        """Turn the flag off. A no-op once the namespace is gone.

        The orchestrator registers an ``atexit`` hook that recovers the fault,
        and the campaign's own teardown has usually already uninstalled the
        release by then. Raising here turns an ordinary clean exit into a
        traceback that looks like a failed recovery.
        """
        try:
            flagd_data = self._read_flags()
        except ConfigMapMissing:
            print(
                f"Fault recovery skipped: no '{self.configmap_name}' in "
                f"namespace '{self.namespace}' (already torn down)."
            )
            return

        if feature_flag in flagd_data["flags"]:
            flagd_data["flags"][feature_flag]["defaultVariant"] = "off"
        else:
            raise ValueError(
                f"Feature flag '{feature_flag}' not found in ConfigMap '{self.configmap_name}'."
            )

        updated_data = {"demo.flagd.json": json.dumps(flagd_data, indent=2)}
        self.kubectl.create_or_update_configmap(
            self.configmap_name, self.namespace, updated_data
        )

        self.kubectl.exec_command(
            f"kubectl rollout restart deployment flagd -n {self.namespace}"
        )
        print(f"Fault recovered: Feature flag '{feature_flag}' set to 'off'.")


# Example usage:
# if __name__ == "__main__":
#     namespace = "astronomy-shop"
#     feature_flag = "adServiceFailure"

#     injector = OtelFaultInjector(namespace)

#     injector.inject_fault(feature_flag)
#     injector.recover_fault(feature_flag)
