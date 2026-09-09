"""Graded doses of the Otel demo paymentServiceFailure feature flag.

The flag ships severity variants (10%/25%/50%/75%/90%/100%) that upstream
never exposes -- ``OtelFaultInjector`` hardcoded ``"100%"``. Registering
each variant as its own problem turns "how quiet is this failure" from a
subjective label into an axis whose ground truth is the injected value.

The ``off`` variant is a SHAM CONTROL, not a no-op. It performs exactly
the same ConfigMap write and ``kubectl rollout restart deployment flagd``
as every other dose, and differs only in having no behavioural effect.

That matters more than it looks. ``NoopFaultInjector`` does literally
nothing (``inject_no_op`` is ``pass``), so ``noop_detection_astronomy_shop-1``
never restarts flagd. Calibrating a null against it would leave the
rollout's pod churn uncancelled, and every real dose would then look
explicit in the event channel purely because flagd was replaced --
a perfectly shaped dose-response step that is entirely an artifact of the
injection tool. The sham arm puts that churn into the null so it cancels.
"""

from typing import Any

from aiopslab.orchestrator.tasks import *
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.apps.astronomy_shop import AstronomyShop
from aiopslab.generators.fault.inject_otel import OtelFaultInjector
from aiopslab.session import SessionItem

#: Variants offered by demo.flagd.json for paymentFailure. Validated
#: again at injection time against the live ConfigMap, since the chart
#: comes from a remote Helm repo and can change underneath us.
DOSE_VARIANTS = ("off", "10%", "25%", "50%", "75%", "90%", "100%")

#: The sham control: injected through the identical code path, zero effect.
SHAM_VARIANT = "off"


class PaymentFailureDoseBaseTask:
    def __init__(self, variant: str):
        if variant not in DOSE_VARIANTS:
            raise ValueError(
                f"variant must be one of {DOSE_VARIANTS}, got {variant!r}"
            )
        self.variant = variant
        self.app = AstronomyShop()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.injector = OtelFaultInjector(namespace=self.namespace)
        self.faulty_service = "payment"

    @property
    def is_sham(self) -> bool:
        return self.variant == SHAM_VARIANT

    def start_workload(self):
        print("== Start Workload ==")
        print("Workload skipped since AstronomyShop has a built-in load generator.")

    def inject_fault(self):
        print(f"== Fault Injection (dose={self.variant}) ==")
        if self.is_sham:
            print("Sham control: same ConfigMap write and flagd rollout, no effect.")
        self.injector.inject_fault("paymentFailure", variant=self.variant)
        print(f"Fault: paymentFailure@{self.variant} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        self.injector.recover_fault("paymentFailure")


################## Detection Problem ##################
class PaymentFailureDoseDetection(PaymentFailureDoseBaseTask, DetectionTask):
    def __init__(self, variant: str):
        PaymentFailureDoseBaseTask.__init__(self, variant)
        DetectionTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        # The sham arm is a genuine negative: nothing is wrong with the
        # application, so "No" is correct there and "Yes" is a false
        # positive. That false-positive rate is what makes the detection
        # rates at the low doses interpretable at all.
        expected_solution = "No" if self.is_sham else "Yes"

        if isinstance(soln, str):
            if soln.strip().lower() == expected_solution.lower():
                print(f"Correct detection: {soln}")
                self.add_result("Detection Accuracy", "Correct")
            else:
                print(f"Incorrect detection: {soln} (expected {expected_solution})")
                self.add_result("Detection Accuracy", "Incorrect")
        else:
            print("Invalid solution format")
            self.add_result("Detection Accuracy", "Invalid Format")

        # Ride the experimental condition into the session JSON so the
        # analysis never has to reconstruct it from the problem id.
        self.add_result("dose_variant", self.variant)
        self.add_result("is_sham", self.is_sham)
        self.add_result("expected", expected_solution)

        return super().eval(soln, trace, duration)


################## Localization Problem ##################
class PaymentFailureDoseLocalization(PaymentFailureDoseBaseTask, LocalizationTask):
    def __init__(self, variant: str):
        if variant == SHAM_VARIANT:
            # There is no faulty service to name when nothing is broken.
            raise ValueError("localization is not defined for the sham control")
        PaymentFailureDoseBaseTask.__init__(self, variant)
        LocalizationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")

        if soln is None:
            print("Solution is None")
            self.add_result("Localization Accuracy", 0.0)
            self.results["success"] = False
            self.results["is_subset"] = False
            self.add_result("dose_variant", self.variant)
            return super().eval(soln, trace, duration)

        is_exact = is_exact_match(soln, self.faulty_service)
        is_sub = is_subset([self.faulty_service], soln)

        if is_exact:
            self.add_result("Localization Accuracy", 100.0)
        elif is_sub:
            self.add_result("Localization Accuracy", (1 / len(soln)) * 100.0)
        else:
            self.add_result("Localization Accuracy", 0.0)

        self.results["success"] = is_exact or (is_sub and len(soln) == 1)
        self.results["is_subset"] = is_sub
        self.add_result("dose_variant", self.variant)

        return super().eval(soln, trace, duration)
