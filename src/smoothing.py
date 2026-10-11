"""Per-slot evidence smoothing with hysteresis."""


class SlotStateMachine:
    """Smooth binary slot observations independently with evidence hysteresis.

    Evidence follows `e <- e + alpha * (obs - e)`. State changes at `on` or
    `off`; one update represents one inference, so response time depends on the
    inference rate rather than the display frame rate.
    """

    def __init__(self, n_slots: int, alpha: float = 0.3, on: float = 0.65, off: float = 0.35):
        if n_slots < 0:
            raise ValueError("n_slots must be non-negative")
        if not 0 < alpha <= 1 or not 0 <= off < on <= 1:
            raise ValueError("expected alpha in (0,1] and 0 <= off < on <= 1")
        self.alpha, self.on, self.off = alpha, on, off
        self.evidence = [0.0] * n_slots
        self.states = [0] * n_slots
        self._initialized = [False] * n_slots

    def update(self, observations: list[int | bool]) -> list[int]:
        if len(observations) != len(self.states):
            raise ValueError("observation count must equal the slot count")
        for index, observation in enumerate(observations):
            obs = float(bool(observation))
            if not self._initialized[index]:
                self.evidence[index] = obs
                self.states[index] = int(bool(observation))
                self._initialized[index] = True
                continue
            evidence = self.evidence[index] + self.alpha * (obs - self.evidence[index])
            self.evidence[index] = evidence
            if evidence >= self.on:
                self.states[index] = 1
            elif evidence <= self.off:
                self.states[index] = 0
        return self.states.copy()
