class PeriodicActionTrigger:
    @staticmethod
    def should_trigger(step: int, period: int) -> bool:
        if period <= 0:
            return False
        return step % period == 0
