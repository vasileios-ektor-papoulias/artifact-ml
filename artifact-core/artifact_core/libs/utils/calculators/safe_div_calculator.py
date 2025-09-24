class SafeDivCalculator:
    @staticmethod
    def compute(num: float, denom: float) -> float:
        return 0.0 if denom == 0 else float(num) / float(denom)
