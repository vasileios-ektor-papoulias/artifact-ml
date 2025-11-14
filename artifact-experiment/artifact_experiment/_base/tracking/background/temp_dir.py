from artifact_experiment._utils.filesystem.temp_dir import ManagedTempDir


class TrackingTempDir(ManagedTempDir):
    _name_ = "artifact_ml_tracking_"

    def __init__(self):
        super().__init__(name=self._name_)
