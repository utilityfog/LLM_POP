class TransformExceptionWrapper(Exception):
    def __init__(self, exception: Exception = None):
        self.exception = exception
        self._processed = 0
        self._skipped = 0

    def __str__(self) -> str:
        return f"TransformExceptionWrapper {self.exception}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def skipped(self):
        return self._skipped

    @skipped.setter
    def skipped(self, value):
        self._skipped = value

    @property
    def processed(self):
        return self._processed

    @processed.setter
    def processed(self, value):
        self._processed = value


class CollateExceptionWrapper(Exception):
    def __init__(self, exception: Exception):
        self.exception = exception
        self._processed = 0
        self._skipped = 0

    def __str__(self) -> str:
        return f"CollateExceptionWrapper {self.exception}"

    def __repr__(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return str(self)

    @property
    def skipped(self):
        return self._skipped

    @skipped.setter
    def skipped(self, value):
        self._skipped = value

    @property
    def processed(self):
        return self._processed

    @processed.setter
    def processed(self, value):
        self._processed = value


class StopChildProcess:
    pass
