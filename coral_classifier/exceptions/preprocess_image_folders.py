class NoneValueException(ValueError):
    def __init__(self, error_message):
        self.error_message = error_message
