from coral_classifier.exceptions.preprocess_image_folders import NoneValueException


class Guard:
    @staticmethod
    def against_none(item, item_name):
        if item is None:
            raise NoneValueException(f"Value of '{item_name}' cannot be None.")
