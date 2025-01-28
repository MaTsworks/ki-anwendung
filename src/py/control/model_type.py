from enum import Enum


class ModelType(Enum):
    PISTOL = "pistol"
    COCO = "coco"

    def __eq__(self, other):
        if isinstance(other, ModelType):
            return self.value == other.value
        return False