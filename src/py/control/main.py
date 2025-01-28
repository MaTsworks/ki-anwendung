from py.control.model_type import ModelType
from robomaster_vision_control import RoboMasterVisionControl

def main():

    model_type = _get_user_model_choice()
    control = RoboMasterVisionControl(model_type)
    control.run()


def _get_user_model_choice() -> ModelType:
    while True:
        choice = input(
            "Which model would you like to use?\n1. Pistol Detection\n2. COCO Dataset (80 classes)\nEnter 1 or 2: ")
        if choice == "1":
            return ModelType.PISTOL
        elif choice == "2":
            return ModelType.COCO
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()

