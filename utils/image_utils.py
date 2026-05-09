import cv2
import numpy as np
import math


def convert_to_display(image):
    """
    Convert image to displayable format for Streamlit.
    """
    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class StepLogger:
    def __init__(self):
        self.steps = []

    def add_step(self, title, image):
        self.steps.append((title, image.copy()))

    def get_steps(self):
        return self.steps