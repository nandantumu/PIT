import math

from pit.dynamics.dynamic_bicycle import DynamicBicycle

class VoronoiClassifier:
    def __init__(self, points):
        """
        Initialize the VoronoiClassifier with a set of labeled points.
        
        :param points: List of tuples (x, y, l) where x and y are coordinates, and l is the label.
        """
        self.points = points

    def classify(self, x, y):
        """
        Classify a point (x, y) based on the closest labeled point.
        
        :param x: X coordinate of the point to classify.
        :param y: Y coordinate of the point to classify.
        :return: Label of the closest point.
        """
        closest_point = None
        min_distance = float('inf')

        for px, py, label in self.points:
            distance = math.sqrt((px - x) ** 2 + (py - y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_point = label

        return closest_point
    



class DynamicsWrapper:
    def __init__(self, parameters, dynamics_model_class):
        """
        Initialize the DynamicsWrapper with sets of parameters and a dynamics model class.

        :param parameters: Dictionary where keys are labels and values are parameter sets.
        :param dynamics_model_class: The class of the dynamics model to use.
        """
        self.parameters = parameters
        self.dynamics_model_class = dynamics_model_class
        self.current_model = None
        self.current_params = None

    def select_parameters(self, label):
        """
        Select a set of parameters based on the label and initialize the dynamics model.

        :param label: The label for the parameter set to use.
        """
        if label in self.parameters:
            self.current_params = self.parameters[label]
            self.current_model = self.dynamics_model_class(**self.current_params)
        else:
            raise ValueError(f"No parameter set found for label {label}")

    def get_current_model(self):
        """
        Get the current dynamics model.

        :return: The current dynamics model.
        """
        return self.current_model

    def hot_swap_parameters(self, new_label):
        """
        Hot-swap the parameters and reinitialize the dynamics model.

        :param new_label: The new label for the parameter set to switch to.
        """
        self.select_parameters(new_label)

# Example usage:
if __name__ == "__main__":
    points = [(1, 2, 0), (3, 4, 1), (5, 6, 2)]
    classifier = VoronoiClassifier(points)
    
    print(classifier.classify(2, 3))  # Output will be the label of the closest point

    # Define parameter sets for different labels
    parameters = {
        0: {"param1": 5, "param2": 6},
        1: {"param1": 7, "param2": 8},
        # Add more parameter sets as needed
    }

    # Initialize the wrapper with the parameter sets and the dynamics model class
    wrapper = DynamicsWrapper(parameters, DynamicBicycle)

    # Select parameters based on a label
    wrapper.select_parameters(0)
    current_model = wrapper.get_current_model()

    # Hot-swap parameters to a new set
    wrapper.hot_swap_parameters(1)
    new_model = wrapper.get_current_model()
