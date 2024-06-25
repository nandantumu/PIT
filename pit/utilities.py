import math

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

# Example usage:
if __name__ == "__main__":
    points = [(1, 2, 0), (3, 4, 1), (5, 6, 2)]
    classifier = VoronoiClassifier(points)
    
    print(classifier.classify(2, 3))  # Output will be the label of the closest point
