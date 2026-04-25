# #################################################################################################################
# DATASET BASE CLASS
class Dataset:
    def __init__(self, name, class_column, random_state=0):
        """
        Dataset initializer - This is the base class of all dataset subclasses.

        Args:
            name: The name of the dataset
            random_state: Controls random number generation. Set this to a fixed integer to get reproducible results.
        """
        self._name = name
        self._random_state = random_state

        self.class_column = class_column
        self.num_classes = 0
