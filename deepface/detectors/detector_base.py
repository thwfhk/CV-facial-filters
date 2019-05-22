import abc


class FaceDetector(object):
    def __init__(self):
        pass

    def __str__(self):
        return self.name()

    @abc.abstractmethod
    def name(self):
        return 'detector'

    @abc.abstractmethod
    def detect(self, npimg):
        """

        :param npimg:
        :return: list of BoundingBox
        """
        pass
