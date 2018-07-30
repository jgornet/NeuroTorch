from numbers import Number


class Vector:
    def __init__(self, *components):
        self.setComponents(components)

    def setComponents(self, components):
        if not all(isinstance(x, Number) for x in components):
            raise ValueError("components must contain all numbers instead" +
                             " it contains {}".format(components))
        self.components = tuple(components)

    def getComponents(self):
        return self.components

    def getDimension(self):
        return len(self.getComponents())

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise ValueError("other must be a vector instead"
                             " it is ".format(type(other)))

        if self.getDimension() != other.getDimension():
            raise ValueError("other must have the same dimension instead "
                             + "self is {} and other is {}".format(self,
                                                                   other))

        result = [s + o for s, o in zip(self.getComponents(),
                                        other.getComponents())]

        return Vector(*result)

    def __mul__(self, other):
        if isinstance(other, Number):
            result = [s * other for s in self.getComponents()]

        elif isinstance(other, Vector):
            result = [s*o for s, o in zip(self.getComponents(),
                                          other.getComponents())]

        else:
            raise ValueError("other must be a number or a vector instead"
                             " it is {}".format(type(other)))

        return Vector(*result)

    def __div__(self, other):
        if isinstance(other, Number):
            return self*(1/other)

        if isinstance(other, Vector):
            result = Vector([1/o for o in other.getComponents()])
            result *= self

            return result

    def __sub__(self, other):
        return self+(other*-1)

    def __eq__(self, other):
        if not isinstance(other, Vector):
            raise ValueError("other must be a vector instead"
                             " it is ".format(type(other)))

        if self.getDimension() != other.getDimension():
            raise ValueError("other must have the same dimension")

        return all(r1 == r2 for r1, r2 in zip(self.getComponents(),
                                              other.getComponents()))

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "{}".format(self.getComponents())

    def __iter__(self):
        return iter(self.getComponents())

    def __getitem__(self, idx):
        if idx < self.getDimension():
            return self.getComponents()[idx]
        else:
            raise IndexError("the index is out of bounds")


class BoundingBox:
    def __init__(self, edge1: Vector, edge2: Vector):
        self.setEdges(edge1, edge2)

    def setEdges(self, edge1: Vector, edge2: Vector):
        if not isinstance(edge1, Vector) or not isinstance(edge2, Vector):
            raise ValueError("edges must be vectors")

        if edge1.getDimension() != edge2.getDimension():
            raise ValueError("edges must have the same dimension instead"
                             + " edge 1 is {} and edge 2 is {}".format(edge1,
                                                                       edge2))

        self.edge1 = edge1
        self.edge2 = edge2

    def getEdges(self):
        return (self.edge1, self.edge2)

    def getDimension(self):
        return self.getEdges()[0].getDimension()

    def getSize(self):
        edge1, edge2 = self.getEdges()
        chunk_size = edge2 - edge1

        return chunk_size

    def isDisjoint(self, other):
        if not isinstance(other, BoundingBox):
            raise ValueError("other must be a vector instead other is "
                             "{}".format(type(other)))

        result = any(r1 > r2 for r1, r2 in zip(self.getEdges()[0],
                                               other.getEdges()[1]))
        result |= any(r1 < r2 for r1, r2 in zip(self.getEdges()[1],
                                                other.getEdges()[0]))

        return result

    def isSubset(self, other):
        if not isinstance(other, BoundingBox):
            raise ValueError("other must be a vector instead other is "
                             "{}".format(type(other)))

        result = any(r1 <= r2 for r1, r2 in zip(self.getEdges()[1],
                                                other.getEdges()[1]))
        result &= any(r1 >= r2 for r1, r2 in zip(self.getEdges()[0],
                                                 other.getEdges()[0]))

        return result

    def isSuperset(self, other):
        return other.isSubset(self)

    def intersect(self, other):
        if self.isDisjoint(other):
            raise ValueError("The bounding boxes must not be disjoint")

        edge1 = Vector(*map(lambda x, y: max(x, y),
                            self.getEdges()[0].getComponents(),
                            other.getEdges()[0].getComponents()))
        edge2 = Vector(*map(lambda x, y: min(x, y),
                            self.getEdges()[1].getComponents(),
                            other.getEdges()[1].getComponents()))

        return BoundingBox(edge1, edge2)

    def __str__(self):
        edge1, edge2 = self.getEdges()
        return "Edge1: {}\nEdge2: {}".format(edge1, edge2)

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise ValueError("other must be a vector instead other is "
                             + "{}".format(type(other)))

        edge1, edge2 = self.getEdges()
        result = BoundingBox(edge1 + other, edge2 + other)
        return result
