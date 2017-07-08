class Line:
    def __init__(self, x1, y1, x2, y2):
        # set start and end points
        self.p1 = (x1, y1)
        self.p2 = (x2, y2)

        # set length
        self.length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        # compute line angle
        dx = x2 - x1
        dy = y2 - y1
        rads = math.atan2(-dy, dx)
        rads %= 2 * math.pi
        degs = -math.degrees(rads)
        if degs <= -180:
            degs += 180
        degs += 90
        self.angle = degs

        # set midpoint
        self.midpoint = (((x1 + x2) / 2), ((y1 + y2) / 2))

    def __str__(self):
        return str(self.angle)

    def __repr__(self):
        return self.__str__()

    def get_coords(self):
        return self.p1, self.p2
