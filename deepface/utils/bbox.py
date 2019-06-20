class BoundingBox:
    __slots__ = ['x', 'y', 'w', 'h', 'score', 'face_name', 'face_score', 'face_feature', 'face_landmark', 'face_roi']

    def __init__(self, x=0, y=0, w=0, h=0, score=0.0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score

        self.face_name = ''
        self.face_score = 0.0
        self.face_feature = None
        self.face_landmark = None
        self.face_roi = None

    def __repr__(self):
        return '%.2f %.2f %.2f %.2f score=%.2f name=%s' % (self.x, self.y, self.w, self.h, self.score, self.face_name)
