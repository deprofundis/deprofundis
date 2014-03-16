class Blub(object):

    shared_v = None

    def __init__(self, value):
        self.v = value

    @property
    def v(self):
        return Blub.shared_v

    @v.setter
    def v(self, value):
        print "Setter called"
        Blub.shared_v = value

    @v.deleter
    def v(self):
        del Blub.shared_v


c = Blub(5)
print str(c.v)
print