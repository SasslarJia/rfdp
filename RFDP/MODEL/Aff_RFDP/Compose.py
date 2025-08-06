#
class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, xfrms: list):
        self.xfrms = xfrms

    def __call__(self, mat):
        for xfrm in self.xfrms:
            mat = xfrm(mat)
        return mat

    def __repr__(self):
        #
        fmat_str = self.__class__.__name__ + '('
        #
        for t in self.xfrms:
            fmat_str += '\n'
            fmat_str += '    {0}'.format(t)
        #
        fmat_str += '\n)'
        return fmat_str
