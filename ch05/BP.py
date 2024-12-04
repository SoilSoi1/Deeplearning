

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout
        dy = dout

        return dx, dy

class Relu:
    '''
> x = np.array( [[1.0, -0.5], [-2.0, 3.0]] )
> print(x)
[[ 1. -0.5]
[-2.  3. ]]
> mask = (x <= 0)
> print(mask)
[[False True]
[ True False]]
    '''
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx