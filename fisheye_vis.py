import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cbook, cm
import numpy as np

def sq(x):
    return x ** 2
sqrt = np.sqrt
atan2 = np.arctan2

x, y = np.meshgrid(np.linspace(-2, 2, 500), np.linspace(-2, 2, 500))
z = np.ones_like(x) * 0
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

eps = 0.01
# f0 = atan2(sqrt(sq(x) + sq(y)), z) / sqrt(sq(x) + sq(y))
# f =  atan2(sqrt(sq(x) + sq(y)) + eps, z) / (sqrt(sq(x) + sq(y)) + eps)
# f0 = sq(x) * z / (sq(x) + sq(y)) / (sq(x) + sq(y) + sq(z)) + sq(y) / (sq(x) + sq(y)) * atan2(sqrt(sq(x) + sq(y)), z) / sqrt(sq(x) + sq(y))
# f = sq(x) * z / (sq(x) + sq(y)) / (sq(x) + sq(y) + sq(z)) + sq(y) / (sq(x) + sq(y) + eps) * atan2(sqrt(sq(x) + sq(y) + eps), z) / sqrt(sq(x) + sq(y) + eps)

# f = sq(x) * z / (sq(x) + sq(y)) / (sq(x) + sq(y) + sq(z)) + sq(y) / (sq(x) + sq(y)) * atan2(sqrt(sq(x) + sq(y)), z) / sqrt(sq(x) + sq(y))
f = x * y * z / (sq(x) + sq(y)) / (sq(x) + sq(y) + sq(z)) + x * y / (sq(x) + sq(y)) * atan2(sqrt(sq(x) + sq(y)), z) / sqrt(sq(x) + sq(y))
# f = sq(x) * z / (sq(x) + sq(y)) / (sq(x) + sq(y) + sq(z)) + sq(y) / (sq(x) + sq(y)) * atan2(sqrt(sq(x) + sq(y)), z) / sqrt(sq(x) + sq(y))
# f = - x / (sq(x) + sq(y) + sq(z))
# f = - x / (sq(x) + sq(y) + sq(z))
df = f 

plt.imshow(df, extent=[x_min,x_max,y_min,y_max])
plt.show()
