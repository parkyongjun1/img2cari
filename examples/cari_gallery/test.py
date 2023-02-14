import numpy as np

with open('cari11.txt', 'w') as f:
    for i in range(0,5151):
        a = '{0:06d}.png\n'.format(i)
        f.write(a)