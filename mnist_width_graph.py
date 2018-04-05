import subprocess
from matplotlib import pyplot as plt
import numpy as np

widths = 2*np.arange(20)+1
accs = []
for w in widths:
	out = subprocess.check_output(['python', 'mnist_width.py', str(w)])
	acc = float(out.strip().split('\n')[-1])
	print w, acc
	accs.append(acc)

plt.semilogy(widths,accs)
plt.show()
