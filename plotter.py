import matplotlib.pyplot as plt
import pickle

data = {
    'dense': 'dump_dense.pkl',
    'small': 'dump_disc_small.pkl',
    'full': 'dump_disc_full.pkl',
    'laplace': 'dump_disc_laplace.pkl',
    'weird': 'dump_disc_weird.pkl',
    'weird_full': 'dump_disc_weird_full.pkl'
}

hist_data = {}
for key in data.keys():
    try:
        with open(data[key], 'r') as f:
            tr, ts = pickle.load(f)
            hist_data[key] = {'train': tr, 'test': ts}
    except IOError:
        continue

ax1 = plt.subplot(2, 2, 1)

for key in hist_data.keys():
    ax1.plot(hist_data[key]['train']['acc'], label=key)
ax1.set_ylabel('Train accucary')
ax1.legend()

ax2 = plt.subplot(2, 2, 2)
for key in hist_data.keys():
    ax2.plot(hist_data[key]['train']['loss'], label=key)
ax2.set_ylabel('Train loss')
ax2.legend()

ax3 = plt.subplot(2, 2, 3)
for key in hist_data.keys():
    ax3.plot(hist_data[key]['test']['acc'], label=key)
ax3.set_ylabel('Val accucary')
ax3.legend()

ax4 = plt.subplot(2, 2, 4)
for key in hist_data.keys():
    ax4.plot(hist_data[key]['test']['loss'], label=key)
ax4.set_ylabel('Val loss')
ax4.legend()

plt.show()