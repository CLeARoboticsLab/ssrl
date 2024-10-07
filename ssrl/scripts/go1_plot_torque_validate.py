import dill
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''


save_path = '/home/jl79444/dev/d4po/saved_policies/model_validation/hardware_data/pred_and_real_forces.pkl'

with open(save_path, 'rb') as f:
    (time, preds, reals) = dill.load(f)

idxs = [[0, 1, 2], [6, 7, 8]]
labels = [['Base X', 'Base Y', 'Base Z'], ['Front-right Hip', 'Front-right Thigh', 'Front-right Calf']]
start = 300
stop = 400

fig, axs = plt.subplots(len(idxs), len(idxs[0]), figsize=(12, 7))

for i, idx in enumerate(idxs):
    for j, k in enumerate(idx):
        axs[i, j].plot(time[:stop-start], reals[start:stop, k], label='Actual')
        axs[i, j].plot(time[:stop-start], preds[start:stop, k], label='Predicted')
        axs[i, j].set_title(labels[i][j])
        # axs[i, j].legend()
        if i == len(idxs) - 1:
            axs[i, j].set_xlabel('Time (s)')

axs[0, 0].set_ylabel('Force (N)')
axs[1, 0].set_ylabel('Torque (N-m)')
axs[0, 0].legend(loc='upper left')

# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=2, fontsize='large')

plt.tight_layout()
plt.show()
plt.savefig('/home/jl79444/dev/d4po/scripts/forces.png')
plt.savefig('/home/jl79444/dev/d4po/scripts/forces.pdf', format='pdf', bbox_inches='tight')
