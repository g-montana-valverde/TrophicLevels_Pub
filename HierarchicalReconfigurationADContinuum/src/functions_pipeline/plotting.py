import matplotlib.pyplot as plt
import ptitprince as pt
from matplotlib.ticker import MaxNLocator

def significance_marker(p):
	if p < 0.001: return '***'
	elif p < 0.01: return '**'
	elif p < 0.05: return '*'
	else: return 'n.s.'
    
def plot_directedness(df, pvals, out_path):
	df_copy = df.copy()
	df_copy['Group'] = df_copy['Group'].str.replace('p', '+').str.replace('n', '-', regex=False)

	groups=df_copy['Group'].unique()
	
	plt.figure(figsize=(8,7))
	pt.RainCloud(x='Group', y='directedness', data=df_copy, bw=0.3, dodge = True,
		palette = sns.color_palette("colorblind", n_colors=len(groups)), width_viol=0.5, width_box=0.3, box_showfliers=False, move = 0.0, point_size=7)

	plt.xticks(fontsize=40)
	plt.yticks(fontsize=38)
	plt.xlabel('Group', fontsize=32), plt.ylabel('');
	
	y_max = df_copy['directedness'].max()
	y_min = df_copy['directedness'].min()

	# generate all pairwise index combinations
	n = len(groups)
	pairs = []
	n = len(groups)
	for delta in range(1, n):
		for i in range(0, n - delta):
			pairs.append((i, i+delta))

	y_offset = (y_max-y_min)/17
	y = y_max + y_offset
	for idx, (i, j) in enumerate(pairs):
		if pvals[idx][0]<0.05:
			plt.plot([i+0.1, j-0.1], [y, y], color='black', lw=1.5)
			plt.text((i + j) / 2, y-y*0.03, significance_marker(pvals[idx][0]), 
					ha='center', va='bottom', fontsize=40)
			if idx >= n - 2:
				y += 2*y_offset  # increase height for next bar
	
	# Adjust y limits to fit all bars
	ymin, y_max = plt.ylim()
	plt.ylim(ymin, y_max+y_offset)


	plt.title('Directedness', fontsize=40)
	plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=5))

	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()
