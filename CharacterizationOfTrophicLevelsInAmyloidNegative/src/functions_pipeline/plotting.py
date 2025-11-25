def plot_correlation_with_mean(x, y, x_mean, y_mean, x_label, y_label, out_path):
	
	x_array=x.flatten()
	y_array=y.flatten()
	
	x_array = zscore(x_array)
	y_array = zscore(y_array)
	x_mean = zscore(x_mean)
	y_mean = zscore(y_mean)

	r, p = pearsonr(x_array,y_array)
	
	if p < 0.001:
		p_str = "***"
	elif p < 0.01:
		p_str = "**"
	elif p < 0.05:
		p_str = "*"
	else:
		p_str = "n.s."
	title = f"r{p_str} = {r:.2f}"
	plt.figure(figsize=(8.5, 6))
	plt.scatter(x_array, y_array, alpha=0.4, s=100, color="#229AC2")
	plt.scatter(x_mean, y_mean, c=y_mean, cmap='jet', s=200)
	
	# Fit line using seaborn regplot
	ax=sns.regplot(x=x_array, y=y_array, scatter=False, line_kws={'color':'red', 'linewidth':4})
	plt.xlabel(f"{x_label} (z)", fontsize=38)
	plt.ylabel(f"{y_label} (z)", fontsize=38)
	plt.title(title, fontsize=40)
	plt.xticks(fontsize=38)
	plt.yticks(fontsize=38)
	ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
	ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
	plt.axhline(0, color='gray', linestyle='--', linewidth=1)
	plt.axvline(0, color='gray', linestyle='--', linewidth=1)
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()
	
def radar_plot_net(zTL_sfn, lower_threshold, upper_threshold, out_path):
	angles = np.linspace(0, 2 * np.pi, len(SFN_names), endpoint=False).tolist()
	angles += angles[:1]  # close the circle

	fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
	zTL_sfn = np.concatenate([zTL_sfn, zTL_sfn[:1]])
	base_vals = zTL_sfn[:-1]

	# colormap mapping using min/max of tl_mean as requested
	norm = Normalize(vmin=np.min(zTL_sfn), vmax=np.max(zTL_sfn))
	cmap = cm.get_cmap('jet')

	# line and fill color based on mean network value
	representative_val = np.mean(base_vals)
	line_color = cmap(norm(representative_val))
	ax.plot(angles, zTL_sfn, linewidth=3, color='black', alpha=0.6)

	ax.scatter(angles[:-1], base_vals, c=cmap(norm(base_vals)), s=300, edgecolors='k', zorder=10)

	theta_circle = np.linspace(0, 2 * np.pi, 400)
	r_lower = np.ones_like(theta_circle) * lower_threshold
	r_upper = np.ones_like(theta_circle) * upper_threshold

	ax.plot(theta_circle, r_lower, color='red', linestyle='--', linewidth=5.5, zorder=20)
	ax.text(np.pi/2, lower_threshold, f'{lower_threshold:.2f}', ha='center', va='bottom', fontsize=45, color='red', zorder=25)

	ax.plot(theta_circle, r_upper, color='purple', linestyle='--', linewidth=5.5, zorder=20)
	ax.text(np.pi/2, upper_threshold, f'{upper_threshold:.2f}', ha='center', va='bottom', fontsize=45, color='purple', zorder=25)

	ax.set_axisbelow(False)
	ax.set_xticks(angles[:-1])
	ax.set_xticklabels(SFN_names, fontsize=45)
	ax.tick_params(axis='x', pad=25)

	min_val = np.min(base_vals)
	max_val = np.max(base_vals)
	pad_min = 0.8 * abs(min_val)
	pad_max = 0.1 * abs(max_val)
	ax.set_ylim(min_val - pad_min, max_val + pad_max)
	ax.yaxis.set_visible(False)
	ax.grid(False)

	ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
	ax.tick_params(axis='y', labelsize=40)

	# fill radial background with 'jet' mapped from (min_val - pad_min) to (max_val + pad_max)
	theta = np.linspace(0, 2 * np.pi, 400)
	r = np.linspace(min_val - pad_min, max_val + pad_max, 200)
	Theta, R = np.meshgrid(theta, r)
	Z = R  # use radius as the value to map to colormap

	bg_norm = Normalize(vmin=np.min(r), vmax=np.max(r))
	ax.pcolormesh(Theta, R, Z, cmap=cm.get_cmap('jet'), norm=bg_norm, shading='auto', zorder=0, alpha=0.7)
	plt.tight_layout()
	plt.savefig(out_path)
