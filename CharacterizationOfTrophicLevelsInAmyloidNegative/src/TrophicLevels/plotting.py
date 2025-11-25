MTX = nib.load(util_dir + 'dbs80_2mm.nii.gz').get_fdata()

def plot_subcort(array, vmin, vmax, measure_name, out_dir):

	if not os.path.exists(out_dir): os.makedirs(out_dir)

	slide=59
	regs_subcort = np.arange(32, 50)

	subcort_mtx=np.zeros_like(MTX, dtype=float)
	for subcort_reg in regs_subcort:
		subcort_mtx[MTX==subcort_reg]=array[subcort_reg-1]
	
	if np.sum(subcort_mtx[:, slide,:])==0: return
	atlas_slide=MTX[:, slide,:]
	subcort_slide=subcort_mtx[:, slide,:]
	atlas_mask = (atlas_slide == 0) | (~np.isin(atlas_slide, regs_subcort))
	atlas_slide = np.ma.masked_array(atlas_slide, mask=atlas_mask)
	subcort_slide = np.ma.masked_array(subcort_slide, mask=(subcort_slide == 0))
	
	myMap='jet'

	plt.figure(figsize=(10, 10))
	plt.imshow(atlas_slide.T, cmap=ListedColormap(['lightgray']), origin='lower', zorder=0, alpha=1)
	plt.imshow(subcort_slide.T, cmap=myMap, origin='lower', alpha=1, vmin=vmin, vmax=vmax, zorder=1)

	plt.axis('off')
	
	x_center = 45  
	y_center = 35   
	x_width = 35    
	y_height = 30   

	plt.xlim(x_center - x_width / 2, x_center + x_width / 2)
	plt.ylim(y_center - y_height / 2, y_center + y_height / 2)
	plt.gca().invert_xaxis()
	plt.tight_layout()
	plt.savefig(out_dir + measure_name + '_subcort.png', dpi=150, transparent=True)
	plt.close('all')

def radar_plot_net(zTL_sfn, lower_threshold, upper_threshold):
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
	plt.savefig(work_dir + 'Radar_Networks.png', dpi=300)
