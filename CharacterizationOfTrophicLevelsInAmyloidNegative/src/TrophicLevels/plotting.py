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
