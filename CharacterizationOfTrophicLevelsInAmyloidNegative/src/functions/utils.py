
SFN_names = ['VN', 'SMN', 'DAN', 'SAN', 'LN', 'CN', 'DMN']

# R2SFN: Region-2-SchaeferFunctionalNetworks
def R2SFN(df):
	
	non_region_cols = [col for col in df.columns if col not in region_labels]
	SFN_df = df[non_region_cols].copy()
	
	for sfn in SFN_names:
		
		regions_in_sfn = [region for region, sfn_map in region_to_SFN.items() if sfn in (sfn_map or "")]
		
		exclude_keywords = ['hippocampus', 'amygdala', 'thalamus', 'caudate', 'accumbens', 'putamen', 'gpe', 'gpi', 'stn'] 
		
		regions_in_sfn = [region for region in regions_in_sfn if not any(keyword in region.lower() for keyword in exclude_keywords)]

		regions_in_sfn = [region for region in regions_in_sfn if region in df.columns]

		if regions_in_sfn:
			SFN_df[sfn] = df[regions_in_sfn].mean(axis=1)
		else:
			SFN_df[sfn] = np.nan
	
	return SFN_df
