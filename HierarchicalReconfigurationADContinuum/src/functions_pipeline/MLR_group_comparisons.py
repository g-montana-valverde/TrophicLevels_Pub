import tqdm as tqdm
from statsmodels.formula.api import ols
from statsmodels.stats import multitest

def run_comparisons(df, demographics, g1, g2):
	full_labels = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'ID']
	
	pvals=[]; tvals=[];
	for col_label in tqdm(full_labels):	
		MLR_df = df[df['Group'].isin([g1, g2])]
    MLR_df = MLR_df.merge(demographics[['ID', 'age', 'gender', 'edu']], on='ID', how='left')
    MLR_df['group'] = MLR_df['Group'].map({g1: 0, g2: 1}).astype(int)
    MLR_df['measure']=MLR_df[col_label]
  
    model = ols('measure ~ C(group) + age + C(gender) + edu', data=MLR_df).fit()
    pvals.append(model.pvalues['C(group)[T.1]'])
    tvals.append(model.tvalues['C(group)[T.1]'])

	_, pvals_corr= np.array(multitest.fdrcorrection(pvals))
	
  corr_values = np.where(pvals_corr < 0.05, tvals, 0)
	
	return corr_values, pvals_corr
