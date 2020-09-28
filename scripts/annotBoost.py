#AnnotBoost: Kim et al. Nature Communications 2020
#some libraries are not used; may delete those unused in this script
import pandas as pd
import numpy as np
import glob
import os
import sys
import xgboost #install xgboost: https://xgboost.readthedocs.io/en/latest/index.html#
import sklearn
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st
import pickle 
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap #install shap: https://github.com/slundberg/shap 
#from xgboost.sklearn import XGBClassifier

#Python version: v.3.6.10 used. 

features = ['Coding_UCSC', 'Coding_UCSC.extend.500',
       'Conserved_LindbladToh', 'Conserved_LindbladToh.extend.500',
       'CTCF_Hoffman', 'CTCF_Hoffman.extend.500', 'DGF_ENCODE',
       'DGF_ENCODE.extend.500', 'DHS_peaks_Trynka', 'DHS_Trynka',
       'DHS_Trynka.extend.500', 'Enhancer_Andersson',
       'Enhancer_Andersson.extend.500', 'Enhancer_Hoffman',
       'Enhancer_Hoffman.extend.500', 'FetalDHS_Trynka',
       'FetalDHS_Trynka.extend.500', 'H3K27ac_Hnisz',
       'H3K27ac_Hnisz.extend.500', 'H3K27ac_PGC2', 'H3K27ac_PGC2.extend.500',
       'H3K4me1_peaks_Trynka', 'H3K4me1_Trynka', 'H3K4me1_Trynka.extend.500',
       'H3K4me3_peaks_Trynka', 'H3K4me3_Trynka', 'H3K4me3_Trynka.extend.500',
       'H3K9ac_peaks_Trynka', 'H3K9ac_Trynka', 'H3K9ac_Trynka.extend.500',
       'Intron_UCSC', 'Intron_UCSC.extend.500', 'PromoterFlanking_Hoffman',
       'PromoterFlanking_Hoffman.extend.500', 'Promoter_UCSC',
       'Promoter_UCSC.extend.500', 'Repressed_Hoffman',
       'Repressed_Hoffman.extend.500', 'SuperEnhancer_Hnisz',
       'SuperEnhancer_Hnisz.extend.500', 'TFBS_ENCODE',
       'TFBS_ENCODE.extend.500', 'Transcr_Hoffman',
       'Transcr_Hoffman.extend.500', 'TSS_Hoffman', 'TSS_Hoffman.extend.500',
       'UTR_3_UCSC', 'UTR_3_UCSC.extend.500', 'UTR_5_UCSC',
       'UTR_5_UCSC.extend.500', 'WeakEnhancer_Hoffman',
       'WeakEnhancer_Hoffman.extend.500', 'GERP.NS', 'GERP.RSsup4', 'MAFbin1',
       'MAFbin2', 'MAFbin3', 'MAFbin4', 'MAFbin5', 'MAFbin6', 'MAFbin7',
       'MAFbin8', 'MAFbin9', 'MAFbin10', 'MAF_Adj_Predicted_Allele_Age',
       'MAF_Adj_LLD_AFR', 'Recomb_Rate_10kb', 'Nucleotide_Diversity_10kb',
       'Backgrd_Selection_Stat', 'CpG_Content_50kb', 'MAF_Adj_ASMC',
       'GTEx_eQTL_MaxCPP', 'BLUEPRINT_H3K27acQTL_MaxCPP',
       'BLUEPRINT_H3K4me1QTL_MaxCPP', 'BLUEPRINT_DNA_methylation_MaxCPP',
       'synonymous', 'non_synonymous', 'Conserved_Vertebrate_phastCons46way',
       'Conserved_Vertebrate_phastCons46way.extend.500',
       'Conserved_Mammal_phastCons46way',
       'Conserved_Mammal_phastCons46way.extend.500',
       'Conserved_Primate_phastCons46way',
       'Conserved_Primate_phastCons46way.extend.500', 'BivFlnk',
       'BivFlnk.extend.500'] #85 features from baseline-LD (v.2.1)
       
'''
if wanting to use latest baseline-LD as features:
run make_baselineLD_feature with latest baseline-LD model annotations
merge annotations from chr 1 to chr22
see baselineLD_allchr.annot.gz (v.2.1 used)
in the manuscript, MAF bins were not used as features; either including or not makes no significant differences. 
'CHR', 'BP', 'SNP', 'CM' are information columns for SNP.
'base' are all SNPs, thus not used as a feature.
'''
def make_baselineLD_feature():
	i = 1
	df = pd.read_csv("baselineLD.%s.annot.gz" %i, sep="\t") #download from https://alkesgroup.broadinstitute.org/LDSCORE/
	for i in range(2,23):
		df2 = pd.read_csv("baselineLD.%s.annot.gz" %i, sep="\t")
		df = pd.concat([df,df2])
	df.to_csv("%s/reference_files/baselineLD_allchr.annot.gz" %args.basedir,sep="\t",compression="gzip",index=False) #used as features
	return

'''
input: input_patho_score full directory
Need two columns: "SNP" and "SCORE"
If you have genomic regions, not SNP, merge with CADD.csv.gz (provided) that have 9997231 1000G common+low-freq SNPs.
see the example: /annotboost/AnnotBoost_sourcecode/reference_files_demo/input/CADD.csv.gz
SNP: RSID, and Y are pathogenicity scores (higher more pathogenic)
if the original score indicates lower more pathogenic, reverse the score (e.g. 1/original score)
'''
def prepare_data(input_patho_score, evenodd):
	df = pd.read_csv("%s/baselineLD_allchr.annot.gz" %args.basedir,sep="\t")
	all = df.drop(["BP", "CM", "base"],axis=1)
	del df
	score = pd.read_csv("%s/%s" %(args.basedir,input_patho_score)) 
	rf = pd.read_csv("%s/1000G_final.vcf.gz" %args.basedir, names=["CHR","BP","SNP","REF","ALT"])
	score = score.merge(rf, on=["CHR","BP","REF","ALT"]) #if window-based, remove REF, ALT here
	all["Y"] = -1
	all.loc[all.SNP.isin(score[score.SCORE >= score.SCORE.quantile(0.9)].SNP.to_list()), "Y"] = 1 #top 10% indicate as Y = 1
	all.loc[all.SNP.isin(score[score.SCORE < score.SCORE.quantile(0.4)].SNP.to_list()), "Y"] = 0 #bottom 40% indicate as Y = 0
	all = all[all.Y != -1] #drop SNPs with no labels
	#all.to_csv("%s/training_matrix_%s.csv.gz" %(args.basedir,args.published_score_filename), index=False, compression='gzip') #SNP + CHR + features + 'Y'
	if evenodd == 'even': #using even chromosome data
		all = all[all.CHR % 2 == 0]
	else: #odd chr
		all = all[all.CHR % 2 == 1]
	Y_train = all.Y.values
	X_train = all[features].values
	del all
	return X_train, Y_train 
	
'''          
if computational time is limited, change n_iter to 5 or less.
these parameters are what has been used in the manuscript.
Optimize as needed, based on your data: suggest not changing gamma and learning rate.
'''
def fit_model(x_train,y_train,n_iter=10,cv=5):
    n_estimators = [200, 250, 300]
    max_depth = [25, 30, 35]
    learning_rate = [0.05] #0.05 fixed to avoid over-fitting
    gamma = [10]
    one_to_left = st.beta(10, 1)
    min_child_weight = [6, 8, 10]
    from_zero_positive = st.expon(0, 50)
    nthread = [1]
    scale_pos_weight = [1, 2, 3, 4, 5]
    subsample = [0.6, 0.8, 1]
    random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
                'learning_rate':learning_rate,
                "colsample_bytree": one_to_left,
                "subsample": subsample,
                'reg_alpha': from_zero_positive,
                "min_child_weight": min_child_weight,
                'gamma':gamma,
                'nthread':nthread,
                'scale_pos_weight':scale_pos_weight}
    xgb_tune = xgboost.XGBClassifier(missing=np.nan)
    xgb_hyper = RandomizedSearchCV(xgb_tune,random_grid, n_iter = n_iter, cv = cv, random_state=42, n_jobs = -1,scoring='roc_auc') 
    xgb_hyper.fit(x_train, y_train)
    return xgb_hyper.best_estimator_ #return the best model based on scoring=roc_auc

#make predictions, ROC/PR curve plot
def test_model(xgb_mod,x_test,y_test):
    predictions = xgb_mod.predict_proba(x_test)[:,1]
    ROC_curve(predictions,y_test)
    PR_curve(predictions,y_test) 
    return predictions
   
#plot ROC curve 
#y_pred is the probabilistic score after applying AnnotBoost
#y_test is binarized label used in the training
def ROC_curve(y_pred, y_test):
    tilte = "ROC curve"
    label = "baseline-LD features"
    y = np.asarray(y_test).reshape((len(y_test),1))
    y_hat = np.asarray(y_pred).reshape((len(y_pred),1))
    fpr, tpr, _ = roc_curve(y,y_hat)
    AUROC = auc(fpr,tpr)
    plt.plot(fpr,tpr,label=label + ' (AUC = %.3f)' % AUROC)
    plt.legend(loc='lower right')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.savefig('%s/ROC_CURVE_%s.png' %(args.basedir, args.published_score_filename))
    plt.close()
    return

#plot PR curve
#y_pred is the probabilistic score after applying AnnotBoost
#y_test is binarized label used in the training
def PR_curve(y_pred, y_test):
    tilte = "PR curve"
    label = "baseline-LD features"
    y = np.asarray(y_test).reshape((len(y_test),1))
    y_hat = np.asarray(y_pred).reshape((len(y_pred),1))
    precision, recall, thresholds = precision_recall_curve(y, y_hat)
    AUPR = average_precision_score(y,y_hat)
    plt.plot(recall,precision,label=label + ' (AP = %.3f)' % AUPR, color='r')
    plt.legend(loc='lower left')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('%s/PR_CURVE_%s.png' %(args.basedir, args.published_score_filename))
    plt.close()
    return

def SHAP_feature_importance(model, X, Y, evenodd, approximate=False):
	explainer = shap.TreeExplainer(model)		
	shap_values = explainer.shap_values(X, Y, approximate)
	shap.summary_plot(shap_values, X, plot_type="bar", show=False)                                                                  
	plt.savefig("%s/SHAP_featurerank_%s_%s.png" %(args.basedir, args.published_score_filename, evenodd), dpi=300, bbox_inches = "tight")
	plt.clf()  
	plt.close()
	shap.summary_plot(shap_values, X, show=False)
	plt.savefig("%s/SHAP_signedsummary_%s_%s.png" %(args.basedir, args.published_score_filename, evenodd), dpi=300, bbox_inches = "tight")
	plt.clf()
	plt.close()  
	return
	
'''
make annotation files (.annot.gz) from genome-wide prediction or published scores
these annotations are used in S-LDSC analysis
top 10%-top0.1% by default (top X% can be optimized based on S-LDSC tau*)
annotations use 1000G EUR phase3 as a reference panel. If other reference panel is desired, use your template annot.gz files.
S-LDSC script (annotations as input) is provided in https://github.com/samskim/networkconnectivity
'''
def prediction_to_annot(result_csv, outfilename, SCORE):
	tf = pd.read_csv("%s/RESULT_%s" %(args.basedir, result_csv)) #assume that annotations are created after running AnnotBoost and predictions are made
	tf0 = tf[(tf[SCORE] >= tf[SCORE].quantile(0.90))] #top 10% SNPs
	tf1 = tf[(tf[SCORE] >= tf[SCORE].quantile(0.95))]
	tf2 = tf[(tf[SCORE] >= tf[SCORE].quantile(0.99))]
	tf3 = tf[(tf[SCORE] >= tf[SCORE].quantile(0.995))]
	tf4 = tf[(tf[SCORE] >= tf[SCORE].quantile(0.999))] #top 0.1% SNPs
	if SCORE == 'SCORE':
		pubbos = "published"
	else:
		pubbos = "boosted"
	for j in range(1,23):
			df = pd.read_csv("%s/example_annotations/example.%s.annot.gz" %(args.basedir, j), sep="\t")
			df = df.drop("SAM_ANNOT",axis=1)
			df["ANNOT"] = 0
			df.loc[df.SNP.isin(tf0.SNP), ['ANNOT']] = 1
			af = df[["ANNOT"]]
			af.to_csv("%s/%s_%s.perc90.%s.annot.gz" %(args.basedir,outfilename,pubbos,j), compression='gzip', index=False)
			df["ANNOT"] = 0
			df.loc[df.SNP.isin(tf1.SNP), ['ANNOT']] = 1
			af = df[["ANNOT"]]
			af.to_csv("%s/%s_%s.perc95.%s.annot.gz" %(args.basedir,outfilename,pubbos,j), compression='gzip', index=False)
			df["ANNOT"] = 0
			df.loc[df.SNP.isin(tf2.SNP), ['ANNOT']] = 1
			af = df[["ANNOT"]]
			af.to_csv("%s/%s_%s.perc99.%s.annot.gz" %(args.basedir,outfilename,pubbos,j), compression='gzip', index=False)
			df["ANNOT"] = 0
			df.loc[df.SNP.isin(tf3.SNP), ['ANNOT']] = 1
			af = df[["ANNOT"]]
			af.to_csv("%s/%s_%s.perc995.%s.annot.gz" %(args.basedir,outfilename,pubbos,j), compression='gzip', index=False)
			df["ANNOT"] = 0
			df.loc[df.SNP.isin(tf4.SNP), ['ANNOT']] = 1
			af = df[["ANNOT"]]
			af.to_csv("%s/%s_%s.perc999.%s.annot.gz" %(args.basedir,outfilename,pubbos,j), compression='gzip', index=False)
	return
		
def main(args):
    if args.mode == 'makeannot': #make annotations from published/boosted scores
    	prediction_to_annot(args.published_score_filename, "demo", "SCORE") #check the file name follows the convention used in this script
    	prediction_to_annot(args.published_score_filename, "demo", "Boosted SCORE") #make a boosted score to annotations
    if args.mode == 'training':
    	print("Training in progress using even chromosome SNPs!")
    	x_train, y_train = prepare_data(args.published_score_filename, "even")
    	if args.debug == 'T':
    		x_train = x_train[0:1000]
    		y_train = y_train[0:1000]
        np.random.seed(1)
        #####
        print("reading input score and appending with the feature matrix completed!")
        xgb_best = fit_model(x_train,y_train)
        print("training completed!")
        print ("writing the model")
		with open("%s/%s:even.pickle" %(args.basedir, args.published_score_filename),'wb') as fp:
			pickle.dump(xgb_best,fp)
			print ("xgboost: training model written in /%s" %args.basedir)
		X_df = pd.DataFrame(x_train)
		X_df.columns = features
		SHAP_feature_importance(xgb_best, X_df, y_train, "even",  approximate=False) #measure unbiased feature importance using SHAP
		print ("SHAP feature importance written!")
		#####
		print("Training in progress using odd chromosome SNPs!")
		x_train, y_train = prepare_data(args.published_score_filename, "odd")
		if args.debug == 'T':
    		x_train = x_train[0:1000]
    		y_train = y_train[0:1000]
        print("reading input score and appending with the feature matrix completed!")
        xgb_best = fit_model(x_train,y_train)
        print("training completed!")
        print ("writing the model")
		with open("%s/%s:odd.pickle" %(args.basedir, args.published_score_filename),'wb') as fp:
			pickle.dump(xgb_best,fp)
			print ("xgboost: training model written in /%s" %args.basedir)
		X_df = pd.DataFrame(x_train)
		X_df.columns = features
		SHAP_feature_importance(xgb_best, X_df, y_train, "odd",  approximate=False) 
		print ("SHAP feature importance written!")
		print ("Training completed!")
    if args.mode == 'applying':    	
        with open("%s/%s:even.pickle" %(args.basedir, args.published_score_filename),'rb') as fp: 
            xgb_even = pickle.load(fp)
        with open("%s/%s:odd.pickle" %(args.basedir, args.published_score_filename),'rb') as fp: 
            xgb_odd = pickle.load(fp)
        print("reading even/odd chr trained model!...applying the model on the other chr SNPs")
        all_df = pd.read_csv("%s/baselineLD_allchr.annot.gz" %args.basedir,sep="\t")
        all_df = all_df.drop(["CM", "base"],axis=1)
        score = pd.read_csv("%s/%s" %(args.basedir, args.published_score_filename))
		rf = pd.read_csv("%s/1000G_final.vcf.gz" %args.basedir, names=["CHR","BP","SNP","REF","ALT"])
		score = score.merge(rf, on=["CHR","BP","REF","ALT"]) #if window-based, remove REF, ALT here
		score = score[["SCORE","SNP"]]
		all_df = all_df.merge(score, how='left')
		all_df["SCORE"] = all_df["SCORE"].fillna(0) #for ROC/PR curve purposes
		all_df["Y"] = -1
		all_df.loc[all_df.SNP.isin(score[score.SCORE >= score.SCORE.quantile(0.9)].SNP.to_list()), "Y"] = 1 #top 10% indicate as Y = 1
		all_df.loc[all_df.SNP.isin(score[score.SCORE < score.SCORE.quantile(0.4)].SNP.to_list()), "Y"] = 0 #bottom 40% indicate as Y = 0
        all_df_even = all_df[all_df.CHR % 2 == 0]
        all_df_odd = all_df[all_df.CHR % 2 == 1]
        print("applying even chr model on odd chr SNPs!")
        odd_predicted = test_model(xgb_even, all_df_odd[features].values, all_df_odd["Y"].values)
        print("applying odd chr model on even chr SNPs!")
        even_predicted = test_model(xgb_odd, all_df_even[features].values, all_df_even["Y"].values)
		all_df_odd["Boosted SCORE"] = odd_predicted
		all_df_even["Boosted SCORE"] = even_predicted
		all = all_df_odd.append(all_df_even)
		all = all.drop(["Y"],axis=1)
		print("predictions completed!") #final result df includes: 'CHR','BP','SNP', 85 features, SCORE (original input), Boosted SCORE
		all.to_csv("%s/RESULTwithfeatures_%s.csv.gz" %(args.basedir, args.published_score_filename.split(".csv.gz")[0]), compression='gzip', index=False)
		all = all[["SNP","CHR","BP","SCORE","Boosted SCORE"]]	
		all.to_csv("%s/RESULT_%s.csv.gz" %(args.basedir, args.published_score_filename.split(".csv.gz")[0]), compression='gzip', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default='training') #training, applying, makeannot (in this order)
    parser.add_argument('--basedir',type=str,default='/n/groups/price/sam/tier2/sam/annotboost_upload/annotboost/AnnotBoost_sourcecode/reference_files_demo/reference_files')
    parser.add_argument('--published_score_filename',type=str,default='CADD_input.csv.gz')
    parser.add_argument('--debug',type=str,default='F') #'T' if want to check if there's any issue by using 1000 SNP samples
    args = parser.parse_args()
    main(args)
