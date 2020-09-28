# AnnotBoost
Kim et al. "Improving the informativeness of Mendelian disease pathogenicity scores for common diseases and complex traits" biorxiv 2020.
First, download neceesary files: "wget https://storage.googleapis.com/broad-alkesgroup-public/LDSCORE/Kim_annotboost/annotboost.tar.gz"

Input pathogenicity scores, boosted scores, and binarized annotations available at https://alkesgroup.broadinstitute.org/LDSCORE/Kim_annotboost/
 - summarized published and boosted scores for 86 baseline-LD annotations + 35 published scores + 82 boosted scores (corresponding to 35 boosted published scores + 47 boosted baseline-LD scores). 
 - annotation format:
    - annot: thin-annot version is used (one column ANNOT)
    - ldscore.gz: CHR, SNP, BP, L2 (delimiter = '\t')

S-LDSC / meta-analysis scripts were previously published on https://github.com/samskim/networkconnectivity 

AnnotBoost framework (see the demo: AnnotBoost_demo.ipynb):
- required file: https://storage.googleapis.com/broad-alkesgroup-public/LDSCORE/Kim_annotboost/annotboost.tar.gz
- Input: .csv.gz file for containing variant scores (formatted with columns = ['CHR', 'BP', 'REF', 'ALT', 'SCORE']. For all our analysis, SNP set is 9,997,231 SNPs with MAF >= 0.5% in 1000G EUR Phase3 referene panel; see the example input file format in AnnotBoost_examplefiles.ipynb)
- Output: .csv.gz file for boosted variant scores, cPickle model file, SHAP feature importance 
- Computational time, memory required: (while it varies on how many iterations for parameter tuning), time for training is <12 hours with ~10GB memory. For missense variant scores where only 0.3% of the genome is scored, training is significantly faster.
- Python 3.X version is suggested to be used. 

Application for AnnotBoost:
Background: pathogenicity scores are instrumental in prioritizing variants for Mnedelian disease. We explore their application to 41 complex traits. 
1. To assess the informativeness of a given variant pathogenicity score on common disease (polygenic common and low-frequency variant architecture)
2. To improve an existing variant pathogenicity score for common disease 
 - Better understanding disease heritability via functional components of heritability suggests the application in functionally-informed fine-mapping. We further demonstrated the these new annotations improve the heritability model fit, quantified by loglss (scripts to run SumHer to compute loglss is included in above URL as well). 

For other details including processing pathogenicity scores, please read the manuscript.
All source codes distributed under the MIT license.
