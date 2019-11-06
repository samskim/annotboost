# AnnotBoost
Kim et al. "Improving the informativeness of Mendelian disease pathogenicity scores for common diseases and complex traits" biorxiv 2019.

Annotations available at https://data.broadinstitute.org/alkesgroup/LDSCORE/Kim_annotboost/

data: 4 SNP sets (GWAS significant SNPs, Farh et al. fine-mapped SNPs, Weissbrod et al. fine-mapped SNPs, and de novo SNPs); these SNPs are restricted to 9,997,231 SNPs (MAF >= 0.5%) found in the reference panel (1000 Genomes Phase 3 EUR). 

model: pretrained models for 82 scores after appyling AnnotBoost to odd/even choromosomes. Load the model to see training parameters used by XGBoost; parameters were optimized with two-fold cross validation (while avoiding over-fitting).

scripts: AnnotBoost source codes
 - S-LDSC / meta-analysis scripts were previously published on https://github.com/samskim/networkconnectivity 
  
shapley: SHAP (SHapley Additive exPlanations) feature importance script + results: see more on https://github.com/slundberg/shap

scores (available on https://data.broadinstitute.org/alkesgroup/LDSCORE/Kim_annotboost/ due to the file size constraint here): summarized published and boosted scores for 86 baseline-LD annotations + 35 published scores + 82 boosted scores (corresponding to 35 boosted published scores + 47 boosted baseline-LD scores). columns: Y1, Y2, Y3, Y4 corresponds to labels for 4 SNP sets (Y1: GWAS, Y2: Farh et al., Y3: Weissbrod et al., Y4: De novo SNPs).

File formats:
scores: probabilstic scores from 0 to 1
annotations: binary annotation based on top X% of the scores (download at https://data.broadinstitute.org/alkesgroup/LDSCORE/Kim_annotboost/)
 - annotation format:
    - annot: thin-annot version is used (one column ANNOT)
    - ldscore.gz: CHR, SNP, BP, L2 (delimiter = '\t')

For other details including processing pathogenicity scores, please read the manuscript.
