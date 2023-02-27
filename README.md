# corr_comm_detection
Multilayer community detection with covariance matrix input

## corr_comm_detection.py
Contains the following functions
* cov_to_corr: function to transform a covariance matrix into a correlation matrix

* con_corr_func: function to generate configuration model correlation matrix from empirical correlation or covariance matrix, using configcorr package

* multicorrcat: function (inspired by multicat.m) to output a flattened modularity matrix

* it_genlouvain_corr_consensus: function to run iterated GenLouvain, a community detection algorithm, on the flattened modularity matrix

* corr_partition_info: function to obtain information about the partition needed for significance calculations

* corr_intra_z: function to calculate significance (Z score for total intralayer weight) of each community in the partition

* main: main function that uses all the above functions to perform community detection with covariance matrix/matrices input and outputs the partition and the Z score for each community detected

## Python package dependencies: 
corr_comm_detection has the following dependencies. These packages need to be installed manually.
* numpy
* pandas
* matplotlib
* scipy
* configcorr, found here: https://github.com/naokimas/config_corr
* netneurotools, found here: https://github.com/netneurolab/netneurotools
* matlab.engine, instructions found here: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
* Also must download the GenLouvain matlab package, found here: https://github.com/GenLouvain/GenLouvain

## multilayer_gene_coexpression_example.ipynb 
Contains example usage of corr_comm_detection with gene coexpression data for four different organs from GTEx

exo_emp_covs.npy and organ_corr_omegas.npy contain data for this example usage.
