# Explanation


The code here contains the evaluation of Mean Absolute Error and other Metrics between the correlation data of Original data and Synthetic data.

Original Data: OD
Synthetic Data: SD

Considering the data contains Y columns of medication data and X columns of diagnostic data. First, we are finding the corrlation matrix between X * Y.
Then we are selecting the top 100 correlations between those columns from OD. 
The same 100 correlations are calculated for SD.
Then we calculate the Mean Absolute Error between those data points and plot them.

**Preprocessing:**

Check if the given data is 3d or 2d and create Dataframe accordingly.

Check for duplicate columns from OD and remove them.

Remove columns containing all zeros in OD and remove those columns from SD too.
Then remove zero columns from SD and remove those columns from OD too.

**Calculations:**

We are using Tensorflow to find Correlation between each columns. [tfp.stats.correlation()]
And using sklearn metric to calculate Mean absolute error.

**Files Explanation**

main.py is used to start the program. We can set the name of header file, OD file and SD file here. It also contains the code to calculate Absolute Mean Difference, other metrics and plot graph.

data_utils.py contains all other functionalities like preprocessing, creating a DataFrame and calculating Correlation.

distance_calculator.py contains other meteric calculation. (Commented out for now!!!)

**Future Work:**

We can use these distance metrics too.
Frobenium Norm
L2 Norm
Wasserstein Distance
Frechet Inception Distance
Students T-test
Kolmogorov-Smir
Shapiro-Wilk Test
Anderson-Darling Test



