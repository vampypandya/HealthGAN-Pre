# Explanation


The code here contains the evaluation of Mean Absolute Error and other Metrics between the correlation data of Original data and Synthetic data but only for Specific diseases.

CCS Data: CD
Original Data: OD
Synthetic Data: SD

We are finding here one disease VS rest.

Considering the data contains Y columns of medication data and X columns of diagnostic data. First, we are finding the corrlation matrix between X * Y.
Then we are selecting the top 20 correlations between those columns from OD. 
The same 20 correlations are calculated for SD.
Then we calculate the Mean Absolute Error between those data points and plot them.

**Preprocessing:**

Check if the given data is 3d or 2d and create Dataframe accordingly.

Get relevant data from CD using the CCS codes given. In this case, following were the CCS codes: ['50'] vs ['108', '210', '157', '159']

Use the indexes of the rows of above data, find similar rows in OD. Now, this is the OD. 

Check for duplicate columns from OD and remove them.

Remove columns containing all zeros in OD and remove those columns from SD too.
Then remove zero columns from SD and remove those columns from OD too.

**Calculations:**

We are using Tensorflow to find Correlation between each columns. [tfp.stats.correlation()]
And using sklearn metric to calculate Mean absolute error.

**Files Explanation**

main.py is used to start the program. We can set the name of disease name, header file, OD file and SD file here. It also contains the code to calculate Absolute Mean Difference, other metrics, preprocessing, creating a DataFrame, calculating Correlation and plotting graph.

data_gen.py It is used for calculating the needed rows from CD and finding the same in OD.

Error for Single Disease VS Synthetic Data: 0.21992789160162024
Error for Rest Diseases VS Synthetic Data: 0.23252348233463457




