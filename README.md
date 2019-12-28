# HealthGAN
Part of Advanced Project

# Task 1:
Aim:
Create Bigram and N-grams of non-zero padded medical data. 

Input:
Given an input containing Dataframe and value of n(in case of n-grams). The dataframe is assumed to contain data for each patient and all the visits. Each visit contain "c" features. As 3d dataframes are not possible, I have taken features as a list converted into string, thus making it a 2d dataframe.  

Output:
Output contains 3 values. First is the number of n-grams, second contains value of n(in case of n-grams) and the third contains the list of n-grams.

Function Explanation:
create_dummy_df: This function is used to create dummy dataframe data. It needs input as number of patients, max visits of patients and number of features of each patient. The dummy dataframe randomly sets values for some visits. If the number of visit for a particular patient is less than the max visit number, than the data is padded with zeroes.

generate_ngrams: This function creates the n-grams. It needs input of numpy array and value of n.

gen_ngrams: This is the main function that takes input the dataframe and the the value of n and outputs the requored 3 values as mentioned above.
