# 472-assignment1 Instructions

https://github.com/rosiestr/472-assignment1.git

### Part 1: Dataset Preparation and Analysis
Step 1: Open the Jupyter notebook called "1.Prep_and_Analysis.ipynb"

Step 2: Run the first code block to import all the required libraries

Step 3: In the second code block, replace the file directory with the full path on your computer to the goemotions.json.gz dataset

Step 4: Run the second block of code to load the dataset

Step 5: Run the third code block to display the pie chart of the emotions

Step 6:Run the fourth code block to display the pie chart of the sentiments



### Part 2: Dataset Preparation and Analysis
*There are two seperate notebooks for each classification (emotions, sentiment)*

Step 1: Open the Jupyter notebook called "2_words_as_features_EMOTION.ipynb" or "2_words_as_features_SENTIMENT.ipynb"

Step 2: In the first code block, replace the file directory with the full path on your computer to the goemotions.json.gz dataset

Step 3: Run the first code block to important necessary libraries and load the dataset

Step 4: Run the second block of code to load the dataset, which creates the countvectorizer and fits the data

Step 5: Run the third code block to get the training data.

Step 6:Run the fourth code block to train and test the Base-MNB model with our data, and display the score.

Step 7:Run the fifth code block to train and test the Base-DT model with our data, and display the score.

Step 8:Run the sixth code block to train and test the Base-MLP model with our data, and display the score.

Step 9:Run the seventh code block to train and test the Top-MNB model with our data, and display the score.

Step 10:Run the eigth code block to train and test the Top-DT model with our data, and display the score.

Step 11:Run the ninth code block to train and test the Top-MLP model with our data, and display the score.

*Note for 2.4: The code written to export the data to performance.txt has been commented out to not append extra data to the file whilst running the program at demo time. The classification report will be printed in the output for viewing*

Step 12: Run the tenth code block to view the classification report as well as the confusion matrix of BASE-MNB

Step 13: Run the eleventh code block to view the classification report as well as the confusion matrix of BASE-DT

Step 14: Run the twelfth code block to view the classification report as well as the confusion matrix of BASE-MLP

Step 15: Run the thirteenth code block to view the classification report as well as the confusion matrix of TOP-MNB

Step 16: Run the fourteenth code block to view the classification report as well as the confusion matrix of TOP-DT

Step 17: Run the fifteenth code block to view the classification report as well as the confusion matrix of BASE-MNB

Step 18: Remaining code blocks pertain to section 2.5. Run each block of code in this section to the parameters and scores of each model. 


  
### Part 3: Embeddings as Features
Step 1: Open the Jupyter notebook called "word2vec_embeddings.ipynb"
Step 1: Run the first block of code with the heading 3.1 to load the word2vec model

Step 2: Replace the jsonfiledirectory with the full path to goemotions.json.gz file in your computer in the second block of code

Step 3: Run the second block (3.2) of code to load the dataset and to tokenize it

Step 4: Run the third block of code (3.3) to get the average embeddings

Step 5: Run the fourth block of code (3.4) to get the overall hit rates

Step 6: Run the fifth block of code (3.5) to get the fit and score for sentiment for the Base-MLP

Step 7: Run the sixth block of code (3.6) to get the fit and score for sentiment for the Top-MLP

Step 8: Replace the jsonfiledirectory with the full path to goemotions.json.gz file in your computer in the eigth block of code

Step 9: Run the eigth block of code to get the libraries for the exploration

Step 10: Run the ninth block of code to load the data and tokenize the posts from the datasite

Step 11: Run the tenth block of code to get the average embeddings per post

Step 12: Run the eleventh block of code to get the hit rate

Step 13: Run the twelfth block of code to fit the data and print score for the Base-MNB
