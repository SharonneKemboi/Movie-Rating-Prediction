# Movie-Rating-Prediction
![Positive Reviews ](https://github.com/SharonneKemboi/Movie-Rating-Prediction/blob/master/Positive%20Reviewa.PNG)

A machine learning project to predict movie review sentiments (positive or negative) using the **IMDB Movie Reviews Dataset**.  
The project covers **data preprocessing, exploratory data analysis (EDA), model training, and evaluation**.  

![Movie Prediction](https://img.shields.io/badge/ML-RecommendationSystem-blue)  
![Regression](https://img.shields.io/badge/Algorithm-Collaborative%20Filtering%20%7C%20Regression-orange)  
![Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-yellow)  
![Python](https://img.shields.io/badge/Language-Python-green)  
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)  
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.1+-orange.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  
![Colab](https://img.shields.io/badge/Google%20Colab-Compatible-yellow.svg)  
![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)  

##  Dataset
- **Source**: IMDB Dataset (`IMDB Dataset.csv`)  
- **Size**: 50,000 movie reviews  
- **Features**:
  - `review`: Raw text of the movie review  
  - `sentiment`: Label (positive / negative)  

Aim is to predict the sentiment of a given review, simulating how a user might rate a movie they haven’t seen yet.

##  Project Workflow

1. **Data Loading & Cleaning**
   - Load dataset
   - Handle missing values and duplicates
   - Convert categorical labels into numerical format  

2. **Exploratory Data Analysis (EDA)**
   - Distribution of positive vs negative reviews
   - WordCloud for frequent terms
   - Review length analysis
   - Sentiment distribution bar plots
   - Feature correlations  

3. **Feature Engineering**
   - Text preprocessing: lowercasing, stopword removal, tokenization
   - TF-IDF Vectorization for feature extraction  

4. **Modeling**
   - Logistic Regression
   - Naive Bayes
   - Random Forest Classifier
   - Support Vector Machines (SVM)
   - Decision Tree Classifier  

5. **Evaluation**
   - Accuracy scores
   - Confusion Matrix
   - Comparison of models with visualization  

---

##  Exploratory Data Analysis (EDA)

###  Sentiment Distribution
- Balanced dataset: 25k positive reviews, 25k negative reviews.


###  WordCloud
- Frequent words for **positive reviews**: *great, amazing, love*  
- Frequent words for **negative reviews**: *bad, boring, worst*  

![WordCloud Negative](https://github.com/SharonneKemboi/Movie-Rating-Prediction/blob/master/Negative%20Reviews.PNG)

###  Review Length Analysis
- Longer reviews tend to be **negative**.
- Shorter reviews are more often **positive**.  

![Review Length Distribution](https://github.com/SharonneKemboi/Movie-Rating-Prediction/blob/master/Review%20Lengths.PNG)


##  Models & Results

We trained and compared multiple machine learning models:

- **Logistic Regression** → Baseline sentiment classifier  
- **Naive Bayes** → Fast and efficient for text classification  
- **Random Forest Classifier** → Better accuracy using ensemble learning  
- **SVM** → Robust performance with large text features  
- **Decision Tree Classifier** → Easy to interpret but lower accuracy  

 **Best model achieved >85% accuracy on test data.**

###  Model Performance Visualization
![Model Accuracy](https://github.com/SharonneKemboi/Movie-Rating-Prediction/blob/master/Model%20Accuracy.PNG)  

##  Key Insights

- Logistic Regression performs well for baseline text sentiment prediction  
- Random Forest improves accuracy by capturing more complex patterns  
- Longer reviews are often **negative**, while short ones are mostly **positive**  
- Text vectorization (TF-IDF/Bag-of-Words) plays a key role in performance  

##  Key Learning Outcomes

From this project, I learned:

- How to preprocess and clean text datasets  
- How to apply NLP techniques (tokenization, vectorization)  
- How to build and evaluate ML classifiers for text data  
- How to compare models and visualize results effectively  


##  Limitations

- Dataset provides **binary sentiment only**, not star ratings (1–5)  
- Predictions are based on reviews only, not **user-specific preferences**  
- Collaborative filtering could be improved by combining with NLP features  

##  Contributing

Contributions are welcome!

- Open issues for bugs or suggestions  
- Submit PRs for new features or visualizations  

### Possible Extensions:
- Add **deep learning models** (LSTM, Transformers, BERT) for better accuracy  
- Implement a **hybrid recommendation system** (content + collaborative)  
- Deploy the model as an API with Flask or FastAPI  


##  License

This project is licensed under the **MIT License** – see the LICENSE file.  


##  Acknowledgments

- **Arch Technologies** for supporting this learning project  
- **Open Source Community** for datasets & ML libraries  
- **Seaborn & Matplotlib** for visualization tools  


<div align="center">

 If you found this project useful, don’t forget to **star this repository**!

**Author**: Sharonne Kemboi  

 Nairobi, Kenya  
 
 *Interests: Data Science | AI | Machine Learning | NLP | Data Analytics* 

[LinkedIn](https://www.linkedin.com/in/sharonne-kemboi/) | [GitHub](https://github.com/SharonneKemboi)

</div>


</div>

