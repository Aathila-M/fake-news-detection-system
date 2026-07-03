FAKE NEWS DETECTION SYSTEM USING NATURAL LANGUAGE PROCESSING AND MACHINE LEARNINGFAKE NEWS DETECTION SYSTEM USING NATURAL LANGUAGE PROCESSING AND MACHINE LEARNING

Abstract
    Fake news has become a serious problem due to the rapid growth of online news portals and social media platforms. False information can mislead people, influence public opinion, and create social instability. This project proposes a Fake News Detection System using Natural Language Processing and Machine Learning techniques. The system analyzes the textual content of news articles and classifies them as Real or Fake.
    The proposed system performs text preprocessing, feature extraction using TF-IDF, model training using algorithms such as Logistic Regression, Passive Aggressive Classifier, and Random Forest, and prediction through a Flask-based web application. The system also includes a dashboard to store and display prediction history using SQLite database. This project helps users verify news content before trusting or sharing it.

Chapter 1: Introduction
    Fake news refers to false or misleading information presented as genuine news. With the increasing use of digital media, fake news spreads faster than traditional news sources. Manual verification of every article is difficult, so an automated system is needed.
    This project uses Natural Language Processing to clean and process news text. Machine learning models are trained on labeled datasets containing real and fake news articles. The trained model predicts whether a given news article is real or fake.

Objectives
        1. To collect and preprocess real and fake news data.
        2. To extract text features using TF-IDF vectorization.
        3. To train machine learning models for fake news classification.
        4. To compare model performance using evaluation metrics.
        5. To develop a Flask web application for user interaction.
        6. To store prediction history in a database.
        7. To provide a dashboard for viewing past predictions.

Chapter 2: Literature Survey
    Existing fake news detection systems use various methods such as rule-based checking, source credibility analysis, machine learning, and deep learning. Traditional approaches depend on manual fact-checking, which is time-consuming. Machine learning approaches are faster and can learn patterns from text data.
    Common algorithms used in previous studies include Naive Bayes, Logistic Regression, Support Vector Machine, Random Forest, and Passive Aggressive Classifier. Recent systems also use deep learning models such as LSTM and BERT for better contextual understanding.
Chapter 3: System Analysis
    Existing System
        1. Manual fact-checking websites verify news manually.
        2. Users must search multiple sources to confirm authenticity.
        3. Detection is slow and depends on human effort.
        4. Existing systems may not provide instant prediction.
    Proposed System
        1. User enters news text into the web application.
        2. The system preprocesses the text.
        3. TF-IDF converts text into numerical features.
        4. The trained ML model predicts Real or Fake.
        5. Result is displayed with confidence score.
        6. Prediction history is stored in SQLite database.
        7. Dashboard displays previous predictions.
    Advantages
        1. Fast prediction.
        2. Easy web interface.
        3. Uses machine learning for classification.
        4. Stores prediction records.
        5. Can be extended with deep learning and explainability.

Chapter 4: System Design
    Architecture
        User
        ↓
        Flask Web Interface
        ↓
        Text Preprocessing
        ↓
        TF-IDF Vectorizer
        ↓
        Machine Learning Model
        ↓
        Prediction Result
        ↓
        SQLite Database
        ↓
        Dashboard
    
    Modules
        1. Dataset Module
        2. Preprocessing Module
        3. Model Training Module
        4. Prediction Module
        5. Database Module
        6. Dashboard Module
        7. Web Application Module
    
    Data Flow
        Input News Text
        ↓
        Cleaning and preprocessing
        ↓
        Feature extraction
        ↓
        Model prediction
        ↓
        Fake / Real result
        ↓
        Save result in database
        ↓
        Display result to user

Chapter 5: Implementation
    Technologies Used
        Programming Language: Python
        Frontend: HTML, CSS
        Backend: Flask
        Database: SQLite
        ML Libraries: Scikit-learn, Pandas, NumPy
        NLP Library: NLTK
        Visualization: Matplotlib, Seaborn
        Model Storage: Joblib / Pickle
    
    Algorithms Used
        1. Logistic Regression
        2. Passive Aggressive Classifier
        3. Random Forest Classifier
        4. Optional: LSTM Deep Learning Model

    Preprocessing Steps
        1. Convert text to lowercase.
        2. Remove URLs.
        3. Remove HTML tags.
        4. Remove punctuation.
        5. Remove numbers.
        6. Remove stopwords.
        7. Apply lemmatization.
    Feature Extraction
        TF-IDF is used to convert textual news data into numerical format. It gives importance to meaningful words based on their frequency in a document and across the dataset.

Chapter 6: Testing And Results
    Model Evaluation Metrics
        Accuracy
        Precision
        Recall
        F1-score
        Confusion Matrix
    
    Current Output
        Best Model: Logistic Regression
        Accuracy: 75%
    
    Sample Test Cases
        | Test Case | Input                             | Expected Output |
        |
        | 1 |         Government announces new policy   | REAL |
        | 2 |         Mobile phones will explode tonight| FAKE |
        | 3 |         NASA releases new space images    | REAL |
        | 4 |         Secret medicine cures all diseases| FAKE |

Chapter 7: Conclusion
    The Fake News Detection System successfully classifies news articles as Real or Fake using Natural Language Processing and Machine Learning. The system provides a simple web interface where users can enter news text and receive instant predictions. The use of TF-IDF and machine learning algorithms makes the system efficient for text classification.
    The project also includes database storage and dashboard functionality, making it suitable for academic demonstration and further development. With a larger dataset and advanced models such as LSTM or BERT, the system can achieve better accuracy and reliability.

Future Enhancements
    1. Use a larger real-world dataset.
    2. Add BERT transformer-based model.
    3. Add multilingual fake news detection.
    4. Add source credibility checking.
    5. Add browser extension support.
    6. Add social media post verification.
    7. Add explainable AI to highlight suspicious words.
    8. Deploy system online using Render or Railway.

Bibliography
    1. Scikit-learn Documentation
    2. Flask Documentation
    3. NLTK Documentation
    4. Kaggle Fake News Dataset
    5. Research papers on Fake News Detection using NLP and Machine Learning
    