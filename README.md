E-commerce Customer Purchase Conversion Prediction (Maven Fuzzy Factory)

Programme: AnalystLab Africa Experience Lab Internship (Week 2)
Track: Track 3 Machine Learning Engineering
Author: DJAGBA

1. Business Problem Understanding (Task 1)

Customer Conversion Definition: Customer conversion occurs when a website browsing session on Maven Fuzzy Factory results in a completed purchase (converted = 1).

Business Problem: Across 472,871 historical sessions, only 32,313 resulted in an order, representing an overall conversion rate of 6.83 percent. Management wants to identify high-intent sessions upon arrival to optimize marketing ad spend, improve user experience, and increase overall conversion rates.

Machine Learning Objective: Build a supervised binary classification model (Logistic Regression) to predict the target variable converted (1 = purchase, 0 = no purchase).

Strict Anti-Data Leakage Rule: No post-purchase attributes (price_usd, cogs_usd, items_purchased, primary_product_id) are used as features. Predictions rely strictly on session-entry metadata (device_type, utm_source, utm_campaign, utm_content, http_referer, is_repeat_session, hour, day of week, and weekend indicator).

2. Data Preparation and Modelling (Tasks 2 and 3)

Missing Value Handling: Null values in UTM tags (83,328 sessions) and http_referer (39,917 sessions) represent organic or direct traffic and were explicitly imputed as organic_or_direct and direct_type_in.

Time-Based Split: Sessions were sorted chronologically by created_at and split into 80 percent training (378,296 sessions) and 20 percent testing (94,575 sessions) to simulate real-world future deployment without temporal leakage.

Class Imbalance Handling: Because only 6.83 percent of sessions convert, Logistic Regression was trained with class_weight balanced inside a Scikit-Learn Pipeline (OneHotEncoder and StandardScaler).

3. Model Evaluation and Business Metric Interpretation (Task 4)

Performance on the chronological test set (94,575 sessions):

Accuracy (37.75 percent): Overall share of correct predictions. Intentionally lower because the model prioritizes capturing actual buyers over predicting the majority non-buyer class (92 percent).

Precision (10.39 percent): Out of all sessions flagged as likely to convert (64,481), 10.39 percent (6,697) actually placed an order. Best suited for low-cost automated interventions such as targeted pop-ups or email nudges.

Recall or Sensitivity (86.05 percent): Out of 7,783 actual orders in the test set, the model successfully identified 6,697, missing only 13.95 percent of revenue opportunities.

F1-Score (18.53 percent): Harmonic mean of Precision and Recall, reflecting the intentional trade-off toward high buyer capture (Recall).

Confusion Matrix (Test Set):
True Negatives (TN): 29,008 non-converting sessions correctly filtered out.
False Positives (FP): 57,784 non-converting sessions flagged as high-potential.
False Negatives (FN): 1,086 actual buyers missed.
True Positives (TP): 6,697 actual buyers accurately detected.

4. Business Insights, Limitations and Recommendations (Task 5)

Most Predictive Features:
1. Device Type (device_type): Desktop sessions are the strongest positive driver of conversion, whereas Mobile sessions significantly reduce purchase likelihood.
2. Marketing Channels (utm_source and utm_campaign): Branded search (utm_campaign_brand), organic/direct traffic, and gsearch convert far better than paid social traffic (socialbook, utm_campaign_pilot).
3. Customer Loyalty (is_repeat_session): Returning visitors demonstrate a consistently higher propensity to buy.

Actionable Business Recommendations:
Fix the Mobile Checkout Funnel: Audit mobile UX, page load speed, and payment steps to close the conversion gap between Mobile and Desktop users.
Reallocate Marketing Spend: Scale high-converting brand and gsearch campaigns while auditing landing pages and audience targeting for socialbook.
Real-Time Session Personalization: Serve targeted product recommendations or urgency cues to high-scoring sessions, and capture leads (newsletter or discount sign-up) on low-scoring mobile social traffic.

Limitations and Future Improvements:
Limitations: Logistic Regression captures only linear relationships, and session-level entry data excludes intra-session browsing behavior.
Next Steps: Engineer pre-checkout behavioral features from website_pageviews.csv (pages viewed, landing page path, session duration before checkout), benchmark non-linear models (Random Forest, XGBoost), and tune probability thresholds (predict_proba) based on campaign ROI.

5. Repository Structure

data/ : Prepared modelling dataset (prepared_modelling_dataset.csv)
models/ : Serialized model (.joblib) and evaluation plots (evaluation_plots.png)
notebooks/ : End-to-end Jupyter Notebook (conversion_prediction.ipynb)
src/ : Production scripts (train.py, predict.py)
README.md : Project documentation, evaluation report and recommendations

6. How to Run the Project

Step 1: Clone the repository and navigate into the project folder
git clone https://github.com/DJAGBA/maven-fuzzy-conversion-ml.git
cd maven-fuzzy-conversion-ml

Step 2: Install required Python libraries
pip install pandas numpy scikit-learn matplotlib seaborn joblib

Step 3: Explore the full analysis and visualizations in Jupyter Notebook
jupyter notebook notebooks/conversion_prediction.ipynb

Step 4: Train and evaluate the Logistic Regression pipeline from the terminal
cd src
python train.py

Step 5: Run real-time session conversion predictions
python predict.py