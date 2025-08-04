<img src="https://flagcdn.com/w40/us.png" alt="English" width="30"/> English
<br>

<h3>💎 CART (Classification and Regression Trees) Implementation</h3>
This project demonstrates a comprehensive implementation of a Classification and Regression Tree (CART) model using Python's scikit-learn library. The purpose is to walk through the complete machine learning pipeline, from data preparation to model optimization and interpretation.

🚀 Project Goal
The primary goal is to build, evaluate, and optimize a decision tree model for a classification task, using various industry-standard techniques to ensure robust performance and to prevent overfitting.

📊 Dataset Used

Data Source: A user-provided CSV file.

Content: A dataset for a classification problem where the target variable is named Outcome.

🧠 Model and Methodology

Data Preparation: The dataset is loaded, and the features (X) are separated from the target variable (y).

Baseline Model: A basic DecisionTreeClassifier is trained on the entire dataset to establish a performance benchmark.

Model Evaluation:

Holdout Method: The data is split into training and testing sets to evaluate the model's out-of-sample performance.

Cross-Validation: The model is evaluated using 5-fold cross-validation with multiple scoring metrics (accuracy, f1, roc_auc).

Hyperparameter Optimization: A GridSearchCV approach is used to systematically search for the best combination of hyperparameters, specifically max_depth and min_samples_split, to find the optimal model.

Final Model: The model is retrained with the best hyperparameters found during the grid search and then evaluated again using cross-validation.

Model Interpretation:

Feature Importance: A bar plot visualizes the importance of each feature in the final model's decision-making process.

Decision Tree Visualization: The final decision tree is plotted to provide a visual representation of the decision rules.

Decision Rules Extraction: The textual rules of the decision tree are extracted and printed.

Validation Curve (Bonus): A validation curve is plotted to analyze the model's performance on both training and validation sets as a single hyperparameter is varied, helping to diagnose bias-variance tradeoffs.

✨ Results

Hyperparameter Tuning:

Best Parameters: {'max_depth': 4, 'min_samples_split': 11}

Best CV Score: ~0.76 (e.g., ROC AUC)

Final Model Performance (Cross-Validation):

Final CV Accuracy: ~0.77

Final CV F1: ~0.67

Final CV ROC AUC: ~0.83

These results demonstrate the successful optimization of the CART model, leading to improved predictive performance and providing a clear understanding of the model's decision process.

<br><br><br>

<img src="https://flagcdn.com/w40/tr.png" alt="Turkish" width="30"/> Türkçe
<br>

<h3>💎 CART (Sınıflandırma ve Regresyon Ağaçları) Uygulaması</h3>
Bu proje, Python'ın scikit-learn kütüphanesini kullanarak kapsamlı bir Sınıflandırma ve Regresyon Ağacı (CART) modelinin uygulamasını göstermektedir. Amacı, veri hazırlığından model optimizasyonuna ve yorumlamaya kadar eksiksiz bir makine öğrenimi sürecini adım adım anlatmaktır.

🚀 Proje Amacı
Temel amaç, doğru performansı sağlamak ve aşırı öğrenmeyi (overfitting) önlemek için çeşitli standart teknikleri kullanarak bir sınıflandırma görevi için bir karar ağacı modelini oluşturmak, değerlendirmek ve optimize etmektir.

📊 Kullanılan Veri Seti

Veri Kaynağı: Kullanıcı tarafından sağlanan bir CSV dosyası.

İçerik: Hedef değişkeni Outcome olarak adlandırılan bir sınıflandırma problemi için bir veri seti.

🧠 Model ve Yöntem

Veri Hazırlığı: Veri seti yüklenir, özellikler (X) hedef değişkenden (y) ayrılır.

Temel Model: Performans için bir başlangıç noktası oluşturmak üzere tüm veri seti üzerinde temel bir DecisionTreeClassifier modeli eğitilir.

Model Değerlendirme:

Holdout Yöntemi: Modelin test verilerindeki performansını değerlendirmek için veri, eğitim ve test setlerine ayrılır.

Çapraz Doğrulama (Cross-Validation): Model, accuracy, f1 ve roc_auc gibi birden fazla metrik kullanılarak 5 katlı çapraz doğrulama ile değerlendirilir.

Hiperparametre Optimizasyonu: Modelin en uygun performansını bulmak için özellikle max_depth ve min_samples_split hiperparametreleri için sistematik bir arama yapan GridSearchCV yaklaşımı kullanılır.

Final Model: En iyi performansı gösteren hiperparametreler kullanılarak model yeniden eğitilir ve çapraz doğrulama ile tekrar değerlendirilir.

Model Yorumlama:

Özellik Önem Grafiği: Son modelin karar verme sürecindeki her bir özelliğin önemini gösteren bir çubuk grafik görselleştirilir.

Karar Ağacı Görselleştirme: Karar kurallarını görsel olarak temsil etmek için nihai karar ağacı çizilir.

Karar Kuralları Çıkarma: Karar ağacının metinsel kuralları çıkarılarak yazdırılır.

Doğrulama Eğrisi (BONUS): Modelin tek bir hiperparametre değiştikçe hem eğitim hem de doğrulama setlerindeki performansını analiz etmek ve sapma-varyans dengesini teşhis etmek için bir doğrulama eğrisi çizilir.

✨ Sonuçlar

Hiperparametre Optimizasyonu:

En İyi Parametreler: {'max_depth': 4, 'min_samples_split': 11}

En İyi CV Skoru: ~0.76 (örn: ROC AUC)

Final Model Performansı (Çapraz Doğrulama):

Final CV Accuracy: ~0.77

Final CV F1: ~0.67

Final CV ROC AUC: ~0.83

Bu sonuçlar, CART modelinin başarılı bir şekilde optimize edildiğini, gelişmiş tahmin performansı sağladığını ve modelin karar verme sürecini net bir şekilde anlama imkanı sunduğunu göstermektedir.
