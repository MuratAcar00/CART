### CART (Classification and Regression Trees) ###

# Gerekli Kütüphaneler:

import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, plot_tree
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

warnings.simplefilter(action="ignore", category=Warning)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


# 1.Veri Yükleme ve Ön İşleme

df = pd.read_csv(r"Dosya Yolu")
print("Veri Kümesi İlk 5 Satır:")
print(df.head())

x = df.drop("Outcome", axis=1)
y = df["Outcome"]


# 2.Modelleme: Temel CART

model = DecisionTreeClassifier(random_state=1).fit(x, y)
y_pred = model.predict(x)
y_prob = model.predict_proba(x)[:, 1]

print("\n### 2. Temel CART Model Performansı (Eğitim Verisi Üzerinde) ###")
print("Confusion Matrix: \n", classification_report(y, y_pred))
print("ROC AUC:", roc_auc_score(y, y_prob))


# 3.Holdout Yöntemi ile Değerlendirme

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
model = DecisionTreeClassifier(random_state=17).fit(x_train, y_train)

print("\n### 3. Holdout Yöntemi ile Değerlendirme ###")
print("Train Score: \n", classification_report(y_train, model.predict(x_train)))
print("Train ROC AUC:", roc_auc_score(y_train, model.predict_proba(x_train)[:, 1]))
print("Test ROC AUC:", roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))


# 4.Cross Validation (CV)

cv_model = DecisionTreeClassifier(random_state=17)
cv_result = cross_validate(cv_model, x, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

print("\n### 4. Cross-Validation Sonuçları ###")
print("CV Accuracy:", cv_result["test_accuracy"].mean())
print("CV F1:", cv_result["test_f1"].mean())
print("CV ROC AUC:", cv_result["test_roc_auc"].mean())


# 5.Hiperparametre Optimizasyonu

params = {"max_depth": range(1, 11), "min_samples_split": range(2, 20)}
grid = GridSearchCV(cv_model, params, cv=5, n_jobs=-1, verbose=1).fit(x, y)

print("\n### 5. Hiperparametre Optimizasyonu Sonuçları ###")
print("Best Params:", grid.best_params_)
print("Best CV Score:", grid.best_score_)


# 6.Final Model

final_model = DecisionTreeClassifier(**grid.best_params_, random_state=17).fit(x, y)
final_cv = cross_validate(final_model, x, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

print("\n### 6. Final Model Sonuçları ###")
print("Final CV Accuracy:", final_cv['test_accuracy'].mean())
print("Final CV F1:", final_cv['test_f1'].mean())
print("Final CV ROC AUC:", final_cv['test_roc_auc'].mean())


# 7. Özellik Önem Grafiği

def plot_importance(model, features, top_n=10):
    importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': model.feature_importances_})
    importance_df = importance_df.sort_values("Importance", ascending=False).head(top_n)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

print("\n### 7. Özellik Önem Grafiği ###")
plot_importance(final_model, x)


# 8. Validation Curve (BONUS)

def val_curve(model, x, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_scores, test_scores = validation_curve(
        estimator=model,
        x=x,
        y=y,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        cv=cv)
    
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, np.mean(train_scores, axis=1), label="Train")
    plt.plot(param_range, np.mean(test_scores, axis=1), label="Validation")
    plt.title(f"Validation Curve: {param_name}")
    plt.xlabel(param_name)
    plt.ylabel(scoring.upper())
    plt.legend()
    plt.tight_layout()
    plt.show()

print("\n### 8. Validation Curve ###")
# validation_curve eğitilmemiş bir model bekler, bu yüzden final_model yerine DecisionTreeClassifier() kullanıldı.
val_curve(DecisionTreeClassifier(random_state=17), x, y, param_name="max_depth", param_range=range(1, 11))
val_curve(DecisionTreeClassifier(random_state=17), x, y, param_name="min_samples_split", param_range=range(2, 20))


# 9. Karar Ağacını Görselleştirme

def plot_tree_inline(model, feature_names):
    plt.figure(figsize=(40, 20))
    plot_tree(model, feature_names=feature_names, filled=True, class_names=True, rounded=True)
    plt.title("Decision Tree")
    plt.show()

print("\n### 9. Karar Ağacını Görselleştirme ###")
plot_tree_inline(final_model, x.columns)


# 10. Karar Kurallarını ve Kodları Çıkarma

print("\n### 10. Karar Kuralları ve Örnek Tahmin ###")
print("\nKarar Ağacı Kuralları:")
print(export_text(final_model, feature_names=list(x.columns)))

# print("\nKarar Ağacı Python Kodu:")
# print(skompile(final_model.predict).to("python/code")) # Bu satır çalışmıyorsa yorum satırı yapabilirsiniz.

# Örnek tahmin
sample = pd.DataFrame([x.iloc[1].values], columns=x.columns)
print("\nÖrnek Veri:")
print(sample)
print("Tahmin:", final_model.predict(sample))
