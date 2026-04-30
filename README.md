Diyabet Tahminleme: Naive Bayes vs. SVM Karşılaştırması (R)

Bu proje, 100.000 gözlemli büyük bir veri seti üzerinde makine öğrenmesi teknikleri kullanarak diyabet hastalığını önceden tahmin etmeyi ve farklı modellerin performansını karşılaştırmayı amaçlar.

## 📊 Proje Özeti
Proje kapsamında; demografik veriler ve klinik göstergeler (BMI, HbA1c, glikoz seviyesi vb.) kullanılarak iki farklı sınıflandırma modeli eğitilmiştir.

## 🛠️ Kullanılan Teknolojiler
* **Dil:** R
* **Kütüphaneler:** `caret`, `pROC`, `e1071`, `LiblineaR`
* **Modeller:** Naive Bayes ve Support Vector Machine (SVM)

## 📈 Model Performans Sonuçları
Yapılan testler sonucunda modellerin başarı oranları şu şekildedir:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | %90.21 | 0.447 | 0.647 | 0.529 | 0.920 |
| **SVM (Linear)** | **%95.83** | **0.854** | 0.613 | **0.714** | **0.960** |

*SVM modelinin genel doğruluk oranı ve AUC değeri Naive Bayes modeline göre daha yüksek performans göstermiştir.*

## 📂 Dosya Yapısı
* `diyabet_proje.R`: Tüm analiz ve modelleme süreçlerini içeren R kodu.
* `model_ciktilari.txt`: Modellerin detaylı Confusion Matrix ve istatistik çıktıları.
* `roc_naive_bayes.png` & `roc_svm_fast.png`: Modellerin başarı grafiklerini gösteren ROC eğrileri.

## 🚀 Nasıl Çalıştırılır?
Gerekli paketleri (`caret`, `pROC`, `e1071`, `LiblineaR`) yükledikten sonra `diyabet_proje.R` dosyasını R Studio üzerinden çalıştırabilirsiniz.
