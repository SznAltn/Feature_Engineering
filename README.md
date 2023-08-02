# Feature_Engineering
Outliers  - Missing Values - Encoding Scaling - Feature Scaling - Feature Extraction - Feature Interactions - End-to-End Aplication

- **Outliers (Aykırı Değerler)**
- **Missing Values (Eksik Değerler)**
- **Encoding Scaling (kodlma ölçeklendirme)**
- **Feature Scaling (Özellik Ölçeklendirme)**
- **Feature Extraction (Özellik Çıkarımı)**
- **Feature Interactions (Özellik Etkileşimleri)**
- **End-to-End Aplication (Uçtan Uca Uygulama)**

**Özellik Mühendisliği:** Özellikler üzerinde gerçekleştirilen çalışmalar, ham veriden değişken üretmek.

**Veri Ön İşleme:** Çalışmalar öncesi verinin uygun hale getirilmesi sürecidir.

## 1.OUTLIERS (Aykırı Değerler)

Verideki genel eğilimin oldukça dışına çıkan değerlere aykırı değer denir.

Aykırı değerlerin etkileri doğrusal problemlerde oldukça yüksek iken ağaç problemlerinde daha düşüktür. Aykırı değerler, göz önünde bulundurulması gereken değerlerdir. Aykırı değerler neye göre belirlenir?

- Sektör Bilgisi
- Standart Sapma Yaklaşımı
- Z- Skoru Yaklaşımı
- Boxplot (interquartile range - IQR) Yöntemi

**Hatırlatma:**

- ****boxplot (kutu grafik) bir sayıal değişkenin dağılım bilgisini verir.
- histogram: bir sayısal değişkeni gösterebileceğimiz boxplot dan sonra en yaygın kullanılan grafiktir.

## Aykırı Değerler Nasıl Yakalanır?

Öncelikle eşik değerlere erişmeliyiz.

**Çok değişkenli aykırı değer:** elimizde olan 2 değişkenin örneğin 17 yaşında olup 3 defa evlenmek aykırı gibi yani tek başına 17 ve 3 aykırı değilken birlikte aykırıdırlar. Buna çok değişkenli etki denir. 

**Locak Outlier Factor:** çok değişkenli aykırı değer belirleme yöntemidir. Gözlemleri bulundukları konumda yoğunluk tabanlı skorlayarak buna göre aykırı değer olabilecek değerleri tanıma imkanı sağlar. Bir noktanın local yoğunluğu demek ilgili noktanın etrafındaki komşuluklar demektir, eğer bir nokta komşularının yoğunluğundan anlamlı bir şekide düşük ise bu durumda bu nokta daha seyrek bir bölgededir ve aykırı değer olabilir.

## 2. MISSING VALUES (Eksik Değerler)

Eksik Değer: Gözlemlerde eksiklik olması durumunu ifade etmektedir.

Eksik değer problemi nasıl çözülür?

- Silme
- Değer Atama Yöntemleri
- Tahmine Dayalı Yöntemler

Eksik veri ile çalışırken göz önünde bulundurulması greken önemli konulardan birisi: Eksik verinin rassallığıdır.

**Eksik değere sahip gözlemlerin veri setinden direkt çıkarılması ve rassallığın incelenmemesi, yapılacak istatistiksel çıkarımlarının ve modelleme çalışmalarının güvenirliğini düşürecektir.   (Alpar, 2011)**

**Eksik gözlemlerin veri setinden direkt çıkarılabilmesi için veri setinden eksikliğin bazı durumlarda kısmen, bazı durumlarda ise tamamen rastlantısal olarak oluşması gerekmektedir.**

**Eğer eksiklikler değişkenler ile ilişkili olarak ortaya çıkan yapısal problrmler  ile meydana gelmiş ise bu durumda yapılacak silme işlemleri ciddi yanlılıklara sebep olacaktır. (Tabachnick ve Fidell, 1996)**

**Silme**: eksik değerler veri setinden çıkarılır.

**Değer Atama Yöntemleri:** eksik değerlere mod, medyan ya da ortalama değer ile doldurulur.

**Tahmine Dayalı Yöntemler:** Eksikliğe sahip olan değişkeni bağımlı değişken, diğer değişkenleri bağımsız değişken kabul edilir. Modelleme işlemi gerçekleştirilir ve modelleme işlemine göre eksik değerleri tahmin etmeye çalışılır. 

## 3. ENCODING (Label Encoding, One-Hot Encoding, Rare Encoding)

**Encod:** Değişkenlerin temsil şekillerinin değiştirilmesi

**Label Encoding > Binary Encoding**

- **Label Encoding:** Label ları yeniden kodlamak yani string i 0-1 gibi ifadelerle doldurmaktır. 2 den fazla sınıfı varsa label encoding denir ve label encoding genel isimlendirmedir. Sınıflar arası fark varken kullanılır. Kategorik değişkenleri sayısal değişken olarak ifade etmemizi sağlar.
- **Binary Encoding:** Kategorik değişkenin 2 sınıfı varsa bu 1-0 gibi isimlendirilirse buna binary encoding denir.
- **One-Hot Encoding:**  Sınıflar arası fark yokken  label encoding de sorun oluyordu, ölçüm problemi yapmadan kategorik değişkenin sınıflarını değişkene dönüştürüyoruz. İkiden fazla sınıfı olan kategorik değişkenlerin 1-0 olarak encode edilmesi için kullanılır.

**Dummi değişken tuzağı:** kukla değişken tuzağı one-hot-encoding yönteminde kategorik değişkenleri sınıflarını değişkene dönüştürürken bu yeni oluşturulan değişkene dummi değişken (kukla değişken) denir. Kukla değişken birbiri üzerinden oluşturulabilir olursa bir ölçme problemi ortaya çıkar ve bu da yüksek korelasyona sebep olur. drop=first diyerek birbiri üzerinden oluşturulma durumu ortadan kaldırılır ve kurtuluruz tuzaktan.

- **Rare Encoding:** rare nadir, ender demektir. Nadir görülen sınıf  model kurulurken bize iş yükü oluşturmasın diye bir eşik değer belirlenir ve bu frekanstan aşağıda olanlar bir araya getirilir ve hepsine rare adı verilir. Elimizde bol kategorik değişken old rare kullanmalıyız.

## **Feature Scaling (Özellik Ölçeklendirme)**

Özellik ölçeklendirmede amaçlarımızdan birisi değişkenler arasındaki ölçüm farklılığını gidermektir. Modellerimizin değişkenlerimize eşit şartlar altında yaklaşmasını sağlamak için standartlaştırma yapıyoruz. İkinci amacımız, gradient descent kullanan algoritmaların train sürelerini kısaltmak. Diğer bir amacımız ise uzaklık temelli yöntemlerde büyük değerlere sahip değişkenler dominantlık gösterir bu da yanlılığa sebep olur ve biz de buna engel olmaya çalışırız.

- **Standard Scaler:** Klasik standartlaştırma. Ortalama tüm gözlem birimlerinden çıkarılır, standart sapmaya bölünür.  z = (x - u) / s
- **RobustScaler:** Medyanı çıkar iqr'a böl. Bu yöntemde merkezi eğilimi ve değişimi göz önünde bulundurmuş oluruz. Aykırı değerlerden etkilenmez.
- **MinMaxScaler:** Verilen 2 değer arasında değişken dönüşümü, ön tanımlı aralık değerleri [0,1] dir.

 X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

 X_scaled = X_std * (max - min) + min

## 4. FEATURE EXTRACTION (Özellik Çıkarımı)

Ham veriden değişken türetmektir. Yapısal olan ya da olmayan verileri feature üreterek numeric şekilde temsil edebilmeliyiz. 

- **Binary Features:** Flag, Bool, True-False. Var olan değişkenler üzerinden 1-0 gibi yeni değişkenler türetmek, dönüşüm yapmıyoruz.
- **Text Features:** Metinler üzerinden özellik türetmeye çalışırız. Örneğin harfleri saydırabilir, kelimeler sayılabilir, dr. gibi değerler çekilebilir.
- **Regex Feature:** regular expuration  lar ile değişken türetme işlem yapıyoruz, mr., mrs., miss gibi ifadelere bakılabilir.
- **Date Features:** yıllara, aylara, günlere, yıl farklarına, ay farklarına ya da benzeri şekilde sınıfları değişken olarak atayıp inceleme yapabilir.
