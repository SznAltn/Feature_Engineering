#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None) # tüm sütunları göster
pd.set_option('display.max_rows', None) # tüm satırları göster
pd.set_option('display.float_format', lambda x: '%.3f' % x) # virgülden sonra 3 basamak göster
pd.set_option('display.width', 500) # yan yana 500 sütuna kadar yaz

# veri okuma fonksiyonu
def load_application_train():
    data = pd.read_csv("Datasets/future_engineering/application_train.csv")
    return data

df = load_application_train()
df.head()


def load():
    data = pd.read_csv("Datasets/future_engineering/titanic.csv")
    return data


df = load()
df.head()



#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

#############################################
# Aykırı Değerleri Yakalama
#############################################

###################
# Grafik Teknikle Aykırı Değerler
###################

# boxplot: kutu grafik bir sayıal değişkenin dağılım bilgisini verir.
# histogram: bir sayısal değişkeni gösterebileceğimiz boxplot dan sonra en yaygın kullanılan grafiktir.
sns.boxplot(x=df["Age"])
plt.show(block=True)

###################
# Aykırı Değerler Nasıl Yakalanır?
###################
# 1. çeyrek değer için
q1 = df["Age"].quantile(0.25)

# q3 değeri için
q3 = df["Age"].quantile(0.75)

iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

# aykırı değerler
df[(df["Age"] < low) | (df["Age"] > up)]

# aykırı değerlerin index bilgileri
df[(df["Age"] < low) | (df["Age"] > up)].index

###################
# Aykırı Değer Var mı Yok mu?
###################

# aykırı değer var mı yok mu hiç boolean
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

# alt değerler var mı hiç
df[(df["Age"] < low)].any(axis=None)

# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

##################################################################################################################
# İşlemleri Fonksiyonlaştırmak
##################################################################################################################

# fonksiyon yazalım aykırı değerler için
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# yaş değişkeni için aykırı değerleri için alt ve üst değer bul
outlier_thresholds(df, "Age")

# fare değişkeni için aykırı değerleri için alt ve üst değer bul
outlier_thresholds(df, "Fare")

# fare değişkeni için low ve up değerleri tut
low, up = outlier_thresholds(df, "Fare")

# fare nin low dan düşükleri veya fare nin up dan büyüklerinin ilk 5 değeri
df[(df["Fare"] < low) | (df["Fare"] > up)].head()

# index lerine erişmek için
df[(df["Fare"] < low) | (df["Fare"] > up)].index

# dataframe nin kolonlarında dolaş, up ve üst limitleri hesapla
# eğer up limitten büyükler veya alt limitten düşükler var ise true döndür değilse false
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age") # true döndü
check_outlier(df, "Fare") # true döndü

##################################################################################################################
# grab_col_names
##################################################################################################################

# büyük verilerde değişken sayısı fazla ve veri tipleri farklı olursa ne yapacağız bunu gözlemleyelim:
dff = load_application_train()
dff.head()

# cat_th = 10 : 10dan az eşsiz sınıfa sahipse ve değişkenin tipi de object değilse
# sayısal olsa dahi bu kategoriktir. numeric görünüyor ama kategoriktir

# car_th = 20 : bir kategorik değişkenin 20 den fazla eşsiz sınıfı var ise ve tipi de kategorik ise
# kategorik görünüyor ama kardinaldir. yani ölçülebilir değildir. çok fazla sınıf sayısı var.

# numeric kolonlar için ise:
# tipi objectten farklı olanlar getirdik, int veya float olanlar gelicek
# numeric görünen ama kategorik olanları çıkarıyoruz
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

# cat_cols, num_cols ve cat_but_car olanlara uygulayalım fonksiyonu
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# passengerId nin içinde olmayan ama numeric olan kolonlar
num_cols = [col for col in num_cols if col not in "PassengerId"]

# numeric kolonlarda gez, kolon, outlier olanları yaz
for col in num_cols:
    print(col, check_outlier(df, col))

# dff dataframe inde yani büyük veri setinde kategorik, nümerik ve kategorik ama kardinal olanları grab_col_names e koy
cat_cols, num_cols, cat_but_car = grab_col_names(dff)

# numerik kolonlar, "SK_ID_CURR" nin içinde olmayan ama numeric olan kolonlar
num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

# numeric kolonlarda gez, kolon, outlier olanları yaz dff verisetinde yani büyük veri setinde
for col in num_cols:
    print(col, check_outlier(dff, col))

###################################################################################################################
# Aykırı Değerlerin Kendilerine Erişmek
###################################################################################################################

# index alıp almak opsiyonlu olsun yani argümanı verip ister true olsun ister false ama default u false olsun
# eğer 10 dan fazla aykırı değer varsa head atsın
# 10 dan büyük değilse hepsi getirsin gözlemleyelim
# son olarak da eğer index true ise return et
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

# df te age için analiz et aykırı değerleri
grab_outliers(df, "Age")

# index true ise age değişkeninde aykırı değerleri analiz et
grab_outliers(df, "Age", True)

# saklamak istersek index bilgisini:
age_index = grab_outliers(df, "Age", True)

# 1.aykırı değerlere bak low ve up limit leri return et
# 2. sadece bir değişkende outlier var mı yok mu
# 3.grab_outliers yap aykırı değerleri yakalamaya çalışmak
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", True)

#################################################################################################################
# Aykırı Değer Problemini Çözme
#################################################################################################################

###################
# Silme
###################

# low ve up ları bul
low, up = outlier_thresholds(df, "Fare")
df.shape

# verisetinde altlimitten aşağıda ya da üstlimitten yukarıda olmayanları getir
df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

# fonk yazalım aykırı değerleri kaldırmak için:
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

# numeric kolonlarda gez kaldırılması gerekenleri new_df olarak ata
for col in num_cols:
    new_df = remove_outlier(df, col)

# eski dataframeden yeni dataframe nin shape[0] ını çıkarıyorum
# kaç değişiklik old saptıyorum
df.shape[0] - new_df.shape[0]

###################
# Baskılama Yöntemi (re-assignment with thresholds)
###################

# eşik değerleri bul
low, up = outlier_thresholds(df, "Fare")

# aykırı değerleri getir
df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]

# 2.yol: aykırı değerleri getirmek için
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

# up denilecekler
df.loc[(df["Fare"] > up), "Fare"] = up

# low denilecekler
df.loc[(df["Fare"] < low), "Fare"] = low

# fonksiyon yazalım bunun için
# limitleri bulalım ve alt limitten küçük olnlara low büyük olanlara up diyelim
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

# outlier var mı yok mu
for col in num_cols:
    print(col, check_outlier(df, col))

# herbir değişken için thresholds ile değişkenlerin değerlerini değiştirelim
for col in num_cols:
    replace_with_thresholds(df, col)

# tekrar soralım outlier var mı yok mu diye true demişti şimdi false diyor
for col in num_cols:
    print(col, check_outlier(df, col))


###################
# Recap
###################

df = load()
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)

remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")




##################################################################################################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
##################################################################################################################
# 17, 3

# örnek olarak diamonds veri setine bakalım
# sayısal değişkenleri seçelim
# eksik değerleri drop ederek bakıyoruz

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.shape

# aykırı değer var mı diye soralım:
for col in df.columns:
    print(col, check_outlier(df, col))

# low ve up değerleri getirelim carat değişkeni için
low, up = outlier_thresholds(df, "carat")

# low değerinin altında veya up değerinin üstündekilerinin sayısı
df[((df["carat"] < low) | (df["carat"] > up))].shape

# # low ve up değerleri getirelim depth değişkeni için
low, up = outlier_thresholds(df, "depth")

# low değerinin altında veya up değerinin üstündekilerinin sayısı
df[((df["depth"] < low) | (df["depth"] > up))].shape


# çok değişkenli yaklaşalalım
# komşuluk sayısı 20 yapıyoruz (ön tanımlı değeri )
# local outlier skorlarını getirecek
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

# negatif olanları tutalım
df_scores = clf.negative_outlier_factor_
df_scores[0:5] # 5 tanesini gözlemleyelim
# df_scores = -df_scores # eksi değerlerle gözlemlemek istemezsek

# -1 e yakın olması inlier
# pozitifteki yorumun tersini yapıyoruz
# sıralayalım en kötü 5 değeri (küçükten büyüğe sıralıyoruz)
np.sort(df_scores)[0:5]
# bu çıktıda -1 e en yakın olanlar en iyilerdir

# scor ları tutalım
scores = pd.DataFrame(np.sort(df_scores))

# her bir nokta eşik değerleri temsil ediyor ve bu eşik değerlere göre grafik oluşturuldu
# eğim değişikliğinin  en son, en bariz, en dik olduğu yer tespit edilir
# ve o nokta eşik değer olarak kabul edilir dirsek yönteminde
# scor lar nekadar aşağıda ise nekadar eksi değere sahipse okadar kötüdür.

scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show(block=True)

# sıralamadan sonra 3. index teki değeri getiriyoruz çünkü 3.index teki değeri eşik değer olarak belirledik
th = np.sort(df_scores)[3]

# eşik değerden daha küçük olanları aykırı değer olarak belirliyoruz
df[df_scores < th]

# kaç tane old görelim
df[df_scores < th].shape
# teker teker bakıldığında binlerce aykırı değer vardı
# ama çok değişkenliye bakıldığında sadece 3 tane aykırı değer geldi
# neden? gözlemleri inceleyelim:
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

# index bilgilerini görelim
df[df_scores < th].index

# istersek silebiliriz çok değişkenli aykırıları
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# ağaç yöntemi kullanıyorsak aykırı değerlere dokunmuyoruz
# ya da çok ucundan yani %5 e %95 gibi

#################################################################################################################
# MISSING VALUES (Eksik Değerler)
#################################################################################################################

#############################################
# Eksik Değerlerin Yakalanması
#############################################

df = load()
df.head()

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

# eksikliğin tüm verisetindeki oranı
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# sadece eksik değerleri seçelim
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


# eksik değerler için fonks yazalım
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False) # eksik değer sayısı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False) # oranı
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio']) # bunları df e çevir
    print(missing_df, end="\n") # yazdır

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)


#############################################################################################
# Eksik Değer Problemini Çözme
#############################################################################################

missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################
# eğer ağaca dayalı yöntemler kullanılıyorsa eksik değerler
# tıpkı aykırı değerler gibi gözardı edilebilir.
# özetle, ağaç yöntemleri daha esnek ve dallara ayırmalı bir şekilde çalışıldığı için
# bu noktadaki aykırılar ve eksiklikler neredeyse yoka yakındır
# istisna; regresyon problemi ise ve bağımlı değişken de sayısal değişken ise sonuca gitme süresi uzar
# 1.çözüm:
df.dropna().shape

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################
# boşluğu ortalama ile doldurma
df["Age"].fillna(df["Age"].mean()).isnull().sum()
# median ile doldur
df["Age"].fillna(df["Age"].median()).isnull().sum()
# sabit bir değer mesela 0 ile doldurma
df["Age"].fillna(0).isnull().sum()

# df.apply(lambda x: x.fillna(x.mean()), axis=0)
# verisetinde tüm değişkenler için yapma axis=0 ifadesi satırlara göre git demek
# sütun bazında satırlar açısından bakmaktır
# ortalama değer ile boşlukları doldur ama değişkenin tipi object ten farklı ise bu değişkeni ortalama ile doldur
# tipi object ten farklı değilse olduğu gibi kalsın
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)
# eksik verilerden kurtulduk ama kabin ve embarked değişkenlerine dokunmadık
# kategorik değişkenlere dokunmadık
# kategorik değişkeni doldurmak için en iyi yöntem modunu almaktır.
# .mode()[0] deyince string karşılığını yani values u görürüz
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

# eksiklikleri string özel bir ifade ile doldurmak istersek(burada missing yazdık)
df["Embarked"].fillna("missing")

# programatik bir şekilde otomatik doldurmak için:
# ilgili değişkeni doldur, mod ile, eğer type ı object ise ve
# eşsiz değer sayısı 10 dan küçükse
# değilse old gibi kalsın
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and (x.nunique()) <= 10) else x, axis=0).isnull().sum() # sor

########################################################################################################################
# Kategorik Değişken Kırılımında Değer Atama
########################################################################################################################

df.groupby("Sex")["Age"].mean()

df["Age"].mean()

# eksik değerleri ortalama ile doldur ama cinsiyete göre kırarak ata bunu transform ile yapıyoruz
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

# şimdi cinsiyete göre kırılan veride kadınların ortalamalarını görelim
df.groupby("Sex")["Age"].mean()["female"]

# 2. yol loc metodu ile
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum() # kabin ve embarked kaldı

##################################################################################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
##################################################################################################

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
# drop_first=True yazdık bu 2 sınıfa sahip kategorik değişkenin ilk sınıfını atar 2.sınıfı tutar
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# değişkenlerin standartlatırılması
scaler = MinMaxScaler() # değerleri 1 ile 0 arasına dönüştür
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# tahmine dayalı boşlukları doldurma yöntemidir.
# boş olan gözlemin en yakın 5 komuşunun ortalamasını alır ve boş olana atar.
# inverse metodu scaler ın dönüştürmesini geri alır
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
# doldurulan yerleri gözlemleyelim eski hali ile kıyaslayalım
df["age_imputed_knn"] = dff[["Age"]]

# atanan değerleri yeni ve eski halini görelim
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]


######################
# Recap
#######################
df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma




######################################################################################
# Gelişmiş Analizler
######################################################################################

###################
# Eksik Veri Yapısının İncelenmesi
###################
# eksik değerleri görsel bar ile gözlemleyelim
msno.bar(df)
plt.show(block=True)

# eksiklikler birarada mı değil mi onu gözlemlememizi sağlar
msno.matrix(df)
plt.show(block=True)

# ısı haritası
msno.heatmap(df)
plt.show(block=True)

#####################################################################################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
#####################################################################################
# eksik değerleri görelim
missing_values_table(df, True)
# eksik değerleri kaydettik
na_cols = missing_values_table(df, True)

# bağımlı değişken survived değişkeni ile eksikliğe sahip olan değişkenler ile ilişkisi
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)



###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)





######################################################################################
# 3. ENCODING (Label Encoding, One-Hot Encoding, Rare Encoding)
######################################################################################

# encod: Değişkenlerin temsil şekillerinin değiştirilmesi

######################################################################################
# Label Encoding & Binary Encoding
######################################################################################
# Label Encoding: label ları yeniden kodlamak yani string i 0-1 gibi ifadelerle doldurmaktır
# 2 den fazla sınıfı varsa label encoding denir ve label encoding genel isimlendirmedir.
# Binary Encoding: kategorik değişkenin 2 sınıfı varsa bu 1-0 gibi isimlendirilirse buna binary encoding denir
# Label Encoding > Binary Encoding
df = load()
df.head()
df["Sex"].head()

# alfabetik olarak örneğin  0 dan 5 e alfabetik olarak artan şekilde isimlendirilir.
le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
# nasıl isimlendirildiğini unutursak bunu öğrenmek için:
le.inverse_transform([0, 1])

# fonksiyonlaştırma:
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

# int ve float olmayan colonların ve eşsiz değer sayısı 2 ise onları seç binary_cols de gez tipine bak
# int ve float değilse çıkanları seç
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
# not: nunique metodu eksik değeri bir sınıf olarak görmez
# len(unique) kullanmadık çünkü bu metod değişkenin içerisindeki eksik değeri de sınıf olarak görür

for col in binary_cols:
    label_encoder(df, col)

df.head()

# daha büyük veri setine bakalım
df = load_application_train()
df.shape

# büyük verisetinde binary cols a bakalım
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head()

# label encoder dan binary_cols u getirelim
for col in binary_cols:
    label_encoder(df, col)


df = load()
df["Embarked"].value_counts()
# sinsi problemimiz:
df["Embarked"].nunique() # eksik değerleri göz önünde bulundurmak istemezsek
len(df["Embarked"].unique()) # eksik değerleri göz önünde bulundurmak istersek

####################################################################################################################
# One-Hot Encoding ()
####################################################################################################################
# sınıflar arası fark yokken  label encoding de sorun oluyordu
# ölçüm problemi yapmadan kategorik değişkenin sınıflarını değişkene dönüştürüyoruz
# dummi değişken tuzağı: kukla değişken tuzağı one-hot-encoding yönteminde
# kategorik değişkenleri sınıflarını değişkene dönüştürürken bu yeni oluşturulan değişkene dummi değişken (kukla değişken) denir.
# kukla değişken birbiri üzerinden oluşturulabilir olursa bir ölçme problemi ortaya çıkar.
# bu da yüksek korelasyona sebep olur drop=first diyerek birbiri üzerinden oluşturulma durumu ortadan kaldırılır.
# drop=first diyerek kurtuluruz bu tuzaktan
df = load()
df.head()
df["Embarked"].value_counts() # sınıflar arası fark yok

# bu metod der ki: verilen dataframe den dönüştürülmesi gereken değişkeni ver der ve dönüştürür.
pd.get_dummies(df, columns=["Embarked"]).head()

# değişkenler birbiri üzerinden türetilmesin diye drop=first diyoruz ve ilk sınıf gider
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

# eksik değerlerin de sınıf olarak gelmesini istersek:dummy_na=True deriz:
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

# iki değişkene birlikte bu özelliği uygulayalım yani hem binary encoding hem one-hot encoding yapalım:
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()
# eğer bir değişkenin sınıf sayısı 2 ise drop_first=True dediğimizde direkt olarak binary encod yapılır.

# fonksiyon yazalım:
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

# ohe yi eşsiz değer sayısı 2 den büyük ve 10dan küçük olanlara uygulayalım
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

# df de gez ohe den geçir
one_hot_encoder(df, ohe_cols).head()

df.head()

#################################################################################################
# Rare Encoding
#################################################################################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts() # bu değişkeni inceleyelim

# kategorik değişkenlere ulaşalım
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# fonksiyon yazalım:
# kategorik değişkenlerin sınıflarını ve bu sınıfların oranlarını getirsin
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

# cat_cols için yapalım
for col in cat_cols:
    cat_summary(df, col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

# bu değişkeni gözlemleyelim
df["NAME_INCOME_TYPE"].value_counts()

# target:hedef , rare kategoriler ile bağımlı değişken ilişkisine bakalım
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# rare analizini fonksiyonlaştıralım:
# bağımlı değişkeni ve kategorik değişkeni veriyoruz
# elimizdeki tüm kategorik değişkenler için rare analizi yapıyoruz:
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

# elimizde bol kategorik değişken old rare kullanmalıyız.

#################################################################################################
# 3. Rare encoder'ın yazılması.
#################################################################################################

# fonksiyonlaştıralım:
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

# 0.01 oranının altında kalan sınıflar için rare dönüşümü yapıyoruz
new_df = rare_encoder(df, 0.01)

# rare analizi yapalım:
rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()


#################################################################################################
# Feature Scaling (Özellik Ölçeklendirme)
#################################################################################################

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################
# standart sapma ve ortalama aykırı değerlerden etkilenir.

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()


###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

# merkezi eğilimi ve değişimi göz önünde bulundurmuş oluruz.
rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()
# sonuçları kıyaslayalım:
age_cols = [col for col in df.columns if "Age" in col]

# bu fonksiyonla sayısal değişkenin çeyreklik değerlerini gözlemleyelim ve grafiklerine bakalım
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################
# qcut metodu bir değişkenin değerlerini küçükten büyüğe sıralar ve çeyrek değerlere göre böler
# yaş değişkenini böl 5 parçaya
df["Age_qcut"] = pd.qcut(df['Age'], 5)

##################################################################################################
# FEATURE EXTRACTION (Özellik Çıkarımı)
##################################################################################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

df = load()
df.head()
# nan olanlara 0, dolu olanlara 1 diyelim:
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

# hedef değişkene göre ortalamalarını groupby a alalım
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

# oran testi uygulayalım:
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Yorum: proportions_ztest te H_0 hipotezi p1 ve p2 pranları arasında fark yoktur der.
# H_0 < 0.05 old reddedilir.
# istatistiksel olarak aralarında anlamlı bir fark vardır.

# SibSp yakın akrabalıkları, Parch uzak akrabalıkları
# bu ikisinin toplamları 0 değilse NEW_IS_ALONE değişkeni türet ve no yaz
# 0 ise yes yaz.

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

# bu yeni değişkene göre hedef değişken analizi yap
df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

# hipotez testi yapalım:
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# yorum: p< 0.05 old h_0 reddedilir.

#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################

df.head()

###################
# Letter Count: harf sayma
###################

# isimlerdeki harfler sayılır ve bu yeni değişkene atanır.
df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count: kelime sayma: sayılır ve bu yeni değişkene atanır.
###################
# isimlerdeki kelimeler
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Özel Yapıları Yakalamak
###################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})

###################
# Regex ile Değişken Türetmek
###################

df.head()

df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#######################################################################################################
# Date Değişkenleri Üretmek
########################################################################################################

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

# date tipini dönüştürelim:
dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month


# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

# date


#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################
# değişkenlerin birbirleri ile etkileşimleri

df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale' # genç erkekler
# yaşı 21 ile 50 arasındaki erkekler
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
#  olgun erkekler yaşı 50den büyük olan
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
# genç kadınlar
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
# olgun kadınlar yaşı 21 ile 50 arasındaki
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
# olgun kadınlar
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()


########################################################################################################
# UYGULAMA: Titanic Uçtan Uca Feature Engineering & Data Preprocessing
########################################################################################################

df = load()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#######################################################################################################
# 2. Outliers (Aykırı Değerler)
#######################################################################################################
# Aykırı değer var mı
for col in num_cols:
    print(col, check_outlier(df, col))
# eşik değerler ile aykırı değerleri yer değiştirelim
for col in num_cols:
    replace_with_thresholds(df, col)
# tekrar aykırı değer var mı diye bakalım
for col in num_cols:
    print(col, check_outlier(df, col))

#########################################################################################################
# 3. Missing Values (Eksik Değerler)
#########################################################################################################

missing_values_table(df)
# kabin değişkeninden kalıcı şekilde kurtulalım
df.drop("CABIN", inplace=True, axis=1)
# kaldırılacak sütunlar
remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

# eksik değerleri median ile dolduralım
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

# yeni değişken oluşturalım
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

# Tipi object olan ve eşsiz değer sayısı 10dan küçük olanları mod ile doldur
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#######################################################################################################
# 4. Label Encoding
#######################################################################################################
# kategorik olanları seçelim eşsiz sınıf sayısı 2 olanları
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


#######################################################################################################
# 5. Rare Encoding
#######################################################################################################

rare_analyser(df, "SURVIVED", cat_cols)


df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

######################################################################################################
# 6. One-Hot Encoding
#######################################################################################################
# veri setindeki tüm kategorik değişkenleri çevirmeliyiz.
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)
# kullanışsız sütunlar : kategorik değişkenlerin sınıf sayısını gözlem sayısına böl,
# herhangi bir tanesinde 0.01 den küçük olan iki sınıflı bir kategorik değişken varsa getir
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# silmek için:
# df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 7. Standart Scaler
#############################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape


#############################################
# 8. Model
#############################################
# bağımlı değişken
y = df["SURVIVED"]
# bağımsız değişkenler
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)
# train ve test olarak 2 ye ayırıyoruz,
# train üzerinde model kurucaz, test seti ile test edicez
# tüm veriyi görüp ezber yapmasın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier
# modeli kuralım
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
# doğruluk oranı skoru
accuracy_score(y_pred, y_test)
# başarı oranı % 80


#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
# başarı oranı % 70

# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)


