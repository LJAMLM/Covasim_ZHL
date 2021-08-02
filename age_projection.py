import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

all_waves = pd.read_csv("Anonymized HH Data v1.1.csv", low_memory=False)
all_waves.drop_duplicates(subset=["hhid", "s2_q8_migrresidence"], inplace=True)
all_waves.dropna(subset=["s2_q2_age"], inplace=True)

kaku_all_waves = all_waves[all_waves["s2_q8_migrresidence"] == "Kakuma camp"]
kalo_all_waves = all_waves[all_waves["s2_q8_migrresidence"] == "Kalobeyei settlement"]
both_places_all_waves = all_waves[(all_waves["s2_q8_migrresidence"] == "Kakuma camp") |
                                  (all_waves["s2_q8_migrresidence"] == "Kalobeyei settlement")]

kakuma_pop = {
    '0-4': 20303,
    '5-11': 33641,
    '12-17': 30947,
    '18-59': 69158,
    '60+': 2183,
}

kalobeyei_pop = {
    '0-4': 6286,
    '5-11': 11259,
    '12-17': 7394,
    '18-59': 12794,
    '60+': 348,
}

kakuma_kalobeyei_pop = {
    '0-4': 26589,
    '5-11': 44900,
    '12-17': 38341,
    '18-59': 81952,
    '60+': 2531,
}

last2pairs_kaku = {k: kakuma_pop[k] for k in list(kakuma_pop)[-2:]}
last2pairs_kalo = {k: kalobeyei_pop[k] for k in list(kalobeyei_pop)[-2:]}
last2pairs_both = {k: kakuma_kalobeyei_pop[k] for k in list(kakuma_kalobeyei_pop)[-2:]}

nr_adults_kaku = sum(last2pairs_kaku.values())
nr_adults_kalo = sum(last2pairs_kalo.values())
nr_adults_both = sum(last2pairs_both.values())

oldest_kaku = int(kaku_all_waves["s2_q2_age"].max())
oldest_kalo = int(kalo_all_waves["s2_q2_age"].max())
oldest_both = int(both_places_all_waves["s2_q2_age"].max())

print('Kakuma')
for i in range(18, oldest_kaku + 1, 10):
    print(i, i + 9, round(len(kaku_all_waves[(kaku_all_waves["s2_q2_age"] >= i) &
                                             (kaku_all_waves["s2_q2_age"] <= i + 9)]) /
                          len(kaku_all_waves) * nr_adults_kaku))

print('Kalobeyei')
for i in range(18, oldest_kalo + 1, 10):
    print(i, i + 9, round(len(kalo_all_waves[(kalo_all_waves["s2_q2_age"] >= i) &
                                             (kalo_all_waves["s2_q2_age"] <= i + 9)]) /
                          len(kalo_all_waves) * nr_adults_kalo))

print('Both')
for i in range(18, oldest_both + 1, 10):
    print(i, i + 9, round(len(both_places_all_waves[(both_places_all_waves["s2_q2_age"] >= i) &
                                                    (both_places_all_waves["s2_q2_age"] <= i + 9)]) /
                          len(both_places_all_waves) * nr_adults_both))

sns.displot(both_places_all_waves["s2_q2_age"])
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution Kakuma & Kalobeyei", fontsize=14)
plt.tight_layout()
plt.savefig('age_dist_kakuma_kalobeyei.png', bbox_inches='tight')
plt.show()
