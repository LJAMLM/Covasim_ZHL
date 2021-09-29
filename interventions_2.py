import datetime
import time
import numpy as np
import covasim as cv
import pandas as pd
import pylab as pl
import sciris as sc
from random import sample

cv.options.set(dpi=100, show=False, close=True, verbose=0)

pop_kakuma_may = 156232
pop_kalobeyei_may = 38081
pop_kakuma_kalobeyei_may = 194313

# Kept in case we want to treat Kakuma and Kalobeyei separately
kakuma_pop_survey = {
    '0-4': 20303,
    '5-11': 33641,
    '12-17': 30947,
    '18-27': 28252,
    '28-37': 21129,
    '38-47': 14482,
    '48-57': 4986,
    '58-67': 1899,
    '68-77': 594,
}

kalobeyei_pop_survey = {
    '0-4': 6286,
    '5-11': 11259,
    '12-17': 7394,
    '18-27': 4508,
    '28-37': 4889,
    '38-47': 2497,
    '48-57': 763,
    '58-67': 416,
    '68-77': 69,
}

# Treat Kakuma and Kalobeyei together
kakuma_kalobeyei_pop_survey = {
    '0-4': 26589,
    '5-11': 44900,
    '12-17': 38341,
    '18-27': 31724,
    '28-37': 27500,
    '38-47': 16724,
    '48-57': 5517,
    '58-67': 2414,
    '68-77': 603,
}

kakuma_hhsize = 6.3     # mentioned in UNHCR kakuma pdf
kalobeyei_hhsize = 5.9  # mentioned in UNHCR kalobeyei pdf
kakuma_kalobeyei_hhsize = (kakuma_hhsize * pop_kakuma_may + kalobeyei_hhsize * pop_kalobeyei_may) / \
                          (pop_kakuma_may + pop_kalobeyei_may)  # 6.222

# cv.data.country_age_data.data['Kakuma'] = kakuma_pop_survey
# cv.data.country_age_data.data['Kalobeyei'] = kalobeyei_pop_survey
# cv.data.household_size_data.data['Kakuma'] = kakuma_hhsize
# cv.data.household_size_data.data['Kalobeyei'] = kalobeyei_hhsize

cv.data.country_age_data.data['Kakuma_Kalobeyei'] = kakuma_kalobeyei_pop_survey
cv.data.household_size_data.data['Kakuma_Kalobeyei'] = kakuma_kalobeyei_hhsize

""" 
Per-contact transmission probability can also be deduced as follows, but this is excluded in this study, due to the
many assumptions made. 
"""
# # Range of R0 in Kenya = [1.78, 3.46] (Brand et al, Apr 2020)
# # R0 = 2.76 in Kenya (Ogana et al, March 2021)
# R0 = 2.76
# # Median duration infectiousness for asymptomatic cases: 6.5-9.5 days (Byrne et al, July 2020)
# infectious_period_asymptomatic = 8
# # Median duration of shedding infectious virus is 8 days post onset of symptoms (van Kampen et al, Jan 2021)
# viral_shedding_after_symptom_onset = 8
# # Infectious 1-3 days before symptom onset!
# viral_shedding_before_symptom_onset = 2
# infectious_period_symptomatic = viral_shedding_before_symptom_onset + viral_shedding_after_symptom_onset
# # Assume 70-30 for symptomatic and asymptomatic cases (best current estimate according to CDC)
# avg_duration_infectiousness = infectious_period_symptomatic * 0.7 + infectious_period_asymptomatic * 0.3
# # Mean number of contacts per person per day in Kenya = 17.7 (Kiti et al, August 2015)
# contact_rate_S_I = 17.7
# # Rewritten formula of R0 by Jones, 2007
# beta = R0 / (contact_rate_S_I * avg_duration_infectiousness)

monthly_data = pd.DataFrame({
    'date': ['2020-05-31', '2020-06-30', '2020-07-31', '2020-08-31', '2020-09-30', '2020-10-31', '2020-11-30',
             '2020-12-31', '2021-01-31', '2021-02-28', '2021-03-31', '2021-04-30', '2021-05-31'],
    'cum_diagnoses': [1, 2, 33, 81, 191, 324, 435,
                      486, 516, 532, 789, 935, 1006],
    'new_diagnoses': [1, 1, 31, 48, 110, 133, 111,
                      51, 30, 16, 257, 146, 71],
    'cum_tests': [22, 88, 777, 1854, 2636, 3714, 5260,
                  6198, 6609, 6976, 7608, 9060, 10432],
    'new_tests': [22, 66, 689, 1077, 782, 1078, 1546,
                  938, 411, 367, 632, 1452, 1372],
    'cum_recovered': [0, 1, 11, 50, 50, 50, 50,
                      50, 50, 50, 50, 50, 50],
    'cum_deaths': [0, 0, 0, 1, 4, 7, 10,
                   10, 10, 10, 12, 12, 12],
    'new_deaths': [0, 0, 0, 1, 3, 3, 3,
                   0, 0, 0, 2, 0, 0],
})

monthly_data.to_csv('monthly_data.csv')
tn_data = cv.test_num('data', symp_test=100, quar_test=2, sensitivity=0.8, test_delay=1, start_day=7)


class StoreSEIR(cv.Analyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # This is necessary to initialize the class properly
        self.t = []
        self.S = []
        self.E = []
        self.I = []
        self.R = []
        return

    def apply(self, sim):
        ppl = sim.people  # Shorthand
        self.t.append(sim.t)
        self.S.append(ppl.susceptible.sum())
        self.E.append(ppl.exposed.sum() - ppl.infectious.sum())
        self.I.append(ppl.infectious.sum())
        self.R.append(ppl.recovered.sum() + ppl.dead.sum())
        return

    def plot(self):
        pl.figure()
        pl.plot(self.t, self.S, label='S')
        pl.plot(self.t, self.E, label='E')
        pl.plot(self.t, self.I, label='I')
        pl.plot(self.t, self.R, label='R')
        pl.legend()
        pl.xlabel('Day')
        pl.ylabel('People')
        sc.setylim()  # Reset y-axis to start at 0
        sc.commaticks()  # Use commas in the y-axis labels
        pl.tight_layout()
        cv.savefig('seir.png', bbox_inches="tight")
        pl.show()
        return


def plot_population():
    sim = cv.Sim(pars=pars, label="Default")
    sim.initialize()            # Create people
    bins = np.arange(0, 81)     # Oldest person is only 77
    min_age = min(bins)
    max_age = max(bins)
    edges = np.append(bins, np.inf)  # Add an extra bin to end to turn them into edges
    age_counts = np.histogram(sim.people.age, edges)[0]

    color = [0.1, 0.1, 0.1]     # Color for the age distribution
    offset = 0.5                # For ensuring the full bars show up
    gridspace = 10              # Spacing of gridlines
    zorder = 10                 # So plots appear on top of gridlines
    alpha = 0.6
    width = 1.0

    pl.bar(bins, age_counts, color=color, alpha=alpha, width=width, zorder=zorder)
    pl.xlim([min_age - offset, max_age + offset])
    pl.xticks(np.arange(0, max_age + 1, gridspace))
    pl.grid(True)
    pl.xlabel('Age')
    pl.ylabel('Number of people')
    pl.title(f'Age distribution ({len(sim.people):n} people total)')
    cv.savefig('pop_dist.png')
    pl.show()
    return


# Implementation of 95% CI
def conf_int(lst, n_runs):
    avg = np.mean(lst)
    std = np.std(lst)
    lb = round(avg - 1.96 * std / np.sqrt(n_runs), 1)
    ub = round(avg + 1.96 * std / np.sqrt(n_runs), 1)
    ci = [lb, ub]
    return ci


# Save the statistics of the MultiSim summary to a list
def msim_to_lst(_msim, n_runs, stat):
    return [_msim.sims[i].summary[stat] for i in range(n_runs)]


def msim_results(_msim, n_runs):
    cum_infections = [i * nr_sectors for i in msim_to_lst(_msim, n_runs, 'cum_infections')]
    cum_severe = [i * nr_sectors for i in msim_to_lst(_msim, n_runs, 'cum_severe')]
    cum_critical = [i * nr_sectors for i in msim_to_lst(_msim, n_runs, 'cum_critical')]
    cum_deaths = [i * nr_sectors for i in msim_to_lst(_msim, n_runs, 'cum_deaths')]
    cum_diagnoses = [i * nr_sectors for i in msim_to_lst(_msim, n_runs, 'cum_diagnoses')]
    cum_quarantined = [i * nr_sectors for i in msim_to_lst(_msim, n_runs, 'cum_quarantined')]
    cum_tests = [i * nr_sectors for i in msim_to_lst(_msim, n_runs, 'cum_tests')]

    r_eff = msim_to_lst(_msim, n_runs, 'r_eff')
    test_yield = msim_to_lst(_msim, n_runs, 'test_yield')
    prevalence = msim_to_lst(_msim, n_runs, 'prevalence')

    ci_infected = conf_int(cum_infections, n_runs)
    ci_severe = conf_int(cum_severe, n_runs)
    ci_critical = conf_int(cum_critical, n_runs)
    ci_dead = conf_int(cum_deaths, n_runs)

    print('Current parameter value:', varying_param)
    print('Average cumulative infected cases:', round(np.mean(cum_infections)))
    print('Average cumulative severe cases:', round(np.mean(cum_severe)))
    print('Average cumulative critical cases:', round(np.mean(cum_critical)))
    print('Average cumulative dead cases:', round(np.mean(cum_deaths)))
    print('Range of total infections:', min(cum_infections), '-', max(cum_infections))
    print('Range of total deaths:', min(cum_deaths), '-', max(cum_deaths))

    print('95% confidence interval infections:', ci_infected)
    print('95% confidence interval severe:', ci_severe)
    print('95% confidence interval critical:', ci_critical)
    print('95% confidence interval dead:', ci_dead)

    print('Average cumulative diagnosed cases:', np.mean(cum_diagnoses))
    print('Average cumulative quarantined people:', np.mean(cum_quarantined))
    print('Average cumulative tests:', np.mean(cum_tests))
    print('Average effective reproduction number:', np.mean(r_eff))
    print('Average testing yield:', np.mean(test_yield))
    print('Average prevalence:', np.mean(prevalence))
    return


def run_same_beta(n_runs):
    default_sim = cv.Sim(pars=pars)
    msim_default = cv.MultiSim(default_sim)
    msim_default.run(n_runs=n_runs)
    msim_results(msim_default, n_runs)
    msim_default.reduce()
    msim_default.plot(fig_args=dict(figsize=(9, 9)), do_save=True, fig_path=r'same_beta.png')
    return


def run_diff_beta(n_runs):
    betas = np.linspace(0.01, 0.05, n_runs)
    beta_sims = []
    for beta in betas:
        beta_sim = cv.Sim(pars=pars, beta=beta, label=f'Beta = {beta}')
        beta_sims.append(beta_sim)
    msim_beta = cv.MultiSim(beta_sims)
    msim_beta.run()
    msim_results(msim_beta, n_runs)
    for _sim in msim_beta.sims:
        print(_sim.brief())
    msim_beta.plot(to_plot=['new_infections', 'cum_infections', 'cum_deaths'], fig_args=dict(figsize=(9, 9)),
                   do_save=True, fig_path=r'diff_beta.png')
    return


def run_orig_sim_analyzer():
    orig_sim = cv.Sim(pars=pars, label='Default', analyzers=cv.age_histogram())
    orig_sim.run()

    age_hist = orig_sim.get_analyzer()
    age_hist.plot(fig_args=dict(figsize=(10, 9)))
    cv.savefig('age_hist.png')

    tt = orig_sim.make_transtree()
    tt.plot(fig_args=dict(figsize=(10, 3)))
    cv.savefig('tt1.png', bbox_inches='tight')
    tt.plot_histograms()
    cv.savefig('tt2.png', bbox_inches='tight')

    sim_class = cv.Sim(pars=pars, analyzers=StoreSEIR(label='seir'))
    sim_class.run()
    seir = sim_class.get_analyzer('seir')  # Retrieve by label
    seir.plot()
    return


if __name__ == '__main__':
    # Use this loop for iterating over the possible values a single parameter can take (mainly for sensitivity analysis)
    for varying_param in sample(range(0, 99999999), 10):

        default_beta = 0.016
        multiplication_factor_beta = 1
        multiplication_factor_imports = 1
        underreporting_factor = 10
        use_of_sectoring = False    # Essentially 'turn sectoring on'
        sector_subdivisions = 1     # Split villages and zones further into subdivisions
        nr_sectors = 16 * sector_subdivisions if use_of_sectoring else 1
        new_arrivals_per_day = 6643 / 365
        prevalence_new_arrivals = 1 / 100

        # Try for Kakuma separately, Kalobeyei separately and combination of both
        pars = {
            'pop_size': pop_kakuma_kalobeyei_may / nr_sectors,
            'location': 'Kakuma_Kalobeyei',
            'pop_infected': 1 * underreporting_factor / nr_sectors,
            'pop_type': 'hybrid',
            'beta': default_beta * multiplication_factor_beta,
            'n_imports': new_arrivals_per_day * prevalence_new_arrivals * multiplication_factor_imports / nr_sectors,
            'use_waning': True,
            'start_day': '2020-05-25',
            'end_day': '2021-05-25',
            'contacts': {'s': 60, 'w': 10, 'c': 40},
            'beta_layer': dict(h=3.0, s=0.6, w=0.6, c=0.3),
            'iso_factor': dict(h=0.3, s=0.1, w=0.1, c=0.1),
            'quar_factor': dict(h=0.6, s=0.2, w=0.2, c=0.2),
            'n_beds_hosp': 16 / nr_sectors,
            'n_beds_icu': 0,
            'rand_seed': varying_param,     # use 2 or 9 for default, 293 for intervention, 2 for different beta!
        }

        start_time = time.time()
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))

        # plot_population()
        # run_same_beta(n_runs=50*nr_sectors)  # Anaconda prompt
        # run_diff_beta(n_runs=5)              # Anaconda prompt
        # run_orig_sim_analyzer()

        start_day = 7

        # Assume all contact layer closures are followed without violations
        school_closure_compliance = 1
        community_closure_compliance = 1
        work_closure_compliance = 1

        # School children are split up into 2 groups and go to school every alternate day
        percentage_school_open = 1/2 / school_closure_compliance
        # Cannot close more than 40% of the community
        percentage_community_open = 3/5 / community_closure_compliance
        # Try to keep as much work open as possible, let 80% go to work every day
        percentage_work_open = 4/5 / work_closure_compliance

        close_schools = cv.clip_edges(days=[start_day], changes=[percentage_school_open], layers='s')
        close_community = cv.clip_edges(days=[start_day], changes=[percentage_community_open], layers='c')
        close_work = cv.clip_edges(days=[start_day], changes=[percentage_work_open], layers='w')

        # N95 masks reduce transmission by 97%, surgical masks by 84% and homemade masks by 67% (Chen et al, Apr 2021)
        n95_efficacy = 0.97
        surg_efficacy = 0.84
        home_efficacy = 0.67
        # Assume availability of N95, surgical and homemade masks, approx 291700 were donated while 72000 were homemade
        home_available = 0.20
        donated_available = 1 - home_available                  # 0.80
        n95_frac = 0.05
        n95_available = n95_frac * donated_available            # 0.04
        surg_available = (1 - n95_frac) * donated_available     # 0.76
        masks = 1 - (n95_efficacy * n95_available + surg_efficacy * surg_available + home_efficacy * home_available)
        # ...and this eventually leads to a relative transmissibility of 0.1888.
        # Handwashing efficacy: 16% reduction for acute respiratory infections (hand hygiene meta-analysis)
        hygiene = 1 - 0.16  # 0.84
        # Physical distancing, proper ventilation etc.
        distance_reduction_factor = 1
        distance_school = percentage_school_open / distance_reduction_factor
        distance_work = percentage_community_open / distance_reduction_factor
        distance_community = percentage_work_open / distance_reduction_factor
        # Assume 100% compliance for hygiene, 75% compliance for masks and distance (could change this per layer later)
        mask_distance_compliance = 0.75
        hygiene_compliance = 1  # Cannot reduce this below 0.84, as this is the effect of hygiene
        # Changes per layer
        household_changes = hygiene / hygiene_compliance
        community_changes = hygiene * masks * distance_community / (mask_distance_compliance * hygiene_compliance)
        work_changes = hygiene * masks * distance_work / (mask_distance_compliance * hygiene_compliance)
        school_changes = hygiene * masks * distance_school / (mask_distance_compliance * hygiene_compliance)

        reduce_beta_workplace = cv.change_beta(days=[start_day], changes=[work_changes], layers='w')
        reduce_beta_schools = cv.change_beta(days=[start_day], changes=[school_changes], layers='s')
        reduce_beta_community = cv.change_beta(days=[start_day], changes=[community_changes], layers='c')
        reduce_beta_household = cv.change_beta(days=[start_day], changes=[household_changes], layers='h')

        # PCR test sensitivity decreases per day of symptom onset, >90% in first 5 days (Miller, Aug 2020)
        # RAD 79.6% sensitivity, with RT-PCR as reference point (Albert, March 2021)
        pcr_sensitivity = 0.9
        rad_sensitivity = 0.796 * pcr_sensitivity
        pcr_ratio = 0.4     # Indicated by UNHCR
        rad_ratio = 0.6     # Indicated by UNHCR
        test_prob = 1 / 3   # Average person tests after 4-5 days of symptoms (UNHCR) --> awareness may lead to ~3 days
        pcr_prob = pcr_ratio * test_prob
        rad_prob = rad_ratio * test_prob
        quar_pcr_prob = 2 * pcr_prob
        quar_rad_prob = 2 * rad_prob
        ideal_quar_period = 14
        quar_compliance = 0.75
        true_quar_period = ideal_quar_period * quar_compliance
        tp = cv.test_prob(symp_prob=pcr_prob, asymp_prob=0, symp_quar_prob=quar_pcr_prob,
                          asymp_quar_prob=pcr_prob/5, start_day=start_day, test_delay=1, sensitivity=pcr_sensitivity)
        tp2 = cv.test_prob(symp_prob=rad_prob, asymp_prob=0, symp_quar_prob=quar_rad_prob,
                           asymp_quar_prob=rad_prob/5, start_day=start_day, test_delay=0, sensitivity=rad_sensitivity)
        # Probabilities and tracing time (2-3 days) as given by UNHCR --> try reducing trace time to 1-2 days
        ct = cv.contact_tracing(start_day=start_day, trace_probs=dict(h=0.9, s=0.2, w=0.9, c=0.6),
                                trace_time=dict(h=1.5, s=1.5, w=1.5, c=1.5), capacity=30, quar_period=true_quar_period)

        # An additional 35 hospital beds were supplied throughout the pandemic (UNHCR)
        # We assume that they were distributed uniformly, so approximately 1 bed per 10 days
        days = range(365)
        vals = [1 if i % 10 == 0 else 0 for i in days]
        more_beds = cv.dynamic_pars(n_beds_hosp=dict(days=days, vals=vals), do_plot=False)


        def shielding(sim):
            elderly = sim.people.age >= 60
            if sim.t == 7:
                sim.people.rel_sus[elderly] = 0.1
            # # Uncomment to turn off shielding after certain day
            # elif sim.t == 90:
            #     sim.people.rel_sus[elderly] = 1.0
            return


        s1 = cv.Sim(pars=pars, label='Default')
        s2 = cv.Sim(pars=pars, interventions=[close_schools, close_community, close_work], label='Partial closures')
        s3 = cv.Sim(pars=pars, interventions=[reduce_beta_community, reduce_beta_workplace, reduce_beta_schools,
                                              reduce_beta_household], label='Transmission reduction')
        s4 = cv.Sim(pars=pars, interventions=[tp, tp2, ct], label='Contact tracing')
        s5 = cv.Sim(pars=pars, interventions=shielding, label='Shielding')
        s6 = cv.Sim(pars=pars, interventions=[more_beds, reduce_beta_community, reduce_beta_workplace,
                                              reduce_beta_schools, reduce_beta_household, tp, tp2, ct, close_schools,
                                              close_community, close_work], label='Combo')

        """ Run the default sim with one random seed """
        # s1.run()
        # s1.plot_result('r_eff', do_save=True, fig_path=r'r_eff_default.png').show()  # R_0 as high as 6.47 in Wuhan
        # print(s1.brief)
        # print(s1.summary)
        # default_plot = s1.plot(do_save=True, fig_path=r'default.png')
        # default_plot.show()
        # s1.to_excel('default.xlsx')
        # s1_df = pd.read_excel('default.xlsx')

        """ Run separate interventions (Anaconda prompt) """
        # msim_separate = cv.MultiSim([s1, s2, s3, s4, s5])
        # msim_separate.run()
        # print(msim_separate.summarize())
        # separate_plot = msim_separate.plot(
        #     to_plot=['new_infections', 'cum_infections', 'cum_deaths'],
        #     fig_args=dict(figsize=(9, 9)),
        #     do_save=True, fig_path=r'separate.png')
        # separate_plot.show()

        """ Run combined interventions with one random seed """
        # s6.run()
        # s6.plot_result('r_eff', do_save=True, fig_path=r'r_eff_intervention.png').show()
        # print(s6.brief)
        # print(s6.summary)
        # combo_plot = s6.plot(
        #     to_plot=['new_infections', 'cum_infections', 'new_diagnoses', 'cum_deaths', 'cum_tests',
        #              'new_quarantined'],
        #     fig_args=dict(figsize=(12, 9)),
        #     do_save=True, fig_path=r'combo.png')
        # combo_plot.show()
        # s6.to_excel('interventions.xlsx')
        # s6_df = pd.read_excel('interventions.xlsx')

        """ Run combined interventions multiple times (Anaconda prompt) """
        # n_runs = 10 * nr_sectors
        # msim_final_int = cv.MultiSim(s6)
        # msim_final_int.run(n_runs=n_runs)
        # print(msim_final_int.summarize())
        # msim_results(msim_final_int, n_runs)
        # msim_final_int.reduce()
        # msim_final_int.plot(
        #     to_plot=['new_infections', 'cum_infections', 'new_diagnoses', 'cum_deaths', 'cum_tests',
        #              'new_quarantined'],
        #     fig_args=dict(figsize=(12, 9)),
        #     do_save=True, fig_path=r'final_intervention.png')

        print("--- %s seconds ---" % (time.time() - start_time))
