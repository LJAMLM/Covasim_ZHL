import numpy as np
import covasim as cv
import pandas as pd
import pylab as pl
from functools import reduce
import sciris as sc

cv.options.set(dpi=100, show=False, close=True, verbose=0)

pop_kakuma_may = 156232
pop_kalobeyei_may = 38081
pop_kakuma_kalobeyei_may = 194313

# Treat Kakuma and Kalobeyei separately
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

kakuma_hhsize = 6.3  # UNHCR kakuma pdf
kalobeyei_hhsize = 5.9  # UNHCR kalobeyei pdf
kakuma_kalobeyei_hhsize = (kakuma_hhsize * pop_kakuma_may + kalobeyei_hhsize * pop_kalobeyei_may) / \
                          (pop_kakuma_may + pop_kalobeyei_may)  # 6.222

cv.data.country_age_data.data['Kakuma'] = kakuma_pop_survey
cv.data.country_age_data.data['Kalobeyei'] = kalobeyei_pop_survey
cv.data.country_age_data.data['Kakuma_Kalobeyei'] = kakuma_kalobeyei_pop_survey

cv.data.household_size_data.data['Kakuma'] = kakuma_hhsize
cv.data.household_size_data.data['Kalobeyei'] = kalobeyei_hhsize
cv.data.household_size_data.data['Kakuma_Kalobeyei'] = kakuma_kalobeyei_hhsize

underreporting_factor = 10
nr_sectors = 1

# Try for Kakuma separately, Kalobeyei separately and combination of both
pars = {
    'pop_size': pop_kakuma_kalobeyei_may / nr_sectors,
    'location': 'Kakuma_Kalobeyei',
    'pop_infected': 1 * underreporting_factor,
    'pop_type': 'hybrid',
    'beta': 0.030,
    'n_imports': 0.18985,
    'use_waning': False,
    'start_day': '2020-05-25',
    'end_day': '2020-10-25',
    'contacts': {'s': 60, 'w': 10, 'c': 40},  # 30 mainly for children with disabilities, more than 100 in class
    # 'n_beds_hosp': 20,
    # 'n_beds_icu': 0,
}

monthly_data = pd.DataFrame({
    'date': ['2020-05-31', '2020-06-30', '2020-07-31', '2020-08-31', '2020-09-30', '2020-10-31', '2020-11-30',
             '2020-12-31', '2021-01-31', '2021-02-31', '2021-03-31', '2021-04-31'],
    'cum_diagnoses': [1, 2, 33, 81, 191, 324, 435,
                      486, 516, 532, 789, 935],
    'new_diagnoses': [1, 1, 31, 48, 110, 133, 111,
                      51, 30, 16, 257, 146],
    'cum_tests': [22, 88, 777, 1854, 2636, 3714, 5260,
                  6198, 6609, 6976, 7608, 9060],
    'new_tests': [22, 66, 689, 1077, 782, 1078, 1546,
                  938, 411, 367, 632, 1452],
    'cum_recovered': [0, 1, 11, 50, 50, 50, 50,
                      50, 50, 50, 50, 50],
    'cum_deaths': [0, 0, 0, 1, 4, 7, 10,
                   10, 10, 10, 12, 12],
    'new_deaths': [0, 0, 0, 1, 3, 3, 3,
                   0, 0, 0, 2, 0],
})

monthly_data.to_csv('monthly_data.csv')
tn_data = cv.test_num('data', symp_test=100)

reduce_beta_workplace = cv.change_beta(days=[7], changes=[0.5], layers='w')
reduce_beta_schools = cv.change_beta(days=[7], changes=[0.5], layers='s')
reduce_beta_community = cv.change_beta(days=[7], changes=[0.5], layers='c')

close_schools = cv.clip_edges(days=[7], changes=[0.33], layers='s')
close_community = cv.clip_edges(days=[7], changes=[0.15], layers='c')
close_work = cv.clip_edges(days=[7], changes=[0.33], layers='w')

# PCR test sensitivity decreases per day of symptom onset, >90% in first 5 days (Miller, Aug 2020)
# RT PCR test sensitivity: 0.777 (Padhye, April 2020)
# rapid antigen test: 81% of PCR
tp = cv.test_prob(symp_prob=0.33, symp_quar_prob=0.33, start_day=7, test_delay=2, sensitivity=0.9, loss_prob=0.05)
tp2 = cv.test_prob(symp_prob=0.33, symp_quar_prob=0.33, start_day=180, test_delay=1, sensitivity=0.729, loss_prob=0.05)
ct = cv.contact_tracing(trace_probs=dict(h=0.9, s=0.3, w=0.4, c=0.1), trace_time=dict(h=0, s=1, w=1, c=2))

more_beds = cv.dynamic_pars(n_beds_hosp=dict(days=7, vals=100000))


def shielding(sim):
    if sim.t == sim.day(day=7):
        elderly = sim.people.age >= 60
        sim.people.rel_sus[elderly] = 0.25


def vaccinate_by_age(sim):
    young = cv.true(sim.people.age < 40)
    middle = cv.true((sim.people.age >= 40) * (sim.people.age < 60))
    old = cv.true(sim.people.age >= 60)
    individuals = sim.people.uid
    values = np.ones(len(sim.people))
    values[young] = 0.10
    values[middle] = 0.50
    values[old] = 0.90
    output = dict(inds=individuals, vals=values)
    return output


# vaccine = cv.simple_vaccine(days=[30, 60], cumulative=[0.5, 0.4], prob=0.6, rel_sus=0.5, rel_symp=0.1)
# vaccine2 = cv.simple_vaccine(days=30, rel_sus=0.8, rel_symp=0.06, subtarget=vaccinate_by_age)


def inf_thresh(self, sim, thresh=2000):
    """ Dynamically define on and off days for a beta change -- use self like it's a method """

    # Meets threshold, activate
    if sim.people.infectious.sum() > thresh:
        if not self.active:
            self.active = True
            self.t_on = sim.t
            self.plot_days.append(self.t_on)

    # Does not meet threshold, deactivate
    else:
        if self.active:
            self.active = False
            self.t_off = sim.t
            self.plot_days.append(self.t_off)

    return [self.t_on, self.t_off]


# # Set up the intervention
# on = 0.2  # Beta less than 1 -- intervention is on
# off = 1.0  # Beta is 1, i.e. normal -- intervention is off
# changes = [on, off]
# plot_args = dict(label='Dynamic beta', show_label=True, line_args={'c': 'blue'})
# db = cv.change_beta(days=inf_thresh, changes=changes, **plot_args)
#
# # Set custom properties
# db.t_on = np.nan
# db.t_off = np.nan
# db.active = False
# db.plot_days = []


def dynamic_imports(sim):
    if sim.t == 15:
        sim['n_imports'] = 100
    elif sim.t == 30:
        sim['n_imports'] = 0
    return


class ProtectElderly(cv.Intervention):

    def __init__(self, start_day=None, end_day=None, age_cutoff=60, rel_sus=0.0, *args, **kwargs):
        super().__init__(**kwargs)  # NB: This line must be included
        self.start_day = start_day
        self.end_day = end_day
        self.age_cutoff = age_cutoff
        self.rel_sus = rel_sus
        return

    def initialize(self, sim):
        super().initialize()  # NB: This line must also be included
        self.start_day = sim.day(self.start_day)  # Convert string or dateobject dates into an integer number of days
        self.end_day = sim.day(self.end_day)
        self.days = [self.start_day, self.end_day]
        self.elderly = sim.people.age > self.age_cutoff  # Find the elderly people here
        self.exposed = np.zeros(sim.npts)  # Initialize results
        self.tvec = sim.tvec  # Copy the time vector into this intervention
        return

    def apply(self, sim):
        self.exposed[sim.t] = sim.people.exposed[self.elderly].sum()
        # Start the intervention
        if sim.t == self.start_day:
            sim.people.rel_sus[self.elderly] = self.rel_sus
        # End the intervention
        elif sim.t == self.end_day:
            sim.people.rel_sus[self.elderly] = 1.0
        return

    def plot(self):
        pl.figure()
        pl.plot(self.tvec, self.exposed)
        pl.xlabel('Day')
        pl.ylabel('Number infected')
        pl.title('Number of elderly people with active COVID')
        return


# protect = ProtectElderly(start_day=7, rel_sus=0.25)  # Create intervention


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


def check_88(simulation):
    people_who_are_88 = simulation.people.age.round() == 88  # Find everyone who's aged 88 (to the nearest year)
    people_exposed = simulation.people.exposed  # Find everyone who's infected with COVID
    people_who_are_88_with_covid = cv.true(
        people_who_are_88 * people_exposed)  # Multiplication is the same as logical "and"
    n = len(people_who_are_88_with_covid)  # Count how many people there are
    if n:
        print(f'Oh no! {n} people aged 88 have covid on timestep {simulation.t} {"🤯" * n}')
    return


def plot_population():
    sim = cv.Sim(pars=pars, label="Default")
    sim.initialize()  # Create people
    fig = sim.people.plot(fig_args=dict(figsize=(30, 15)))  # Show statistics of the people
    cv.savefig('pop_dist.png')
    fig.show()
    return


def run_same_beta():
    n_runs = 5
    default_sim = cv.Sim(pars=pars)
    msim_default = cv.MultiSim(default_sim)
    msim_default.run(n_runs=n_runs)
    for _sim in msim_default.sims:
        print(_sim.brief())
    msim_default.mean()
    default_beta_plot = msim_default.plot(
        # to_plot=['new_infections', 'cum_infections', 'cum_deaths'],
        fig_args=dict(figsize=(9, 9)), do_save=True, fig_path='same_beta.png')
    default_beta_plot.show()
    dead_lst = [msim_default.sims[i].summary.cum_deaths for i in range(n_runs)]
    infected_lst = [msim_default.sims[i].summary.cum_infections for i in range(n_runs)]
    severe_lst = [msim_default.sims[i].summary.cum_severe for i in range(n_runs)]
    critical_lst = [msim_default.sims[i].summary.cum_critical for i in range(n_runs)]
    avg_severe, avg_critical = np.mean(severe_lst), np.mean(critical_lst)
    print('Average cumulative severe cases:', avg_severe)
    print('Average cumulative critical cases:', avg_critical)
    print('Range of total deaths:', min(dead_lst), '-', max(dead_lst))
    print('Range of total infections:', min(infected_lst), '-', max(infected_lst))
    print(msim_default)
    return


def run_diff_beta():
    betas = np.linspace(0.015, 0.045, 5)  # Sweep beta from 0.015 to 0.045 with 5 values
    beta_sims = []
    for beta in betas:
        beta_sim = cv.Sim(pars=pars, beta=beta, label=f'Beta = {beta}')
        beta_sims.append(beta_sim)
    msim_beta = cv.MultiSim(beta_sims)
    msim_beta.run()
    for _sim in msim_beta.sims:
        print(_sim.brief())
    diff_beta_plot = msim_beta.plot(
        to_plot=['new_infections', 'cum_infections', 'cum_deaths'],
        fig_args=dict(figsize=(9, 9)), do_save=True, fig_path='diff_beta.png')
    diff_beta_plot.show()
    return


def run_orig_sim_analyzer():
    orig_sim = cv.Sim(pars=pars, label='Default', analyzers=cv.age_histogram())
    orig_sim.run()
    print(orig_sim.brief())
    orig_sim_plot = orig_sim.plot(do_save=True, fig_path='orig_sim.png')
    orig_sim_plot.show()

    age_hist = orig_sim.get_analyzer()
    age_hist.plot()
    cv.savefig('age_hist.png')

    tt = orig_sim.make_transtree()
    tt.plot()
    cv.savefig('tt1.png')
    tt.plot_histograms()
    cv.savefig('tt2.png')

    sim_class = cv.Sim(pars=pars, analyzers=StoreSEIR(label='seir'))
    sim_class.run()
    seir = sim_class.get_analyzer('seir')  # Retrieve by label
    seir.plot()
    return


if __name__ == '__main__':
    # # Initialize and plot the population
    # plot_population()

    # # Same probability of transmission
    # run_same_beta()
    #
    # # Varying probability of transmission
    # run_diff_beta()
    #
    # # Analyzers implementation
    # run_orig_sim_analyzer()

    s1 = cv.Sim(pars=pars, label='Default')
    s2 = cv.Sim(pars=pars, interventions=shielding, label='Shielding for 60+ age group')
    s9 = cv.Sim(pars=pars, interventions=[tp, ct], label='Contact tracing')
    s20 = cv.Sim(pars=pars, interventions=[close_schools, close_community, close_work],
                 label='Partial closure of schools, work and community')
    s23 = cv.Sim(pars=pars, interventions=[reduce_beta_community, reduce_beta_workplace, reduce_beta_schools, tp, ct,
                                           close_schools, close_community, close_work], label='Combo')
    s25 = cv.Sim(pars=pars, interventions=[reduce_beta_community, reduce_beta_workplace, reduce_beta_schools],
                 label='Transmission reduction in schools, work and community')

    # Default
    s1.run()
    print(s1.brief)
    default_plot = s1.plot(
        # to_plot=['new_infections', 'cum_infections', 'new_deaths', 'cum_deaths', 'cum_tests',
        #          'new_quarantined', 'cum_severe', 'cum_critical'],
        fig_args=dict(figsize=(12, 9)), do_save=True, fig_path=r'default.png')
    default_plot.show()

    # # Default vs intervention
    # msim_interventions = cv.MultiSim([s1, s23])
    # msim_interventions.run()
    # for _sim in msim_interventions.sims:
    #     print(_sim.brief)
    # interventions_plot = msim_interventions.plot(
    #     to_plot=['new_infections', 'cum_infections', 'new_deaths', 'cum_deaths', 'cum_tests', 'new_quarantined'],
    #     fig_args=dict(figsize=(9, 9)), do_save=True, fig_path=r'default_vs_combo.png')
    # interventions_plot.show()

    # # Only intervention
    # s23.run()
    # print(s23.brief)
    # combo_plot = s23.plot(
    #     to_plot=['new_infections', 'cum_infections', 'new_deaths', 'cum_deaths', 'cum_tests', 'new_quarantined',
    #              'cum_severe', 'cum_critical'],
    #     # fig_args=dict(figsize=(12, 9)),
    #     do_save=True, fig_path=r'combo.png')
    # combo_plot.show()
