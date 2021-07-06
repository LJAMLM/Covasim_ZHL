import numpy as np
import covasim as cv
import pandas as pd
import pylab as pl
from functools import reduce
import sciris as sc

cv.options.set(dpi=100, show=False, close=True, verbose=0)

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

kakuma_pop_survey = {
    '0-4': 20303,
    '5-11': 33641,
    '12-17': 30947,
    '18-27': 28347,
    '28-37': 21208,
    '38-47': 14153,
    '48-57': 5199,
    '58+': 2393,
}

kalobeyei_pop_survey = {
    '0-4': 6286,
    '5-11': 11259,
    '12-17': 7394,
    '18-27': 4454,
    '28-37': 5109,
    '38-47': 2336,
    '48-57': 806,
    '58+': 437,
}

pop_kakuma_may = sum(kakuma_pop.values())  # 156232
pop_kalobeyei_may = sum(kalobeyei_pop.values())  # 38081
pop_kakuma_kalobeyei_may = pop_kakuma_may + pop_kalobeyei_may  # 194313

# Add kakuma dict to kalobeyei dict.
lst_kakuma_kalobeyei = [kakuma_pop_survey, kalobeyei_pop_survey]
kakuma_kalobeyei_pop = reduce(lambda x, y: dict((k, v + y[k]) for k, v in x.items()), lst_kakuma_kalobeyei)

# kakuma_kalobeyei_pop = {
#     '0-4': 26589,
#     '5-11': 44900,
#     '12-17': 38341,
#     '18-59': 81952,
#     '60+': 2531,
# }

kakuma_kalobeyei_pop_survey = {
    '0-4': 26589,
    '5-11': 44900,
    '12-17': 38341,
    '18-27': 31803,
    '28-37': 27878,
    '38-47': 16137,
    '48-57': 5808,
    '58+': 2826,
}

kakuma_hhsize = 6.3
kalobeyei_hhsize = 5.9
kakuma_kalobeyei_hhsize = (kakuma_hhsize * pop_kakuma_may + kalobeyei_hhsize * pop_kalobeyei_may) / \
                          (pop_kakuma_may + pop_kalobeyei_may)  # 6.222
# kakuma_kalobeyei_household_size_survey = 6.132  # 5.863, 5.708

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
    'start_day': '2020-05-25',
    # 'n_days': 250,
    'end_day': '2021-05-25',
    'contacts': {'s': 35, 'w': 10, 'c': 40},    # 30 mainly for children with disabilities, more than 100 in class
    'n_beds_hosp': 20,
    'n_beds_icu': 0,
}

monthly_data = pd.DataFrame({
    'date': ['2020-05-31', '2020-06-30', '2020-07-31', '2020-08-31', '2020-09-30',
             '2020-10-31', '2020-11-30', '2020-12-31', '2021-01-31'],
    'cum_diagnoses': [1, 2, 33, 81, 191,
                      324, 435, 486, 516],
    'new_diagnoses': [1, 1, 31, 48, 110,
                      133, 111, 51, 30],
    'cum_tests': [22, 88, 777, 1854, 2636,
                  3714, 5260, 6198, 6609],
    'new_tests': [22, 66, 689, 1077, 782,
                  1078, 1546, 938, 411],
    'cum_recovered': [0, 1, 11, 50, 50,
                      50, 50, 50, 50],
    'cum_deaths': [0, 0, 0, 1, 4,
                   7, 10, 10, 10],
    'new_deaths': [0, 0, 0, 1, 3,
                   3, 3, 0, 0],
})

monthly_data.to_csv('monthly_data.csv')

reduce_beta_workplace = cv.change_beta(days=7, changes=0.7, layers='w')
reduce_transmission = cv.change_beta(days=7, changes=0.7, layers=['s', 'w', 'c'])
reduce_nr_of_contacts = cv.clip_edges(days=[30, 60, 90, 120], changes=[0.0, 0.1, 0.2, 0.3], layers=['s', 'w', 'c'])
close_schools = cv.clip_edges(days=7, changes=0.2, layers='s')
close_work = cv.clip_edges(days=7, changes=0.2, layers='w')
close_community = cv.clip_edges(days=7, changes=0.2, layers='c')
partial_lockdown = cv.clip_edges(days=7, changes=0.2, layers=['s', 'w', 'c'])

tn_data = cv.test_num('data', symp_test=100)
# On average a person will get tested with 20% probability per day
tp = cv.test_prob(symp_prob=0.2, symp_quar_prob=0.3, start_day=7, test_delay=2)
ct = cv.contact_tracing(trace_probs=dict(h=0.8, s=0.3, w=0.3, c=0.1))  # reduce probabilities by a lot!

vaccine = cv.simple_vaccine(days=[30, 60], cumulative=[0.5, 0.4], prob=0.6, rel_sus=0.5, rel_symp=0.1)

imports = cv.dynamic_pars(n_imports=dict(days=[15, 30], vals=[100, 0]))


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


vaccine2 = cv.simple_vaccine(days=30, rel_sus=0.8, rel_symp=0.06, subtarget=vaccinate_by_age)


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


# Set up the intervention
on = 0.2  # Beta less than 1 -- intervention is on
off = 1.0  # Beta is 1, i.e. normal -- intervention is off
changes = [on, off]
plot_args = dict(label='Dynamic beta', show_label=True, line_args={'c': 'blue'})
db = cv.change_beta(days=inf_thresh, changes=changes, **plot_args)

# Set custom properties
db.t_on = np.nan
db.t_off = np.nan
db.active = False
db.plot_days = []


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


protect = ProtectElderly(start_day=7, rel_sus=0.25)  # Create intervention


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


if __name__ == '__main__':
    # sim = cv.Sim(pars=pars, label="Default")
    # sim.initialize()  # Create people
    # fig = sim.people.plot(fig_args=dict(figsize=(30, 15))).show()  # Show statistics of the people
    #
    # orig_sim = cv.Sim(pars=pars, beta=0.015, label='Default', analyzers=cv.age_histogram())
    # orig_sim.run()
    # print(orig_sim.brief())
    # orig_sim.plot().show()
    #
    # # Same probability of transmission
    # default_sim = cv.Sim(pars=pars)
    # msim_default = cv.MultiSim(default_sim)
    # msim_default.run(n_runs=5)
    # for _sim in msim_default.sims:
    #     print(_sim.brief())
    # msim_default.mean()
    # msim_default.plot(fig_args=dict(figsize=(9, 9))).show()
    # msim_default.save('default.sim')
    #
    # # Varying probability of transmission
    # betas = np.linspace(0.010, 0.020, 5)  # Sweep beta from 0.01 to 0.02 with 5 values
    # beta_sims = []
    # for beta in betas:
    #     beta_sim = cv.Sim(pars=pars, beta=beta, label=f'Beta = {beta}')
    #     beta_sims.append(beta_sim)
    # msim_beta = cv.MultiSim(beta_sims)
    # msim_beta.run()
    # for _sim in msim_beta.sims:
    #     print(_sim.brief())
    # msim_beta.plot(fig_args=dict(figsize=(9, 9))).show()
    # msim_beta.save('beta.sim')
    #
    # # Analyzers implementation
    # age_hist = orig_sim.get_analyzer()
    # age_hist.plot()
    # pl.show()
    #
    # tt = orig_sim.make_transtree()
    # fig1 = tt.plot()
    # pl.show()
    # fig2 = tt.plot_histograms()
    # pl.show()
    #
    # sim_class = cv.Sim(pars=pars, analyzers=StoreSEIR(label='seir'))
    # sim_class.run()
    # seir = sim_class.get_analyzer('seir')  # Retrieve by label
    # seir.plot()
    # pl.show()
    #
    # sim_check = cv.Sim(pars=pars, analyzers=check_88)
    # sim_check.run()

    s1 = cv.Sim(pars=pars, label='Default')
    # s2 = cv.Sim(pars=pars, interventions=shielding, label='Shielding for 60+ group')
    # s3 = cv.Sim(pars=pars, datafile="monthly_data.csv", interventions=cv.test_num(daily_tests='data'), label='Test')
    # s4 = cv.Sim(pars=pars, interventions=reduce_transmission, label='Reduce transmission')
    # s5 = cv.Sim(pars=pars, interventions=reduce_nr_of_contacts, label='Reduce contacts / close buildings')
    # s6 = cv.Sim(pars=pars, interventions=[reduce_transmission, reduce_nr_of_contacts],
    #             label='Reduce contacts and use masks in buildings')
    # s7 = cv.Sim(pars=pars, datafile='monthly_data.csv', interventions=[tn_data, ct],
    #             label='Number of tests from data')
    # s8 = cv.Sim(pars=pars, interventions=tp, label='Probability-based testing')
    # s9 = cv.Sim(pars=pars, interventions=[tp, ct], label='Contact tracing')
    # s10 = cv.Sim(pars=pars, interventions=vaccine, label='Simple vaccine')
    # s11 = cv.Sim(pars=pars, interventions=vaccine2, label='Age-dependent vaccination')
    # s12 = cv.Sim(pars=pars, interventions=imports, label='With imported infections')
    # s13 = cv.Sim(pars=pars, interventions=db, label='Dynamic beta')
    # s14 = cv.Sim(pars=pars, interventions=dynamic_imports, label='Dynamic imports')
    # s15 = cv.Sim(pars=pars, interventions=protect, label='Protect the elderly')
    # s16 = cv.Sim(pars=pars, interventions=[shielding, reduce_transmission, reduce_nr_of_contacts],
    #              label='Combo shielding, reducing transmission & partially closing places')
    # s17 = cv.Sim(pars=pars, interventions=close_schools, label='Close schools')
    # s18 = cv.Sim(pars=pars, interventions=close_work, label='Close work')
    # s19 = cv.Sim(pars=pars, interventions=close_community, label='Close community')
    # s20 = cv.Sim(pars=pars, interventions=[close_schools, close_community, close_work],
    #              label='Partial lockdown')
    # s21 = cv.Sim(pars=pars, interventions=[close_schools, close_work, close_community], label='Lockdown for 2 months')
    # s22 = cv.Sim(pars=pars, interventions=partial_lockdown, label='Partial lockdown 2')
    s23 = cv.Sim(pars=pars, interventions=[partial_lockdown], label='Combo')

    msim_interventions = cv.MultiSim([s1, s23])
    msim_interventions.run()
    for _sim in msim_interventions.sims:
        print(_sim.brief())
        # _sim.save(f'{_sim.label.replace(" ", "_")}.sim')
    cv.options.set(font_size=8, show=True)
    msim_interventions.plot(to_plot=['new_infections', 'new_tests', 'cum_infections', 'cum_deaths',
                                     'new_quarantined', 'test_yield']).show()
    cv.options.set(font_size='default', show=False)  # Reset to the default value
    # msim_interventions.save('default_vs_closing_schools_and_community_and_less_beta_workplace.sim')
