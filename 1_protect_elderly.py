import covasim as cv


# Custom intervention -- see tutorial 5
def protect_elderly(sim):
    if sim.t == sim.day("2020-04-01"):
        elderly = sim.people.age > 70
        sim.people.rel_sus[elderly] = 0.0


pars = {
    "pop_type": "hybrid",
    "location": "japan",
    "pop_size": 50e3,
    "pop_infected": 100,
    "n_days": 90,
    "verbose": 0,
}

# Running with multisims -- see tutorial 3
if __name__ == "__main__":
    s1 = cv.Sim(pars=pars, label="Default")
    s2 = cv.Sim(pars=pars, interventions=protect_elderly, label="Protect the elderly")
    msim = cv.MultiSim([s1, s2])
    msim.run()
    msim.plot(to_plot=["cum_deaths", "cum_infections"])
