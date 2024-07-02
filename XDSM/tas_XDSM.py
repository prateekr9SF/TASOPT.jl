from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

# Change `use_sfmath` to False to use computer modern
x = XDSM(use_sfmath=True)

x.add_system("opt", OPT, r"\text{tasopt.jl}")
x.add_system("wsize", SOLVER, r"\text{wsize()}")

# Preprocessing step-----------------------------------------
x.add_system("preproc", SOLVER, r"\text{preprocess()}")

x.add_system("atmos", FUNC, r"\text{atmos()}")
x.add_system("fusebl", FUNC, r"\text{fusebl()}")
x.add_system("waypaymax", FUNC, r"\text{waypaymax()}")
x.add_system("weightfrac", FUNC, r"\text{weightfrac()}")

x.add_system("structures", FUNC, r"\text{structures()}")

x.add_system("atm_cons", FUNC, r"\text{atmCond()}")

x.connect("opt", "wsize", "para, pare, parg, pari")
x.connect("wsize", "preproc", "para, pare, parg, pari")
x.connect("preproc", "waypaymax", "Max. Pax X Wpax")
x.connect("preproc", "fusebl", "para, parg, pari, ipcruise1")
x.connect("preproc", "weightfrac", "parg")
x.connect("preproc", "structures", "parg")

x.connect("atmos", "wsize", "TSL, rhoSL, aSL, muSL")
x.connect("fusebl", "wsize", "KAfTE, DAfwake, PAfinf")
x.connect("waypaymax", "wsize", "maxPayload")
x.connect("waypaymax", "weightfrac", "maxPayload")

x.connect("weightfrac", "tfweight", "feadd, pylon")
x.connect("weightfrac", "fuseW", "fstring, fframe, ffadd")

x.connect("weightfrac", "wIter", "fhadd ,fvadd, fwadd, fhpesys, flgnose, flgmain")

x.add_output("weightfrac", r"W_{apu}, W_{padd}, W_{seat}", side=LEFT)
x.add_output("structures", r"\sigma W_{cap}, \tau_{web}, \sigma_{skin}, \sigma_{bend}, \sigma_{strut}, \sigma_{caph}, \sigma_{capv}", side=LEFT)

x.connect("atm_cons", "atmos", "altitude")
x.connect("atmos", "atm_cons", r"T, \rho, a, \mu")
x.connect("atm_cons", "wsize", "parae, para")

# Initialzation step-----------------------------------------
x.add_system("guess", SOLVER, r"\text{guess()}")


# Weight iteration step-----------------------------------------
x.add_system("wIter", SOLVER, r"\text{wIter()}")
x.add_system("fuseW", FUNC, r"\text{fuseW()}")
x.add_system("tfweight", FUNC, r"\text{tfweight()}")




#x.connect("F", "opt", "f")


#x.add_output("F2", "y_2^*", side=LEFT)
#x.add_output("F", "f^*", side=LEFT)
#x.add_output("G", "g^*", side=LEFT)
x.write("mdf")