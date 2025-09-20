from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNF, LEFT, GROUP, FUNS, FLUID, SOLID, PROPULSION

# Change `use_sfmath` to False to use computer modern
x = XDSM(use_sfmath=False)

#-------------------------DIAGONAL BLOCK DEFINITION--------------------------#
# Add optimizer
#x.add_system("Optimizer", OPT, r"\text{Optimizer}")

x.add_system("Mission", SOLVER, r"\text{Mission}")

# Add fluid block
x.add_system("Fluid", FLUID, r"\text{Aerodynamics}")

# Add MDA block
x.add_system("Init", OPT, r"\text{Initialize}")

# Add MDA block
x.add_system("MDA", SOLVER, r"\text{Gauss-Siedel}")





# Add solid block
#x.add_system("Solid", SOLID, r"\text{Structure}")

# Add propulsion block
#x.add_system("Prop", PROPULSION, r"\text{Propulsion}")

# Define aircraft components:: Aerodynamics
x.add_system("FuseAero", FUNF, r"\text{Fuselage}", stack=False)

#x.add_system("WingAero", FUNF, r"\text{Wing}", stack=False)

#x.add_system("H.TailAero", FUNF, r"\text{Horizontal Tail}", stack=False)

#x.add_system("V.TailAero", FUNF, r"\text{Vertical Tail}", stack=False)

# DEFINE COMPONENTS :: STRUCTURES

#x.add_system("FuseStruct", FUNS, r"\text{Fuselage}", stack=False)

x.add_system("WingStruct", FUNS, r"\text{Wing}", stack=False)

x.add_system("TailStruct", FUNS, r"\text{H/V Tail}", stack=False)

x.add_system("NacelleStruct", FUNS, r"\text{Nacelle}", stack=False)

x.add_system("FuelFrac", FUNS, r"\text{Fuel}", stack=False)

#x.add_system("H.TailStruct", FUNS, r"\text{Horizontal Tail}", stack=False)

#x.add_system("V.TailStruct", FUNS, r"\text{Vertical Tail}", stack=False)

#-------------------------DIAGONAL BLOCK DEFINITION:END--------------------------#





# IPOPT Reached out to MDA solver

# Mission passes values to MDA solver
x.connect("Mission", "Init", "Range, Payload")

#x.connect("Init", "MDA", r"\text{fuel}")

x.connect("Init", "WingStruct", r"\text{wing geometry, payload}")

x.connect("Init", "TailStruct", r"\text{tail geometry, payload}")

x.connect("Init", "NacelleStruct", r"\text{geometry}")

x.connect("Init", "FuelFrac", r"\text{Fuel estimate}")

# Within Aerodynamcis, call Fuselage for aero properties
x.connect("Fluid", "FuseAero", r"\text{Fuselage geometry}")

x.connect("FuseAero", "Mission", r"\text{Fuselage drag}")

x.connect("WingStruct", "MDA", r"{weight^t, moment^t, centeroid^t}")

x.connect("TailStruct", "MDA", r"{weight^t, moment^t, centeroid^t}")

x.connect("NacelleStruct", "MDA", r"{length^t, weight-fraction^t}")

x.connect("FuelFrac", "MDA", r"{fuel weight^t}")

# MDA passes structural and propulsion variables to fluid
#x.connect("MDA", "Fluid", "y_s,y_p")

# MDA passespropulsion variables to solid
#x.connect("MDA", "Solid", "y_p")

# Fluid passes aerodynamic variables to elastic and propulsion
#x.connect("Fluid", "Solid", "y_a")
#x.connect("Fluid", "Prop", "y_a")

# Solid passes elastic variable to propulsion
#x.connect("Solid", "Prop", "y_s")

# Fluid passes aerodynamic variable to MDA
#x.connect("Fluid", "MDA", "y_a")

# Solid passes aerodynamic variable to MDA
#x.connect("Solid", "MDA", "y_s")

# Solid passes aerodynamic variable to MDA
#x.connect("Prop", "MDA", "y_p")

# Fluid passes converged aerodynamic variable to Functions
#x.connect("Fluid", "Func", "y_a^*")
#x.connect("Solid", "Func", "y_s^*")
#x.connect("Prop", "Func", "y_p^*")



#-------------------------INDIVIDUAL DISCIPLINES OUTPUT:BEGIN--------------------------#

#x.add_output("Solid", "y_s^*", side=LEFT)
#x.add_output("Prop", "y_p^*", side=LEFT)
#x.add_output("Optimizer", "x^*", side=LEFT)

#x.connect("Func", "Optimizer", r"f,g, \nabla f, \nabla g")
#-------------------------INDIVIDUAL DISCIPLINES OUTPUT:END--------------------------#


#-------------------------INDIVIDUAL DISCIPLINES INPUT:BEGIN--------------------------#
#x.add_input("Optimizer", "x^o")
#-------------------------INDIVIDUAL DISCIPLINES INPUT:END--------------------------#

#x.add_process(['Optimizer', 'MDA', 'Fluid', 'Solid', 'Prop', 'MDA'], arrow=False)

#x.add_process(['Optimizer', 'MDA', 'Func', 'Optimizer'], arrow=False)

x.write("TASOPT_XDSM")