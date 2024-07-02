using Printf
"""
    wsize(pari, parg, parm, para, pare,
        itermax, wrlx1, wrlx2, wrlx3,
        initwgt, initeng, iairf, Ldebug, printiter, saveODperf)

Main weight sizing function. Calls on various sub-functions to calculate weight of fuselage, wings, tails, etc.,
and iterates until the MTOW converges to within a specified tolerance.

CAUTION: This is dev version

!!! details "🔃 Inputs and Outputs"
    **Inputs:**
    - Array of flags that control design choices - fuel types, where to store fuel, etc.
    - Geometric and structural parameters - dimensions primarily
    - Aerodynamic parameters - CL, CD, KE dissipation, etc.
    - Mission-specific parameters - alt, mach, P, T etc.
    - Engine-specific parameters 

    **Outputs:**
    - No explicit outputs. Computed quantities are saved to `par` arrays of `aircraft` model.
"""

function wsizeDev(pari, parg, parm, para, pare,
    itermax, wrlx1, wrlx2, wrlx3,
    initwgt, initeng, iairf, Ldebug, printiter, saveODperf)

    time_propsys = 0.0

    println("Entering wsizeDev...")

    if pari[iiengmodel] == 0
        # Drela engine model
        println("Using Drela Engine model")
        use_NPSS = false
    else
        # NPSS
        use_NPSS = true
    end

    inite1 = 0

    if use_NPSS
        NPSS_PT = true
        NPSSsuccess = true
        wsize_fail = false
    else
        ichoke5 = zeros(iptotal)
        ichoke7 = zeros(iptotal)
        Tmrow = zeros(ncrowx)
        epsrow = zeros(ncrowx)
        epsrow_Tt3 = zeros(ncrowx)
        epsrow_Tt4 = zeros(ncrowx)
        epsrow_Trr = zeros(ncrowx)
    end

     # Weight convergence tolerance 
     tolerW = 1.0e-8

     errw = 1.0

    # Initialze some variables
    fsum = 0.0
    ifirst = true

    ifuel = pari[iifuel]
    ifwcen = pari[iifwcen]
    iwplan = pari[iiwplan]
    iengloc = pari[iiengloc]
    iengwgt = pari[iiengwgt]
    iBLIc = pari[iiBLIc]
    ifclose = pari[iifclose]
    iHTsize = pari[iiHTsize]
    iVTsize = pari[iiVTsize]
    ixwmove = pari[iixwmove]
    ifwing = pari[iifwing]

    print_table_design_params(pari)

    # Unpack number of powertrain elements
    nfan = parpt[ipt_nfan]
    ngen = parpt[ipt_ngen]
    nTshaft = parpt[ipt_nTshaft]


    # Atmospheric conditions at sea-level
    TSL, pSL, ρSL, aSL, μSL = atmos(0.0)

    # Calculate fuselage B.L. development at start of cruise: ipcruise1
    time_fusebl = @elapsed fusebl!(pari, parg, para, ipcruise1)

    # Kinetic energy area at T.E. at start of cruise: ipcruise1
    KAfTE = para[iaKAfTE, ipcruise1]

    # Surface dissapation area at start of cruise: ipcruise1
    DAfsurf = para[iaDAfsurf, ipcruise1]

    # Wake dissipation area
    DAfwake = para[iaDAfwake, ipcruise1]

    # Momentum area at ∞
    PAfinf = para[iaPAfinf, ipcruise1]

    # Assume K.E., Disspation and momentum areas are const. for all mission points:
    para[iaKAfTE, :] .= KAfTE
    para[iaDAfsurf, :] .= DAfsurf
    para[iaDAfwake, :] .= DAfwake
    para[iaPAfinf, :] .= PAfinf

    # Unpack payload and range for design mission - this is the mission that the structures are sized for
    Rangetot = parm[imRange]

   #Typical payload := Passengers * Weight per passenger
   Wpay = parm[imWpay]

   # Max payload := Max. Pax * Wpax
   Wpaymax = parg[igWpaymax] 

    # if Wpay or Wpaymax is unset
    if (Wpaymax == 0)
        println("Max payload weight was not set, setting Wpaymax = Wpay")
        Wpaymax = parg[igWpaymax] = max(Wpay, Wpaymax)
    end

    print_table_mission_loads(parm, parg)

    # Store the design mission in the geometry array as well
    parg[igRange] = Rangetot
    parg[igWpay] = Wpay

    # Fixed weight and location of fixed weight
    Wfix = parg[igWfix] # Cockpit + pilots weight
    xfix = parg[igxfix] # Location of cockpit + pilots -> inline with pilot's eye


    # Weight fractions
    fapu = parg[igfapu]
    fpadd = parg[igfpadd]
    fseat = parg[igfseat]
    feadd = parg[igfeadd]   # used for tfweight
    fnace = parg[igfnace]   # not used anywhere!
    fhadd = parg[igfhadd]   # used for tail sizing
    fvadd = parg[igfvadd]  # used for tail sizing
    fwadd = parg[igfflap] + parg[igfslat] +  # used for wing weight sizing
            parg[igfaile] + parg[igflete] + parg[igfribs] + parg[igfspoi] + parg[igfwatt]


    fstring = parg[igfstring] #used for fuseW
    fframe = parg[igfframe]     # used for fuseW
    ffadd = parg[igffadd]   # used for fuseW
        
    fpylon = parg[igfpylon] # used for tfweight

    fhpesys = parg[igfhpesys]
    flgnose = parg[igflgnose]
    flgmain = parg[igflgmain]

    freserve = parg[igfreserve]

    # fuselage lift carryover loss, tip lift loss fractions

    fLo = parg[igfLo]
    fLt = parg[igfLt]

    # fuselage dimensions and coordinates
    Rfuse = parg[igRfuse]
    dRfuse = parg[igdRfuse]
    wfb = parg[igwfb]
    nfweb = parg[ignfweb]
    hfloor = parg[ighfloor]
    xnose = parg[igxnose]
    xend = parg[igxend]
    xshell1 = parg[igxshell1]
    xshell2 = parg[igxshell2]
    xconend = parg[igxconend]
    xwbox = parg[igxwbox]
    xhbox = parg[igxhbox]
    xvbox = parg[igxvbox]
    xapu = parg[igxapu]
    xeng = parg[igxeng]

    print_table_fuselage_params(parm, parg)

    # calculate payload proportional weights from weight fractions
    Wapu = Wpaymax * fapu
    Wpadd = Wpaymax * fpadd
    Wseat = Wpaymax * fseat

    # window and insulation densities per length and per area
    Wpwindow = parg[igWpwindow]
    Wppinsul = parg[igWppinsul]
    Wppfloor = parg[igWppfloor]

    # fuselage-bending inertial relief factors
    rMh = parg[igrMh]
    rMv = parg[igrMv]

    # tail CL's at structural sizing cases
    CLhmax = parg[igCLhmax]
    CLvmax = parg[igCLvmax]

    # wing break, wing tip taper ratios
    λs = parg[iglambdas]
    λt = parg[iglambdat]

    # tail surface taper ratios (no inner panel, so λs=1)
    λhs = 1.0
    λh = parg[iglambdah]
    λvs = 1.0
    λv = parg[iglambdav]

    # tailcone taper ratio
    λc = parg[iglambdac]

    # wing geometry parameters
    sweep = parg[igsweep]
    wbox = parg[igwbox]
    hboxo = parg[ighboxo]
    hboxs = parg[ighboxs]
    rh = parg[igrh]
    AR = parg[igAR]
    bo = parg[igbo]
    ηs = parg[igetas]
    Xaxis = parg[igXaxis]

    # tail geometry parameters
    sweeph = parg[igsweeph]
    wboxh = parg[igwboxh]
    hboxh = parg[ighboxh]
    rhh = parg[igrhh]
    ARh = parg[igARh]
    boh = parg[igboh]

    sweepv = parg[igsweepv]
    wboxv = parg[igwboxv]
    hboxv = parg[ighboxv]
    rhv = parg[igrhv]
    ARv = parg[igARv]
    bov = parg[igbov]

    # number of vertical tails
    nvtail = parg[ignvtail]

    # strut vertical base height, h/c, strut shell t/h
    zs = parg[igzs]
    hstrut = parg[ighstrut]
    tohstrut = 0.05

    # assume no struts on tails
    zsh = 0.0
    zsv = 0.0

    # max g load factors for wing, fuselage
    Nlift = parg[igNlift]
    Nland = parg[igNland]

    # never-exceed dynamic pressure for sizing tail structure
    Vne = parg[igVne]
    qne = 0.5 * ρSL * Vne^2

    # wingbox stresses and densities [section 3.1.9 taspot.pdf [prash]] #reference
    σcap = parg[igsigcap] * parg[igsigfac]
    tauweb = parg[igtauweb] * parg[igsigfac]
    rhoweb = parg[igrhoweb]
    rhocap = parg[igrhocap]

    # fuselage stresses and densities
    σskin = parg[igsigskin] * parg[igsigfac]
    σbend = parg[igsigbend] * parg[igsigfac]
    rhoskin = parg[igrhoskin]
    rhobend = parg[igrhobend]

    # fuselage shell bending/skin modulus ratio
    rEshell = parg[igrEshell]

    # strut stress and density
    σstrut = parg[igsigstrut] * parg[igsigfac]
    rhostrut = parg[igrhostrut]


    # assume tail stresses and densities are same as wing's (keeps it simpler)
    σcaph = σcap
    tauwebh = tauweb
    rhowebh = rhoweb
    rhocaph = rhocap

    σcapv = σcap
    tauwebv = tauweb
    rhowebv = rhoweb
    rhocapv = rhocap

    # number of engines, y-position of outermost engine
    neng = parg[igneng]
    yeng = parg[igyeng]

    # fan hub/tip ratio
    HTRf = parg[igHTRf]

    # nacelle wetted area / fan area ratio
    rSnace = parg[igrSnace]

    # set cruise-altitude atmospheric conditions
    ip = ipcruise1
    altkm = para[iaalt, ip] / 1000.0
    T0, p0, ρ0, a0, μ0 = atmos(altkm)
    Mach = para[iaMach, ip]
    pare[iep0, ip] = p0
    pare[ieT0, ip] = T0
    pare[iea0, ip] = a0
    pare[ierho0, ip] = ρ0
    pare[iemu0, ip] = μ0
    pare[ieM0, ip] = Mach
    pare[ieu0, ip] = Mach * a0
    para[iaReunit, ip] = Mach * a0 * ρ0 / μ0

    # set takeoff-altitude atmospheric conditions
    ip = iprotate
    altkm = para[iaalt, ip] / 1000.0
    T0, p0, ρ0, a0, μ0 = atmos(altkm)
    Mach = 0.25
    pare[iep0, ip] = p0
    pare[ieT0, ip] = T0
    pare[iea0, ip] = a0
    pare[ierho0, ip] = ρ0
    pare[iemu0, ip] = μ0
    pare[ieM0, ip] = Mach
    pare[ieu0, ip] = Mach * a0
    para[iaReunit, ip] = Mach * a0 * ρ0 / μ0

    # Set atmos conditions for top of climb
    ip = ipclimbn
    altkm = para[iaalt, ipcruise1] / 1000.0
    T0, p0, ρ0, a0, μ0 = atmos(altkm)
    Mach = para[iaMach, ip]
    pare[iep0, ip] = p0
    pare[ieT0, ip] = T0
    pare[iea0, ip] = a0
    pare[ierho0, ip] = ρ0
    pare[iemu0, ip] = μ0
    pare[ieM0, ip] = Mach
    pare[ieu0, ip] = Mach * a0
    para[iaReunit, ip] = Mach * a0 * ρ0 / μ0

    # -------------------------------------------------------    
    ## Initial guess section [Section 3.2 of TASOPT docs]
    # -------------------------------------------------------

    # Allow first iteration

    if (initwgt == 0)

        println("Starting first iteration....")

        # Estimate H.T weight
        Whtail = 0.05 * Wpay / parg[igsigfac]

        # Estimate H.T weight
        Wvtail = Whtail

        # Estimate wing weight based on payload weight and stress factor
        Wwing = 0.5 * Wpay / parg[igsigfac]

        # Strut weight
        Wstrut = 0.0

        # Engine weight (sized later)
        Weng = 0.0 * Wpay

        # Engine weight fraction
        feng = 0.0


        dxWhtail = 0.0
        dxWvtail = 0.0

        # Wing planform sizing at start of cruise
        ip = ipcruise1

        # Total weight as some fracion of payload weight
        W = 5.0 * Wpay
        
        # Compute the surface area of the wing based on design CL at start of cruise, flight speed and L = W
        # Here, we assume that the wing takes on the entire lift. 
        S = W / (0.5 * pare[ierho0, ip] * pare[ieu0, ip]^2 * para[iaCL, ip])


        #Estimate span
        b = sqrt(S * parg[igAR])

        # Span at Yehudi break
        bs = b * ηs

        # Weight at innner pannel
        Winn = 0.15 * Wpay / parg[igsigfac]

        # Weight at outer pannel
        Wout = 0.05 * Wpay / parg[igsigfac]

        # Weight moment
        dyWinn = Winn * 0.30 * (0.5 * (bs - bo))

        # Weight moment for outer pannel
        dyWout = Wout * 0.25 * (0.5 * (b - bs))

        # Assign estimated values to geometry arrays -> extract @ second iteration
        parg[igWhtail] = Whtail
        parg[igWvtail] = Wvtail
        parg[igWwing] = Wwing
        parg[igWstrut] = Wstrut
        parg[igWeng] = Weng
        parg[igWinn] = Winn
        parg[igWout] = Wout
        parg[igdxWhtail] = dxWhtail
        parg[igdxWvtail] = dxWvtail
        parg[igdyWinn] = dyWinn
        parg[igdyWout] = dyWout

        # wing centroid x-offset form wingbox
        dxwing, macco = surfdx(b, bs, bo, λt, λs, sweep)
        xwing = xwbox + dxwing
        parg[igxwing] = xwing


        # tail area centroid locations (assume no offset from sweep initially)
        parg[igxhtail], parg[igxvtail] = xhbox, xvbox

        # center wingbox chord extent for fuse weight calcs (small effect)
        cbox = 0.0

        # nacelle, fan duct, core, cowl lengths ℛℯ calcs
        parg[iglnace] = 0.5 * S / b

        # nacelle Awet/S
        fSnace = 0.2
        parg[igfSnace] = fSnace

        # Estimate the fuel required from BRE
        LoD = 18.0
        TSFC = 1.0 / 7000.0
        # Get flight speed at beginning of cruise
        V = pare[ieu0, ipcruise1]

        ffburn = (1.0 - exp(-Rangetot * TSFC / (V * LoD))) # ffburn = Wfuel/WMTO
        ffburn = min(ffburn, 0.8 / (1.0 + freserve))   # 0.8 is the fuel useability?

        # mission-point fuel fractions ⇾ ffuel = Wfuel/WMTO
        ffuelb = ffburn * (1.0 + freserve)  # start of climb
        ffuelc = ffburn * (0.90 + freserve)  # start of cruise
        ffueld = ffburn * (0.02 + freserve)  # start of descent
        ffuele = ffburn * (0.0 + freserve)  # end of descent (landing)

        # max fuel fraction is at start of climb
        ffuel = ffuelb # ffuel is max fuel fraction

        # Set initial climb γ = 0 to force intial guesses
        para[iagamV, :] .= 0.0

        # Put initial-guess weight fractions in mission-point array.

        # These are points before climb starts
        para[iafracW, ipstatic] = 1.0
        para[iafracW, iprotate] = 1.0
        para[iafracW, iptakeoff] = 1.0
        para[iafracW, ipcutback] = 1.0 

        print_fuel_fractions(ffburn, ffuelb, ffuelc,ffueld,ffuele, ffuel)

        # Interpolate for the rest
        #--------- Simple linear interpolation ---------
        #     y-y1   y2-y1
        #     ---- = ----- => y = y1*(1-frac) + y2*frac
        #     x-x1   x2-x1
        #-----------------------------------------------
        # Climb
        @inbounds for ip = ipclimb1:ipclimbn
            frac = float(ip - ipclimb1) / float(ipclimbn - ipclimb1)
            ffp = ffuelb * (1.0 - frac) + ffuelc * frac
            para[iafracW, ip] = 1.0 - ffuel + ffp
        end

        # Cruise
        @inbounds for ip = ipcruise1:ipcruisen
            frac = float(ip - ipcruise1) / float(ipcruisen - ipcruise1)
            ffp = ffuelc * (1.0 - frac) + ffueld * frac
            para[iafracW, ip] = 1.0 - ffuel + ffp
        end

        # Descent
        @inbounds for ip = ipdescent1:ipdescentn
            frac = float(ip - ipdescent1) / float(ipdescentn - ipdescent1)
            ffp = ffueld * (1.0 - frac) + ffuele * frac
            para[iafracW, ip] = 1.0 - ffuel + ffp
        end

        # Initial tail info for sizing of fuselage bending and torsion added material 
        # NoteL CLh and CLv are user-input. 
        Sh = (2.0 * Wpaymax) / (qne * CLhmax)
        Sv = (2.0 * Wpaymax) / (qne * CLvmax)
        bv = sqrt(Sv * ARv)

        # Store the wetted areas for H.T and V.T
        parg[igSh] = Sh
        parg[igSv] = Sv

        # Initial cruise-climb angle gamVcr needed to estimate end-of-cruise altitude
        gamVcr = 0.0002
        # Estimate the CD @ ipcruise1 based on user-defined CL and LoD
        para[iaCD, ipcruise1] = para[iaCL, ipcruise1] / LoD
        para[iagamV, ipcruise1] = gamVcr       


        # Pressure and altitude at start of cruise
        Mach = para[iaMach, ipcruise1]
        p0c = pare[iep0, ipcruise1]
        altc = para[iaalt, ipcruise1]

        # Guess pressure at end-of-cruise (scales with weight)
        p0d = p0c * (1.0 - ffuel + ffueld) / (1.0 - ffuel + ffuelc)
        pare[iep0, ipcruisen] = p0d

        # Guess for OEI #TODO This needs some thinking about what is "One engine out" mean for a turbo-electric aircraft
        # pare[ieFe, iprotate] = 2.0*Wpay/neng
        pare[ieFe, iprotate] = 2.0 * Wpay # ieFe now stores total thrust
        pare[ieu0, iprotate] = 70.0
        Afan = 3.0e-5 * Wpay / neng
        parg[igdfan] = sqrt(Afan * 4.0 / π)
        
        # Guess fan face mach numbers for nacelle CD calcs
        M2des = pare[ieM2, ipcruise1] = 0.6
        pare[ieM2, ipstatic:ipcruisen] .= M2des
        pare[ieM2, ipdescent1:ipdescentn] .= 0.8 * M2des

        if (!use_NPSS)
            # calculate initial guesses for cooling mass flow ratios epsrow(.)
            ip = iprotate
            cpc = 1080.0
            cp4 = 1340.0
            Rgc = 288.0
            Rg4 = 288.0
            M0to = pare[ieu0, ip] / pare[iea0, ip]
            T0to = pare[ieT0, ip]
            epolhc = pare[ieepolhc, ip]
            OPRto = pare[iepilc, ipcruise1] * pare[iepihc, ipcruise1]
            Tt4to = pare[ieTt4, ip]
            dTstrk = pare[iedTstrk, ip]
            Mtexit = pare[ieMtexit, ip]
            efilm = pare[ieefilm, ip]
            tfilm = pare[ietfilm, ip]
            StA = pare[ieStA, ip]
            for icrow = 1:ncrowx
                Tmrow[icrow] = parg[igTmetal]
            end

            Tt2to = T0to * (1.0 + 0.5 * (gamSL - 1.0) * M0to^2)
            Tt3to = Tt2to * OPRto^(Rgc / (epolhc * cpc))
            Trrat = 1.0 / (1.0 + 0.5 * Rg4 / (cp4 - Rg4) * Mtexit^2)

            ncrow, epsrow, epsrow_Tt3, epsrow_Tt4, epsrow_Trr = mcool(ncrowx, Tmrow,
                Tt3to, Tt4to, dTstrk, Trrat, efilm, tfilm, StA)

            epstot = 0.0
            for icrow = 1:ncrow
                epstot = epstot + epsrow[icrow]
            end
            fo = pare[iemofft, ip] / pare[iemcore, ip]
            fc = (1.0 - fo) * epstot

            for jp = 1:iptotal
                pare[iefc, jp] = fc
                for icrow = 1:ncrowx
                    pare[ieepsc1+icrow-1, jp] = epsrow[icrow]
                    pare[ieTmet1+icrow-1, jp] = Tmrow[icrow]
                end
            end
        end
        
        println("Fist iteration complete!")

    else

        # Wing parameters
        S = parg[igS]
        b = parg[igb]
        bs = parg[igbs]
        bo = parg[igbo]

        bh = parg[igbh]
        bv = parg[igbv]

        coh = parg[igcoh]
        cov = parg[igcov]

        cbox = parg[igco] * parg[igwbox]

        Whtail = parg[igWhtail]
        Wvtail = parg[igWvtail]
        Wwing = parg[igWwing]
        Wstrut = parg[igWstrut]
        Weng = parg[igWeng]
        Winn = parg[igWinn]
        Wout = parg[igWout]
        dxWhtail = parg[igdxWhtail]
        dxWvtail = parg[igdxWvtail]
        dyWinn = parg[igdyWinn]
        dyWout = parg[igdyWout]

        WMTO = parg[igWMTO]
        feng = parg[igWeng] / WMTO
        ffuel = parg[igWfuel] / WMTO

        xwing = parg[igxwing]
        dxwing = parg[igxwing] - parg[igxwbox]

        xhtail = parg[igxhtail]
        xvtail = parg[igxvtail]

        xhbox = parg[igxhbox]
        xvbox = parg[igxvbox]

        fSnace = parg[igfSnace]

        Sh = parg[igSh]
        Sv = parg[igSv]
        ARh = parg[igARh]
        ARv = parg[igARv]

        # Turbo-electric system
        Wtesys = parg[igWtesys]
        Wftank = parg[igWftank]
    end   
end