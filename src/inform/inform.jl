using Printf

function print_table_design_params(pari)

    # Assign the values from the pari array
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

    # Print the variables in a tabular manner
    println("-------------------------------------------------")
    @printf("%-40s %-10s\n", "Variable", "Value")
    println("-------------------------------------------------")
    @printf("%-40s %-10d\n", "Fuel type (Jet A)", ifuel == 24 ? 1 : 0)
    @printf("%-40s %-10s\n", "Fuel stored in center tanks", ifwcen == 0 ? "NO" : "YES")
    @printf("%-40s %-10s\n", "Cantilevered wing", iwplan == 1 ? "YES" : "NO")
    @printf("%-40s %-10s\n", "Engine mounted on wing", iengloc == 1 ? "YES" : "NO")
    @printf("%-40s %-10s\n", "Engine weight model", iengwgt == 1 ? "Basic" : "Advanced")
    @printf("%-40s %-10s\n", "Engine core in clean flow", iBLIc == 0 ? "YES" : "NO")

    if ifclose == 0
        @printf("%-40s %-10s\n", "CAUTION: Fuselage tapers to a point!", "")
        @printf("%-40s %-10s\n", "Recommend setting ifclose to 1 (tapers to edge)", "")
    else
        @printf("%-40s %-10s\n", "Fuselage tapers to edge", "YES")
    end

    @printf("%-40s %-10s\n", "Horizontal tail sizing method", iHTsize == 1 ? "Tail volume coefficient" : "Max. forward C.G")
    @printf("%-40s %-10s\n", "Vertical tail sizing method", iVTsize == 1 ? "Tail volume coefficient" : "Max. engine-out conditions")

    @printf("%-40s %-10s\n", "Wing position fixed", ixwmove == 0 ? "YES" : "NO")
    if ixwmove == 1
        @printf("%-40s %-10s\n", "Wing move criteria", "Get CLh=\"CLhspec\" in cruise")
    else
        @printf("%-40s %-10s\n", "Wing move criteria", "Get min static margin")
    end

    @printf("%-40s %-10s\n", "Store fuel in wing", ifwing == 1 ? "YES" : "NO")

    println("-------------------------------------------------")
end

using Printf

function print_table_mission_loads(parm, parg)
    # Define the indices for the parameters
    
    # Extract values from the parameter arrays
    Rangetot = parm[imRange]
    Wpay = parm[imWpay]
    Wpaymax = parg[igWpaymax]

    # Print the mission loads in a formatted manner
    println("-------------------------------------------------")
    println("                Mission Loads                   ")
    println("-------------------------------------------------")
    @printf("%-30s: %d\n", "Design range", Rangetot)
    @printf("%-30s: %d\n", "Typical payload", Wpay)
    @printf("%-30s: %d\n", "Max payload (Max. Pax * Wpax)", Wpaymax)
    println("-------------------------------------------------")
end

using Printf

function print_table_fuselage_params(parm, parg)

    # Extract values from the parameter arrays
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

    # Print the fuselage parameters in a formatted manner
    println("-------------------------------------------------")
    println("                Fuselage Parameters              ")
    println("-------------------------------------------------")
    @printf("%-50s: %d\n", "Fuselage radius (m)", Rfuse)
    @printf("%-50s: %d\n", "Bottom bubble downward extension", dRfuse)
    @printf("%-50s: %d\n", "Fuselage double-bubble half-width (m)", wfb)
    @printf("%-50s: %d\n", "Number of fuselage web (nfweb)", nfweb)
    @printf("%-50s: %d\n", "Fuselage floor height (m)", hfloor)
    @printf("%-50s: %d\n", "Fuselage nose position (m)", xnose)
    @printf("%-50s: %d\n", "Fuselage trailing edge position (m)", xend)
    @printf("%-50s: %d\n", "Pressurized shell start position (m)", xshell1)
    @printf("%-50s: %d\n", "Pressurized shell end position (m)", xshell2)
    @printf("%-50s: %d\n", "x-end-position of tailcone's primary structure (m)", xconend)
    @printf("%-50s: %d\n", "x-position of wingbox (m)", xwbox)
    @printf("%-50s: %d\n", "x-position of H.T wingbox (m)", xhbox)
    @printf("%-50s: %d\n", "x-position of V.T wingbox (m)", xvbox)
    @printf("%-50s: %d\n", "x-position of APU (m)", xapu)
    @printf("%-50s: %d\n", "x-position of engine (m)", xeng)
    println("-------------------------------------------------")
end


using Printf

function print_fuel_fractions(ffburn, ffuelb, ffuelc, ffueld, ffuele, ffuel)
    println("-------------------------------------------------")
    println("                 Fuel Fractions                  ")
    println("-------------------------------------------------")
    @printf("%-30s: %f\n", "Mission fuel fraction", ffburn)
    @printf("%-30s: %f\n", "Climb fuel fraction", ffuelb)
    @printf("%-30s: %f\n", "Cruise fuel fraction", ffuelc)
    @printf("%-30s: %f\n", "TOD fuel fraction", ffueld)
    @printf("%-30s: %f\n", "EOD fuel fraction", ffuele)
    @printf("%-30s: %f\n", "Total fuel fraction", ffuel)
    println("-------------------------------------------------")
end
