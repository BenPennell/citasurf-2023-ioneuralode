# https://gist.github.com/xzackli/56219698015994fd4b437389719b9f3b

# vvvvv
using Bolt, JSON
𝕡 = CosmoParams(Ω_c=0.224) # set kwargs like so to change the default values
bg = Background(𝕡)
𝕣 = Bolt.RECFAST(bg; OmegaB=𝕡.Ω_b, Yp=𝕡.Y_p, OmegaG=𝕡.Ω_r, Tnow=2.725)

@time rhist = Bolt.recfastsolve(𝕣);

# extraction routines for z > 3500
get_H_from_H_He_evo(𝕣, z, sol) = sol(z, idxs=1)
get_He_from_H_He_evo(𝕣, z, sol) = sol(z, idxs=2)
function get_H_from_He_evo(𝕣, z, sol)
    y = sol(z, idxs=1)
    rhs = exp(1.5 * log(𝕣.CR*𝕣.Tnow/(1+z)) - 𝕣.CB1/(𝕣.Tnow*(1+z))) / 𝕣.Nnow
    x_H0 = 0.5 * (sqrt(rhs^2+4*rhs) - rhs)
    return x_H0
end
get_He_from_He_evo(𝕣, z, sol) = sol(z, idxs=1)
function later_X_H(rhist, z) 
    𝕣 = rhist.𝕣
    if (z > rhist.z_He_evo_start)
        return 1.0
    elseif (z > rhist.z_H_He_evo_start)
        return get_H_from_He_evo(𝕣, z, rhist.sol_He)
    else
        return get_H_from_H_He_evo(𝕣, z, rhist.sol_H_He)
    end
end

function later_X_He(rhist, z) 
    𝕣 = rhist.𝕣
    if (z > 5000.)
        rhs = exp(1.5 * log(𝕣.CR*𝕣.Tnow/(1+z)) - 𝕣.CB1_He2/(𝕣.Tnow*(1+z)) ) / 𝕣.Nnow
        rhs = rhs*1.  # ratio of g's is 1 for He++ <-> He+
        x0 = 0.5 * (sqrt( (rhs-1-𝕣.fHe)^2 + 4*(1+2𝕣.fHe)*rhs) - (rhs-1-𝕣.fHe) )
        x_He0 = (x0 - 1.)/𝕣.fHe
        return x_He0
    elseif (z > 3500.)
        return 1.0
    elseif (z > rhist.z_He_evo_start)
        rhs = exp(1.5 * log(𝕣.CR*𝕣.Tnow/(1+z)) - 𝕣.CB1_He1/(𝕣.Tnow*(1+z)) ) / 𝕣.Nnow
        rhs = rhs*4  # ratio of g's is 4 for He+ <-> He0
        x0 = 0.5 * ( sqrt( (rhs-1)^2 + 4*(1+𝕣.fHe)*rhs ) - (rhs-1))
	    x_He0 = (x0 - 1.)/𝕣.fHe
        return x_He0
    elseif (z > rhist.z_H_He_evo_start)
        return get_He_from_He_evo(𝕣, z, rhist.sol_He)
    else
        return get_He_from_H_He_evo(𝕣, z, rhist.sol_H_He)
    end
end
# ^^^^^

### OTHER STUFF
Tᵧ::Float64 = 2.73 # kelvin * 10^4
SAMPLE_SIZE = 40
z₀::Float64 = 3000.
z₁::Float64 = 700.

a(z) = 1/(1 + z)
z(a) = 1/a - 1
T₄(a) = Tᵧ * (1/a) # kelvin * 10^4 trivial temperature -- REPLACE --

aspan = (a(z₀), a(z₁));
asteps = range(aspan[1], aspan[2], length=SAMPLE_SIZE);

zsteps = z.(asteps);

data_h = later_X_H.((rhist,), zsteps);
data_he = later_X_He.((rhist,), zsteps);
data_t4 = T₄.(asteps);

output = Dict("H" => data_h,
            "He" => data_he,
            "a" => asteps,
            "T4" => data_t4);

open("./Saha++/RecfastData.json", "w") do f
    JSON.print(f, output)
end