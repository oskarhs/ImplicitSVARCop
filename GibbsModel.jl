using AbstractPPL

f(y,z) = sleep(0.02)

cond = Dict{VarName, Any}()
sig = Dict{VarName, Vector{Symbol}}()
cond[@varname(x)] = f
sig[@varname(x)] = [:y, :z]

macro insert(vname, cond, sig)
    return quote
        func = $(cond)[$vname]
        args = $(sig)[$vname]
        func($(args...))     
    end |> esc
end

# We do not have to store these in dictionaries. Could do vectors instead in the Gibbs structure.

vn = @varname(x)
y = 10
z = 10
@insert(vn, cond, sig) # Performance penalty is small so long as the sampling/computation is somewhat expensive

function f(x)
    return 2*x
end
x = 2

vn = @varname(x)
@insert vn


y = 2
@insert y