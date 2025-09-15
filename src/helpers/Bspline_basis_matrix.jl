"""
    B_spline_basis_matrix(x::AbstractVector{<:Real}, order::Int, dim_basis::Int) 

Compute the design matrix when using a B-spline basis of a given order using `dim_basis` basis functions.

# Returns
* `bx`: The design matrix for the given basis, e.g. bx[i,:]
"""
function B_spline_basis_matrix(x::AbstractVector{<:Real}, order::Int, dim_basis::Int)
    n = length(x)
    b = BSplineBasis(BSplineOrder(order), LinRange(0, 1, dim_basis - order + 2))
    b_eval = b.(x)
    bx = zeros(Float64, n, length(b))

    # Get B-spline basis matrix
    for i in eachindex(b_eval)
        j, val = b_eval[i]
        ind = j:-1:j-(order-1) # this library returns the evaluated basis funcs in "reverse" order.
        bx[i, ind] .= val
    end
    return bx
end