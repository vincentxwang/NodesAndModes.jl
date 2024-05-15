using LinearAlgebra
using Plots
using NodesAndModes
using Test
using BenchmarkTools
using SparseArrays

"""
    bernstein_basis(::Tri, N, r, s, t)

Returns an Np × Np matrix, where V_mn = the n-th Bernstein basis evaluated at the m-th point, followed by its derivative matrices.

We order the Bernstein basis with degrees (i,j,k) by dictionary order.

# Arguments
- `::Tri`: Tri structure
- `N::Integer`: Bernstein basis degree
- `r,s,t::AbstractArray{T,1}`: Np-sized vectors of distinct 2D barycentric coordinates to be used as interpolatory points

TODO:
Can we call evaluate_bernstein_derivatives only once? Can we not compute N! multiple times?
"""
function bernstein_basis(::Tri, N, r, s, t)
    V = hcat([(factorial(N)/(factorial(i) * factorial(j) * factorial(N - i - j))) .* r.^i .* s.^j .* t.^(N - i - j) for i in 0:N for j in 0:N-i]...)
    Vi = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[1] for i in 0:N for j in 0:N-i]...)
    Vj = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[2] for i in 0:N for j in 0:N-i]...)
    Vk = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[3] for i in 0:N for j in 0:N-i]...)
    return V, Vi, Vj, Vk
end
 
"""
    evaluate_bernstein_derivatives(::Tri, N, r, s, t, i, j, k)

Evaluates the derivatives of the (i, j, k)-th 2D Bernstein basis function at points defined by r, s, t. Throws error if ``i + j + k != N ``.

Outputs a vector for each direction.
"""
function evaluate_bernstein_derivatives(::Tri, N, r, s, t, i, j, k)
    if i + j + k != N
        @error("Barycentric coordinates do not sum to total degree.")
    end
    if i > 0
        vi = (factorial(N)/(factorial(i-1) * factorial(j) * factorial(k))) .* r.^(i-1) .* s.^j .* t.^k
    else 
        # vector with all zeros
        vi = 0 .* r
    end
    if j > 0
        vj = (factorial(N)/(factorial(i) * factorial(j-1) * factorial(k))) .* r.^i .* s.^(j-1) .* t.^k
    else 
        # vector with all zeros
        vj = 0 .* s
    end
    if k > 0
        vk = (factorial(N)/(factorial(i) * factorial(j) * factorial(k-1))) .* r.^i .* s.^j .* t.^(k-1)
    else 
        # vector with all zeros
        vk = 0 .* t
    end
    return vi, vj, vk
end

"""
    cartesian_to_barycentric(coords)

Converts a matrix of 2D cartesian coordinates into a matrix of barycentric coordinates.
"""
function cartesian_to_barycentric(coords)
    hcat([[(col[2] + 1) / 2, - (col[1] + col[2])/2, (col[1] + 1) / 2] for col in eachcol(coords)]...)
end



"""
    multiindex_to_1dindex(a, N)

Converts a tuple (multi-index) a into an integer (one-dimensional index) by dictionary order (e.g. (0, 0, 0, 0) 
maps to 1, (0, 0, 0, 1) maps to 2, etc.).
"""
function multiindex_to_1dindex(a, N)
    dim = length(a)
    index = 1
    for i in dim:-1:1
        if a[i] > N
            @error("Multi-index entry greater than N.")
        end
        index += a[i] * (N + 1)^(dim - i)
    end
    return index
end

N = 3
r, s = nodes(Tri(), N)
scatter(r,s)
coords = transpose(hcat(r,s))
bary_coords = cartesian_to_barycentric(coords)
map!(x -> isapprox(x, 0, atol=1e-12) ? 0 : x, bary_coords, bary_coords)

@btime V, Vi, Vj, Vk = bernstein_basis(Tri(), N, bary_coords[1,:], bary_coords[2,:], bary_coords[3,:])
V, Vi, Vj, Vk = bernstein_basis(Tri(), N, bary_coords[1,:], bary_coords[2,:], bary_coords[3,:])
Di = V \ Vi
map!(x -> isapprox(x, 0, atol=1e-12) ? 0 : x, Di, Di)
Dj = V \ Vj
map!(x -> isapprox(x, 0, atol=1e-12) ? 0 : x, Dj, Dj)
Dk = V \ Vk
map!(x -> isapprox(x, 0, atol=1e-12) ? 0 : x, Dk, Dk)


