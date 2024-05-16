using Test

"""
bernstein.jl
"""

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

Does not work for N > 20 because of factorial() limitations.

# Arguments
- `::Tri`: Tri structure
- `N::Integer`: Bernstein basis degree
- `r,s,t::AbstractArray{T,1}`: Np-sized vectors of distinct 2D barycentric coordinates to be used as interpolatory points
"""
function bernstein_basis(::Tri, N, r, s, t)
    V = hcat([(factorial(N)/(factorial(i) * factorial(j) * factorial(N - i - j))) .* r.^i .* s.^j .* t.^(N - i - j) for i in 0:N for j in 0:N-i]...)
    Vi = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[1] for i in 0:N for j in 0:N-i]...)
    Vj = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[2] for i in 0:N for j in 0:N-i]...)
    Vk = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[3] for i in 0:N for j in 0:N-i]...)
    # drop small values
    map!(x -> isapprox(x, 0, atol=1e-12) ? 0 : x, V, V)
    map!(x -> isapprox(x, 0, atol=1e-12) ? 0 : x, Vi, Vi)
    map!(x -> isapprox(x, 0, atol=1e-12) ? 0 : x, Vj, Vj)
    map!(x -> isapprox(x, 0, atol=1e-12) ? 0 : x, Vk, Vk)
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



function evaluate_2dbernstein_derivative_matrices(N)
    r, s = nodes(Tri(), N)
    coords = transpose(hcat(r,s))
    bary_coords = cartesian_to_barycentric(coords)
    V, Vi, Vj, Vk = bernstein_basis(Tri(), N, bary_coords[1,:], bary_coords[2,:], bary_coords[3,:])
    return V \ Vi, V \ Vj, V \ Vk
end

"""
new content
"""


"Q: what to put here? integer? or just T...?"
struct Bernstein2DDerivativeMatrix <: AbstractArray{Int, 2}
    N::Int
    dir::Int
end

function Base.size(D::Bernstein2DDerivativeMatrix)
    Np = div((D.N + 1) * (D.N + 2), 2)
    return (Np, Np)
end

function Base.getindex(D::Bernstein2DDerivativeMatrix, m::Int, n::Int)
    a_m = bernstein_scalar_to_multiindex(m, D.N, 2)
    a_n = bernstein_scalar_to_multiindex(n, D.N, 2)

    row_lookup = bernstein_2d_derivative_index_to_coefficient(a_n, D.dir)
    return get(row_lookup, a_m, 0)
end

"""
b - fixed column, multiindex
dir - direction

returns dictionary
"""
function bernstein_2d_derivative_index_to_coefficient(b, dir)
    "Q: what would be better for indexing? 0, 1, 2 like the paper? or 1, 2, 3? or string?"
    if dir == 0
        return Dict(
            (b[1],      b[2],       b[3])       => b[1],
            (b[1] - 1,  b[2] + 1,   b[3])       => b[2] + 1,
            (b[1] - 1,  b[2],       b[3] + 1)   => b[3] + 1,)
    elseif dir == 1
        return Dict(
            (b[1] + 1,  b[2] - 1,   b[3])       => b[1] + 1,
            (b[1],      b[2],       b[3])       => b[2],
            (b[1],      b[2] - 1,   b[3] + 1)   => b[3] + 1,)
    elseif dir == 2
        return Dict(
            (b[1] + 1,  b[2],       b[3] - 1)   => b[1] + 1,
            (b[1],      b[2] + 1,   b[3] - 1)   => b[2] + 1,
            (b[1],      b[2],       b[3])       => b[3],)
    end
end

"""
    bernstein_multiindex_to_scalar(a, N)

Finds the scalar index of the Bernstein basis function defined by a, according to dictionary order.
"""
function bernstein_multiindex_to_scalar(a, N)
    dim = length(a) - 1
    if dim == 2
        lookup = bernstein_2d_scalar_to_multiindex_lookup(N)
    elseif dim == 3
        lookup = bernstein_3d_scalar_to_multiindex_lookup(N)
    else
        throw(DomainError(a, "Multiindex is not two or three dimensional."))
    end
    
    return findfirst(tuple -> tuple == a, lookup)
end

function bernstein_scalar_to_multiindex(index, N, dim)
    if dim == 2
        lookup = bernstein_2d_scalar_to_multiindex_lookup(N)
    elseif dim == 3
        lookup = bernstein_3d_scalar_to_multiindex_lookup(N)
    else
        throw(DomainError(a, "Multiindex is not two or three dimensional."))
    end

    return lookup[index]
end

function bernstein_2d_scalar_to_multiindex_lookup(N)
    table = [(i,j,k) for i in 0:N for j in 0:N for k in 0:N]
    return filter(tup -> tup[1] + tup[2] + tup[3] == N, table)
end

function bernstein_3d_scalar_to_multiindex_lookup(N)
    table = [(i,j,k,l) for i in 0:N for j in 0:N for k in 0:N for l in 0:N]
    return filter(tup -> tup[1] + tup[2] + tup[3] + tup[4] == N, table)
end

@testset "Bernstein derivative verification" begin
    @test evaluate_2dbernstein_derivative_matrices(3)[1] ≈ Bernstein2DDerivativeMatrix(3,0)
    @test evaluate_2dbernstein_derivative_matrices(3)[2] ≈ Bernstein2DDerivativeMatrix(3,1)
    @test evaluate_2dbernstein_derivative_matrices(3)[3] ≈ Bernstein2DDerivativeMatrix(3,2)

    @test evaluate_2dbernstein_derivative_matrices(5)[1] ≈ Bernstein2DDerivativeMatrix(5,0)
    @test evaluate_2dbernstein_derivative_matrices(5)[2] ≈ Bernstein2DDerivativeMatrix(5,1)
    @test evaluate_2dbernstein_derivative_matrices(5)[3] ≈ Bernstein2DDerivativeMatrix(5,2)
end

