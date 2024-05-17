using Test
using LinearAlgebra
using BenchmarkTools

"""
    Bernstein2DDerivativeMatrix{N, DIR} <: AbstractArray{Int, 2}

An `AbstractArray` subtype that represents the derivative matrix of the 2-dimensional degree N Bernstein basis.

The terms of the Bernstein basis are ordered dictionary-ordered by multiindex.

# Type Parameters
- `N::Int`: Bernstein basis degree
- `DIR::Integer`: Direction of derivative. Only `0,1,2` permissible with `(i,j,k)` directions respectively.

# values
- `lookup::Vector{Tuple{Int, Int, Int}}`: Vector that maps scalars to multiindex. Should not be manually specified.
"""
struct Bernstein2DDerivativeMatrix{N, DIR} <: AbstractMatrix{Int}
    lookup::Vector{Tuple{Int, Int, Int}}
end

function Bernstein2DDerivativeMatrix{N, DIR}() where {N, DIR}
    lookup = bernstein_2d_scalar_to_multiindex_lookup(N)
    return Bernstein2DDerivativeMatrix{N, DIR}(lookup)
end

function Base.size(::Bernstein2DDerivativeMatrix{N, DIR}) where {N, DIR}
    Np = div((N + 1) * (N + 2), 2)
    return (Np, Np)
end

function Base.getindex(::Bernstein2DDerivativeMatrix{N, DIR}, m::Int, n::Int) where {N, DIR}
    a_m = bernstein_scalar_to_multiindex(m, N, 2)
    a_n = bernstein_scalar_to_multiindex(n, N, 2)

    row_lookup = bernstein_2d_derivative_index_to_coefficient(a_n, DIR)
    return get(row_lookup, a_m, 0)
end

"""
    bernstein_2d_derivative_index_to_coefficient(b, dir)

Returns a dictionary that maps row multiindices to coefficients of the 2D Bernstein derivative matrix in a fixed column b.
"""
function bernstein_2d_derivative_index_to_coefficient(b, dir)
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

Finds the scalar index of the 2D/3D Bernstein basis function defined by the multi-index a, according to dictionary order.
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

"""
    bernstein_2d_scalar_to_multiindex_lookup(N)

Creates a vector where the `i`-th scalar index maps to the corresponding multi-index.
"""
function bernstein_2d_scalar_to_multiindex_lookup(N)
    table = [(i,j,k) for i in 0:N for j in 0:N for k in 0:N]
    return filter(tup -> tup[1] + tup[2] + tup[3] == N, table)
end

function bernstein_3d_scalar_to_multiindex_lookup(N)
    table = [(i,j,k,l) for i in 0:N for j in 0:N for k in 0:N for l in 0:N]
    return filter(tup -> tup[1] + tup[2] + tup[3] + tup[4] == N, table)
end

"""
TODO: currently, this only implements the 0-direction derivative"
rewrites out to compute A x
"""
function fast!(out::Vector{T}, A::Bernstein2DDerivativeMatrix{N, DIR}, x::Vector{T}) where {T, N, DIR}
    Np = length(A.lookup)
    for index in 1:Np
        (i,j,k) = A.lookup[index]
        coeff = i
        if j >= 1 coeff += j end
        if k >= 1 coeff += k end
        out[index] = coeff * x[index]
    end
    return out
end

@testset "2D Bernstein derivative verification" begin
    @test evaluate_2dbernstein_derivative_matrices(3)[1] ≈ Bernstein2DDerivativeMatrix{3,0}()
    @test evaluate_2dbernstein_derivative_matrices(3)[2] ≈ Bernstein2DDerivativeMatrix{3,1}()
    @test evaluate_2dbernstein_derivative_matrices(3)[3] ≈ Bernstein2DDerivativeMatrix{3,2}()

    @test evaluate_2dbernstein_derivative_matrices(5)[1] ≈ Bernstein2DDerivativeMatrix{5,0}()
    @test evaluate_2dbernstein_derivative_matrices(5)[2] ≈ Bernstein2DDerivativeMatrix{5,1}()
    @test evaluate_2dbernstein_derivative_matrices(5)[3] ≈ Bernstein2DDerivativeMatrix{5,2}()
end

@testset "2D Bernstein fast! verification" begin
    x_3 = rand(Float64, div((3 + 1) * (3 + 2), 2))
    x_5 = rand(Float64, div((5 + 1) * (5 + 2), 2))
    x_7 = rand(Float64, div((7 + 1) * (7 + 2), 2))

    b_3 = similar(x_3)
    b_5 = similar(x_5)
    b_7 = similar(x_7)

    @test mul!(b_3, evaluate_2dbernstein_derivative_matrices(3)[1], x_3) ≈ fast!(b_3, Bernstein2DDerivativeMatrix{3,0}(), x_3)
    @test mul!(b_5, evaluate_2dbernstein_derivative_matrices(5)[2], x_5) ≈ fast!(b_5, Bernstein2DDerivativeMatrix{5,0}(), x_5)
    @test mul!(b_7, evaluate_2dbernstein_derivative_matrices(7)[3], x_7) ≈ fast!(b_7, Bernstein2DDerivativeMatrix{7,0}(), x_7)
end

"Benchmarks"

"zero optimization, N = 3, N = 5, N = 7"

function test(N)
    x_N = rand(Float64, div((N + 1) * (N + 2), 2))
    b_N = similar(x_N)
    A = evaluate_2dbernstein_derivative_matrices(N)[1]
    B = Bernstein2DDerivativeMatrix{N, 0}()

    @btime mul!($b_N, $A, $x_N)
    @btime fast!($b_N, $B, $x_N)
end

test(3)
test(5)
test(7)
test(15)
