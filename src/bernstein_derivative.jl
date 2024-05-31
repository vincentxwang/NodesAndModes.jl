using Test
using LinearAlgebra
using BenchmarkTools
using NodesAndModes
using MuladdMacro

"""
TODO:
1) optimize 2D multiplication with unrolling?
2) implement 3D matrices
"""

"""
    BernsteinDerivativeMatrix{ELEM, N, DIR} <: AbstractMatrix{Int}

An `AbstractArray` subtype that represents the derivative matrix of the 2-dimensional degree N Bernstein basis.

The terms of the Bernstein basis are ordered by reverse-dictionary order by all terms in the multiindex excluding the last term.

# Type Parameters
- `ELEM::Union{Tri, Tet}`: Tri -> 2D, Tet -> 3D
- `N::Int`: Bernstein basis degree
- `DIR::Int`: Direction of derivative. Only `0,1,2` permissible with `(i,j,k)` directions respectively for 2D, 
and similarly 0..3 for 3D
"""
struct BernsteinDerivativeMatrix{ELEM, N, DIR} <: AbstractMatrix{Int} end

"Container for matrices in all directions"
struct BernsteinDerivativeMatrices{ELEM, N}
    matrices::Tuple{Vararg{BernsteinDerivativeMatrix{ELEM, N, Int}}}
end

function create_derivative_matrices(::Type{Tri}, N::Int)
    return BernsteinDerivativeMatrices{Tri, N}(
        (
            BernsteinDerivativeMatrix{Tri, N, 0}(),
            BernsteinDerivativeMatrix{Tri, N, 1}(),
            BernsteinDerivativeMatrix{Tri, N, 2}()
        )
    )
end

function create_derivative_matrices(::Type{Tet}, N::Int)
    return BernsteinDerivativeMatrices{Tet, N}(
        (
            BernsteinDerivativeMatrix{Tet, N, 0}(),
            BernsteinDerivativeMatrix{Tet, N, 1}(),
            BernsteinDerivativeMatrix{Tet, N, 2}(),
            BernsteinDerivativeMatrix{Tet, N, 3}()
        )
    )
end

function Base.size(::BernsteinDerivativeMatrix{ELEM, N, DIR}) where {ELEM<:Tri, N, DIR}
    Np = div((N + 1) * (N + 2), 2)
    return (Np, Np)
end

# indices of non-zero values for i-th direction 2d derivative. others can be found through permutations.
function get_coeff(i1, j1, k1, i2, j2, k2)
    if (i1,j1,k1) == (i2,j2,k2) return i1
    elseif (i1+1,j1-1,k1) == (i2,j2,k2) return j1
    elseif (i1+1,j1,k1-1) == (i2,j2,k2) return k1
    else return 0 end
end

function Base.getindex(::BernsteinDerivativeMatrix{ELEM, N, DIR}, m, n) where {ELEM<:Tri, N, DIR}
    scalar_to_multiindex = bernstein_2d_scalar_multiindex_lookup(N)[1]
    (i1,j1,k1) = scalar_to_multiindex[m]
    (i2,j2,k2) = scalar_to_multiindex[n]
    if DIR == 0 return get_coeff(i1,j1,k1,i2,j2,k2) end
    if DIR == 1 return get_coeff(j1,i1,k1,j2,i2,k2) end
    if DIR == 2 return get_coeff(k1,j1,i1,k2,j2,i2) end
end

"""
    bernstein_2d_scalar_multiindex_lookup(N)

Returns a vector that maps scalar indices -> multi-indices.
"""
function bernstein_2d_scalar_multiindex_lookup(N)
    scalar_to_multiindex = [(i,j,N-i-j) for j in 0:N for i in 0:N-j]
    multiindex_to_scalar = Dict(zip(scalar_to_multiindex, collect(1:length(scalar_to_multiindex))))
    return scalar_to_multiindex, multiindex_to_scalar
end

function bernstein_3d_scalar_to_multiindex_lookup(N)
    return [(i,j,k,N-i-j) for k in 0:N for j in 0:N-k for i in 0:N-j-k]
end

"Multiplication algorithm"

function offsets(::Tri, N::Integer)
    count = div((N + 1) * (N + 2), 2)
    tup = []
    for i in 1:(N+1)
        count -= i
        push!(tup, count)
    end
    reverse!(tup)
    return tuple(tup...)
end

ij_to_linear(i, j, offset) = @inbounds max(0, i + offset[j+1]) + 1
multiindex_to_linear(i, j, k, offset) = @inbounds (min(i,j,k) < 0) ? 0 : ij_to_linear(i, j, offset) # if any index is negative, the coeff = 0, so we just return 1

function fast!(out::Vector{Float64}, ::BernsteinDerivativeMatrix{ELEM, N, 0}, x::Vector{Float64}, offset) where {ELEM<:Tri, N}
    out .= 0.0
    row = 1
    @inbounds for j in 0:N 
        for i in 0:N-j
            k = N-i-j
            # (i,j,k) term (diagonal)
            val = muladd(i, x[row], 0)

            # (i+1,j-1,k) term
            val = muladd(j, x[multiindex_to_linear(i+1, j-1, k, offset)], val)
            
            # (i+1,j,k-1) term
            val = muladd(k, x[multiindex_to_linear(i+1, j, k-1, offset)], val)

            out[row] = val

            row += 1
        end
    end
    return out
end

function fast!(out::Vector{Float64}, ::BernsteinDerivativeMatrix{ELEM, N, 1}, x::Vector{Float64}, offset) where {ELEM<:Tri, N}
    out .= 0.0
    row = 1
    @inbounds for j in 0:N 
        for i in 0:N-j
            k = N-i-j
            val = muladd(j, x[row], 0)
            val = muladd(i, x[multiindex_to_linear(i-1, j+1, k, offset)], val)
            val = muladd(k, x[multiindex_to_linear(i, j+1, k-1, offset)], val)
            out[row] = val
            row += 1
        end
    end
    return out
end

function fast!(out::Vector{Float64}, ::BernsteinDerivativeMatrix{ELEM, N, 2}, x::Vector{Float64}, offset) where {ELEM<:Tri, N}
    row = 1
    @inbounds for j in 0:N 
        for i in 0:N-j
            k = N-i-j
            val = muladd(k, x[row], 0)
            val = muladd(i, x[multiindex_to_linear(i-1, j, k+1, offset)], val)
            val = muladd(j, x[multiindex_to_linear(i, j-1, k+1, offset)], val)
            out[row] = val
            row += 1
        end
    end
    return out
end

"Tests"

@testset "2D Bernstein derivative verification" begin
    @test evaluate_bernstein_derivative_matrices(Tri(), 3)[1] ≈ BernsteinDerivativeMatrix{Tri,3,0}()
    @test evaluate_bernstein_derivative_matrices(Tri(), 3)[2] ≈ BernsteinDerivativeMatrix{Tri,3,1}()
    @test evaluate_bernstein_derivative_matrices(Tri(), 3)[3] ≈ BernsteinDerivativeMatrix{Tri,3,2}()

    @test evaluate_bernstein_derivative_matrices(Tri(), 5)[1] ≈ BernsteinDerivativeMatrix{Tri,5,0}()
    @test evaluate_bernstein_derivative_matrices(Tri(), 5)[2] ≈ BernsteinDerivativeMatrix{Tri,5,1}()
    @test evaluate_bernstein_derivative_matrices(Tri(), 5)[3] ≈ BernsteinDerivativeMatrix{Tri,5,2}()
end

@testset "2D Bernstein fast! verification" begin
    x_3 = rand(Float64, div((3 + 1) * (3 + 2), 2))
    x_5 = rand(Float64, div((5 + 1) * (5 + 2), 2))
    x_7 = rand(Float64, div((7 + 1) * (7 + 2), 2))

    b_3 = similar(x_3)
    b_5 = similar(x_5)
    b_7 = similar(x_7)

    @test mul!(copy(b_3), evaluate_bernstein_derivative_matrices(Tri(), 3)[1], x_3) ≈ fast!(copy(b_3), BernsteinDerivativeMatrix{Tri,3,0}(), x_3, offsets(Tri(), 3))
    @test mul!(copy(b_5), evaluate_bernstein_derivative_matrices(Tri(), 5)[2], x_5) ≈ fast!(copy(b_5), BernsteinDerivativeMatrix{Tri,5,1}(), x_5, offsets(Tri(), 5))
    @test mul!(copy(b_7), evaluate_bernstein_derivative_matrices(Tri(), 7)[3], x_7) ≈ fast!(copy(b_7), BernsteinDerivativeMatrix{Tri,7,2}(), x_7, offsets(Tri(), 7))
end

"Benchmarks"

function test(N)
    x_N = rand(Float64, div((N + 1) * (N + 2), 2))
    b_N = similar(x_N)
    A = evaluate_bernstein_derivative_matrices(Tri(), N)[1]
    B = BernsteinDerivativeMatrix{Tri, N, 0}()
    offset = offsets(Tri(), N)
    println("2D mul! vs fast!, N = ", N)
    @btime mul!($b_N, $A, $x_N)
    @btime fast!($b_N, $B, $x_N, $offset)
end

test(3)
test(5)
test(7)
test(9)
test(15)
test(20)
