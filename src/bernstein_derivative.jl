using Test
using LinearAlgebra
using BenchmarkTools
using NodesAndModes
using MuladdMacro

"""
    Bernstein2DDerivativeMatrix{N, DIR} <: AbstractMatrix{Int}

An `AbstractArray` subtype that represents the derivative matrix of the 2-dimensional degree N Bernstein basis.

The terms of the Bernstein basis are ordered dictionary-ordered by multiindex.

# Type Parameters
- `N::Int`: Bernstein basis degree
- `DIR::Integer`: Direction of derivative. Only `0,1,2` permissible with `(i,j,k)` directions respectively.

# values
- `scalar_to_multiindex::Vector{Tuple{Int, Int, Int}}`: Maps scalars to multiindex. 
- `multiindex_to_scalar::Dict}`: Maps multiindex to scalars.
- `lil_matrix::Vector{Vector{Tuple{Int, Int}}}`: Stores the matrix using a list-of-lists format, storing one list per row, with each entry containing a 
   tuple `(column index, value)`.
"""
struct Bernstein2DDerivativeMatrix{N, DIR} <: AbstractMatrix{Int}
    lil_matrix::Vector{Vector{Tuple{Int, Int}}}
end

function Bernstein2DDerivativeMatrix{N, DIR}() where {N, DIR}
    scalar_to_multiindex, multiindex_to_scalar = bernstein_2d_scalar_multiindex_lookup(N)
    lil_matrix = []
    for index in 1:length(scalar_to_multiindex) #1:Np
        (i,j,k) = scalar_to_multiindex[index]
        entries = [(index, i)]
        if j >= 1 push!(entries, (multiindex_to_scalar[(i+1,j-1,k)], j)) end
        if k >= 1 push!(entries, (multiindex_to_scalar[(i+1,j,k-1)], k)) end
        push!(lil_matrix, entries)
    end
    return Bernstein2DDerivativeMatrix{N, DIR}(lil_matrix)
end

function Base.size(::Bernstein2DDerivativeMatrix{N, DIR}) where {N, DIR}
    Np = div((N + 1) * (N + 2), 2)
    return (Np, Np)
end

function Base.getindex(A::Bernstein2DDerivativeMatrix{N, DIR}, m::Int, n::Int) where {N, DIR}
    idx = findfirst(tup -> tup[1] == n, A.lil_matrix[m])
    if isnothing(idx)
        return 0
    else 
        return A.lil_matrix[m][idx][2]
    end
end
# is there a findfirst that returns the value


"""
    bernstein_multiindex_to_scalar(a, N)

Finds the scalar index of the 2D/3D Bernstein basis function defined by the multi-index a, according to dictionary order.
"""
function bernstein_multiindex_to_scalar(a, N)
    dim = length(a) - 1
    if dim == 2
        lookup = bernstein_2d_scalar_multiindex_lookup(N)[1]
    elseif dim == 3
        lookup = bernstein_3d_scalar_to_multiindex_lookup(N)
    else
        throw(DomainError(a, "Multiindex is not two or three dimensional."))
    end
    
    return findfirst(tuple -> tuple == a, lookup)
end

function bernstein_scalar_to_multiindex(index, N, dim)
    if dim == 2
        lookup = bernstein_2d_scalar_multiindex_lookup(N)[1]
    elseif dim == 3
        lookup = bernstein_3d_scalar_to_multiindex_lookup(N)
    else
        throw(DomainError(a, "Multiindex is not two or three dimensional."))
    end

    return lookup[index]
end

"""
    bernstein_2d_scalar_multiindex_lookup(N)

Returns a vector that maps scalar indices -> multi-indices.
"""
function bernstein_2d_scalar_multiindex_lookup(N)
    scalar_to_multiindex = [(i,j,k) for i in 0:N for j in 0:N for k in 0:N]
    scalar_to_multiindex = filter(tup -> tup[1] + tup[2] + tup[3] == N, scalar_to_multiindex)
    multiindex_to_scalar = Dict(zip(scalar_to_multiindex, collect(1:length(scalar_to_multiindex))))
    return scalar_to_multiindex, multiindex_to_scalar
end

function bernstein_3d_scalar_to_multiindex_lookup(N)
    table = [(i,j,k,l) for i in 0:N for j in 0:N for k in 0:N for l in 0:N]
    return filter(tup -> tup[1] + tup[2] + tup[3] + tup[4] == N, table)
end

"""
TODO: currently, this only implements the 0-direction derivative"
rewrites out to compute A x
"""
ij_to_linear(i, j) = max(0, i + (0, 4, 7, 9)[j+1]) + 1
multiindex_to_linear(i, j, k) = (min(i,j,k) < 0) ? 1 : ij_to_linear(i, j) # if any index is negative, the coeff = 0, so we just return 1

function fast!(out::Vector{Float64}, A::Bernstein2DDerivativeMatrix{N, DIR}, x::Vector{Float64}) where {N, DIR}
    row = 1
    for j in 0:N 
        for i in 0:N-j
            k = N-i-j
            val = out[row]

            # (i,j,k) term (diagonal)
            @inbounds muladd(i, x[row], val)

            # (i+1,j-1,k) term
            @inbounds muladd(j, x[multiindex_to_linear(i+1, j-1, k)], val)
            
            # (i+1,j,k-1) term
            @inbounds muladd(k, x[multiindex_to_linear(i+1, j, k-1)], val)        

            out[row] = val

            row += 1
        end
    end
end


@testset "2D Bernstein derivative verification" begin
    @test evaluate_bernstein_derivative_matrices(Tri(), 3)[1] ≈ Bernstein2DDerivativeMatrix{3,0}()
    # @test evaluate_bernstein_derivative_matrices(Tri(), 3)[2] ≈ Bernstein2DDerivativeMatrix{3,1}()
    # @test evaluate_bernstein_derivative_matrices(Tri(), 3)[3] ≈ Bernstein2DDerivativeMatrix{3,2}()

    @test evaluate_bernstein_derivative_matrices(Tri(), 5)[1] ≈ Bernstein2DDerivativeMatrix{5,0}()
    # @test evaluate_bernstein_derivative_matrices(Tri(), 5)[2] ≈ Bernstein2DDerivativeMatrix{5,1}()
    # @test evaluate_bernstein_derivative_matrices(Tri(), 5)[3] ≈ Bernstein2DDerivativeMatrix{5,2}()
end

@testset "2D Bernstein fast! verification" begin
    x_3 = rand(Float64, div((3 + 1) * (3 + 2), 2))
    x_5 = rand(Float64, div((5 + 1) * (5 + 2), 2))
    x_7 = rand(Float64, div((7 + 1) * (7 + 2), 2))

    b_3 = similar(x_3)
    b_5 = similar(x_5)
    b_7 = similar(x_7)

    @test mul!(copy(b_3), evaluate_bernstein_derivative_matrices(Tri(), 3)[1], x_3) ≈ fast!(copy(b_3), Bernstein2DDerivativeMatrix{3,0}(), x_3)
    @test mul!(copy(b_5), evaluate_bernstein_derivative_matrices(Tri(), 5)[1], x_5) ≈ fast!(copy(b_5), Bernstein2DDerivativeMatrix{5,0}(), x_5)
    @test mul!(copy(b_7), evaluate_bernstein_derivative_matrices(Tri(), 7)[1], x_7) ≈ fast!(copy(b_7), Bernstein2DDerivativeMatrix{7,0}(), x_7)
end

"Benchmarks"

function test(N)
    x_N = rand(Float64, div((N + 1) * (N + 2), 2))
    b_N = similar(x_N)
    A = evaluate_bernstein_derivative_matrices(Tri(), N)[1]
    B = Bernstein2DDerivativeMatrix{N, 0}()
    println("mul! vs fast!, N = ", N)
    @btime mul!($b_N, $A, $x_N)
    @btime fast!($b_N, $B, $x_N)
end

test(3)
test(5)
test(7)
test(9)
test(15)
test(20)
