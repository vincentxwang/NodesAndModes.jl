using Test
using LinearAlgebra
using BenchmarkTools
using NodesAndModes
using MuladdMacro
using Plots

"""
    Bernstein2DDerivativeMatrix{N} <: AbstractMatrix{Number}

An `AbstractArray` subtype that represents the derivative matrix of the 2-dimensional degree N Bernstein basis.
"""
struct Bernstein2DDerivativeMatrix_2D_r{N} <: AbstractMatrix{Number} end
struct Bernstein2DDerivativeMatrix_2D_s{N} <: AbstractMatrix{Number} end

function Base.size(::Bernstein2DDerivativeMatrix_2D_r{N}) where {N}
    Np = div((N + 1) * (N + 2), 2)
    return (Np, Np)
end

Base.size(::Bernstein2DDerivativeMatrix_2D_s{N}) where {N} = size(Bernstein2DDerivativeMatrix_2D_r{N}())

function offsets(N::Integer)
    tup = [0]
    count = 0
    for i in 1:N
        count += N + 2 - i
        push!(tup, count)
    end
    return tuple(tup...)
end

ij_to_linear(i, j, offset) = @inbounds offset[j+1] + i + 1

# indices of non-zero values for i-th direction 2d derivative. other directions can be found through permuations.
function get_coeff(i1, j1, k1, i2, j2, k2)
    if (i1,j1,k1) == (i2,j2,k2) return i1
    elseif (i1+1,j1-1,k1) == (i2,j2,k2) return j1
    elseif (i1+1,j1,k1-1) == (i2,j2,k2) return k1
    else return 0 end
end

function linear_to_ijk_lookup(N)
    return [(i,j,N-i-j) for j in 0:N for i in 0:N-j]
end

function Base.getindex(::Bernstein2DDerivativeMatrix_2D_r{N}, m, n) where {N}
    linear_to_ijk = linear_to_ijk_lookup(N)
    (i1, j1, k1) = linear_to_ijk[m]
    (i2, j2, k2) = linear_to_ijk[n]
    # du/dr = 1/2 du/di - 1/2 du/dk
    return 0.5 * (get_coeff(i1,j1,k1,i2,j2,k2) - get_coeff(k1,j1,i1,k2,j2,i2))
end

function Base.getindex(::Bernstein2DDerivativeMatrix_2D_s{N}, m, n) where {N}
    linear_to_ijk = linear_to_ijk_lookup(N)
    (i1, j1, k1) = linear_to_ijk[m]
    (i2, j2, k2) = linear_to_ijk[n]
    # du/ds = 1/2 du/dj - 1/2 du/dk
    return 0.5 * (get_coeff(j1,i1,k1,j2,i2,k2) - get_coeff(k1,j1,i1,k2,j2,i2))
end

function fast!(out, ::Bernstein2DDerivativeMatrix_2D_r{N}, x) where {N}
    row = 1
    offset = offsets(N)
    @inbounds for j in 0:N
        for i in 0:N-j
            k = N-i-j
            val = 0.0
            x_row = x[row]

            x1 = x_row
            x3 = (i > 0) ? x[ij_to_linear(i-1, j, offset)] : 0.0
            @fastmath val += i * (x1 - x3)

            if j > 0
                x1 = x[ij_to_linear(i+1, j-1, offset)]
                x3 = x[ij_to_linear(i, j-1, offset)]
                @fastmath val += j * (x1 - x3)
            end

            x1 = (k > 0) ? x[ij_to_linear(i+1, j, offset)] : 0.0
            x3 = x_row
            @fastmath val += k * (x1 - x3)

            @fastmath out[row] = 0.5 * val

            row += 1
        end
    end
    return out
end

function fast!(out, ::Bernstein2DDerivativeMatrix_2D_s{N}, x) where {N}
    row = 1
    offset = offsets(N)
    @inbounds for j in 0:N
        for i in 0:N-j
            k = N-i-j
            val = 0
            x_row = x[row]

            if i > 0
                x2 = x[ij_to_linear(i-1, j+1, offset)]
                x3 = x[ij_to_linear(i-1, j, offset)]
                @fastmath val += i * (x2 - x3)
            end

            x2 = x_row
            x3 = (j > 0) ? x[ij_to_linear(i, j-1, offset)] : 0
            @fastmath val += j * (x2 - x3)


            x2 = (k > 0) ? x[ij_to_linear(i, j+1, offset)] : 0
            x3 = x_row
            @fastmath val += k * (x2 - x3)

            @fastmath out[row] = 0.5 * val

            row += 1
        end
    end
    return out
end

"""
load bernstein_derivative.jl and bernstein.jl first
"""
 
@testset "r/s matricies vs. i/j/k matrices" begin
    @test Bernstein2DDerivativeMatrix_2D_r{5}() ≈ 0.5 * ((BernsteinDerivativeMatrix{Tri, 5, 0}() -  BernsteinDerivativeMatrix{Tri, 5, 2}()))
    @test Bernstein2DDerivativeMatrix_2D_s{5}() ≈ 0.5 * ((BernsteinDerivativeMatrix{Tri, 5, 1}() -  BernsteinDerivativeMatrix{Tri, 5, 2}()))

    x_7 = rand(Float64, div((7 + 1) * (7 + 2), 2))
    b_7 = similar(x_7)
    @test 0.5 * (mul!(copy(b_7), evaluate_bernstein_derivative_matrices(Tri(), 7)[1], x_7) - mul!(copy(b_7), evaluate_bernstein_derivative_matrices(Tri(), 7)[3], x_7)) ≈ fast!(copy(b_7), Bernstein2DDerivativeMatrix_2D_r{7}(), x_7)
    @test 0.5 * (mul!(copy(b_7), evaluate_bernstein_derivative_matrices(Tri(), 7)[2], x_7) - mul!(copy(b_7), evaluate_bernstein_derivative_matrices(Tri(), 7)[3], x_7)) ≈ fast!(copy(b_7), Bernstein2DDerivativeMatrix_2D_s{7}(), x_7)
end

@btime fast!(copy(b_7), Bernstein2DDerivativeMatrix_2D_r{7}(), x_7)

spy(Bernstein2DDerivativeMatrix_2D_r{5}()) 




