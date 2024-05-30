using NodesAndModes
using Plots
using LinearAlgebra
using Test
using BenchmarkTools
# Degree elevation operator E^N_{N-1}
struct ElevationMatrix{N} <: AbstractMatrix{Float64} end

function Base.size(::ElevationMatrix{N}) where {N}
    return (div((N + 1) * (N + 2), 2), div(N * (N + 1), 2))
end

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
multiindex_to_linear(i, j, k, offset) = @inbounds (min(i,j,k) < 0) ? 1 : ij_to_linear(i, j, offset) 

function bernstein_2d_scalar_multiindex_lookup(N)
    scalar_to_multiindex = [(i,j,N-i-j) for j in 0:N for i in 0:N-j]
    multiindex_to_scalar = Dict(zip(scalar_to_multiindex, collect(1:length(scalar_to_multiindex))))
    return scalar_to_multiindex, multiindex_to_scalar
end

function Base.getindex(::ElevationMatrix{N}, m, n) where {N}
    (i1,j1,k1) = bernstein_2d_scalar_multiindex_lookup(N)[1][m]
    (i2,j2,k2) = bernstein_2d_scalar_multiindex_lookup(N-1)[1][n]
    if ((i1, j1, k1) == (i2 + 1,j2,k2)) return (i2 + 1)/N
    elseif ((i1, j1, k1) == (i2,j2 + 1,k2)) return (j2 + 1)/N
    elseif ((i1, j1, k1) == (i2,j2,k2 + 1)) return (k2 + 1)/N
    else return 0 end
end

ElevationMatrix{7}()

function lift(N)
    L0 = (N + 1)^2/2 * transpose(ElevationMatrix{N+1}()) * ElevationMatrix{N+1}()
    Lf = L0
    E = I
    for i in 1:N
        E = E * ElevationMatrix{N+1-i}() 
        Lf = vcat(Lf, (-1)^i * binomial(N, i) / (1 + i) * transpose(E) * L0)
    end
    return Lf
end

struct LiftMatrix{N, DIR} <: AbstractMatrix{Float64} 
    Lf::Matrix{Float64}
end

LiftMatrix{N, DIR}() where {N, DIR} = LiftMatrix{N, DIR}(lift(N))

function Base.size(::LiftMatrix{N, DIR}) where {N, DIR}
    return (div((N + 1) * (N + 2) * (N + 3), 6), div((N + 1) * (N + 2), 2))
end

function Base.getindex(L::LiftMatrix{N, 3}, m, n) where {N}
    return L.Lf[m, n]
end

@inline function ijk_to_linear(i, j, k) 
    # @fastmath j_offset = (j==0) ? 0 : (0, 4, 7, 9)[j+1] - k # N = 3
    # return @fastmath i + j_offset + (0, 10, 16, 19)[k+1] + 1
    @fastmath j_offset = (j==0) ? 0 : (0, 8, 15, 21, 26, 30, 33, 35)[j+1] - k # # N = 7
    return @fastmath i + j_offset + (0, 36, 64, 85, 100, 110, 116, 119)[k+1] + 1
end

L = lift(7)

N = 7
mat = Matrix{Float64}(undef, 0, div((N + 1) * (N + 2), 2))
transpose(L[ijk_to_linear(7,0,0),:])
for l in 0:N
    for k in 0:N-l
        for j in 0:N-l-k
            i = N-l-k-j
            mat = vcat(mat, L[ijk_to_linear(i,j,k),:])
        end
    end
end

LiftMatrix{7,3}()

@btime lift(7)
