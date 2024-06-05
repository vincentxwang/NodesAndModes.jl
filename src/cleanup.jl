using NodesAndModes
using Plots
using LinearAlgebra
using Test
using BenchmarkTools
using SparseArrays
using Profile

# Degree elevation operator E^N_{N-1}
struct ElevationMatrix{N} <: AbstractMatrix{Float64} end

function Base.size(::ElevationMatrix{N}) where {N}
    return (div((N + 1) * (N + 2), 2), div(N * (N + 1), 2))
end

function fast!(out, ::ElevationMatrix{N}, x, offset) where {N}
    row = 1
    @inbounds for j in 0:N
        for i in 0:N-j
            k = N-i-j
            val = muladd(i/N, x[multiindex_to_linear(i-1, j, k, offset)], 0.0)
            val = muladd(j/N, x[multiindex_to_linear(i, j-1, k, offset)], val)
            val = muladd(k/N, x[multiindex_to_linear(i, j, k-1, offset)], val)

            out[row] = val # not sure why putting / N above is faster?
            row += 1
        end
    end
    return out
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
    else return 0.0 end
end


@btime ElevationMatrix{7}()

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

function tri_offsets(N)
    tup = [0]
    count = 0
    for i in 1:N
        count += N + 2 - i
        push!(tup, count)
    end
    return tuple(tup...)
end

function tet_offsets(N)
    tup = [0]
    count = 0
    for i in 1:N
        count += (N + 2 - i) * (N + 3 - i) / 2
        push!(tup, count)
    end
    return tuple(tup...)
end

function ijk_to_linear(i,j,k, tri_offsets, tet_offsets)
    return i + tri_offsets[j+1] + 1 + tet_offsets[k+1] - j * k
end
L = lift(4)
spy(L)

# N = 7
# mat = Matrix{Float64}(undef, 0, div((N + 1) * (N + 2), 2))
# transpose(L[ijk_to_linear(7,0,0, tri_offsets(N), tet_offsets(N)),:])
# for k in 0:N
#     for j in 0:N-k
#         for i in 0:N-k-j
#             l = N-i-j-k
#             mat = vcat(mat, transpose(L[ijk_to_linear(i,k,j, tri_offsets(N), tet_offsets(N)),:]))
#         end
#     end
# end


# spy(mat, markersize = 5)
# LiftMatrix{7,3}()

# lift(7)
# @btime lift(7)

struct ReductionMatrix{N} <: AbstractMatrix{Float64} end

function Base.size(::ReductionMatrix{N}) where {N}
    return (div(N * (N + 1), 2), div((N + 1) * (N + 2), 2))
end

function Base.getindex(::ReductionMatrix{N}, m, n) where {N}
    return ElevationMatrix{N}()[n, m]
end

ReductionMatrix{6}()
ElevationMatrix{6}()
@test ReductionMatrix{6}() ≈ transpose(ElevationMatrix{6}())

# pass in offsets(N).
"""
multiplies x by the reduction matrix: N -> N - 1

pass in offsets(Tri(), N)
"""
function reduction_multiply!(out, N, x, offset)
    row = 1
    @inbounds for j in 0:N-1
        for i in 0:N-1-j
            k = N-1-i-j
            val = muladd((i+1), x[multiindex_to_linear(i+1, j, k, offset)], 0.0)
            val = muladd((j+1), x[multiindex_to_linear(i, j+1, k, offset)], val)
            val = muladd((k+1), x[multiindex_to_linear(i, j, k+1, offset)], val)

            # val = 0
            # val += i/N * x[multiindex_to_linear(i-1, j, k, offsets2)]
            # val += j/N * x[multiindex_to_linear(i, j-1, k, offsets2)]
            # val += k/N * x[multiindex_to_linear(i, j, k-1, offsets2)]
            out[row] = val/N # not sure why putting / N above is faster?
            row += 1
        end
    end
    return out
end

x = rand(Float64, 36)

@btime reduction_multiply!($(zeros(28)), $(ReductionMatrix{7}()), $(x), $(offsets(Tri(), 7)))
@test reduction_multiply!(zeros(28), ReductionMatrix{7}(), x, offsets(Tri(), 7)) ≈ transpose(transpose(x) * ElevationMatrix{7}())

""" Multiplication """


"""
L0 - as defined in paper
x - input vector
offsets - precomputed vector of offset tuples generated by offsets(Tri(), N). concatenated vertically, etc.
"""
Npe = ntuple(i -> div((i + 1) * (i + 2), 2), 7)

function fast!(out, ::ElevationMatrix{N}, L0, x, Npe, offsets) where {N}
    E = L0 * x # note L0 is very very sparse, could speedup later, 1 alloc
    index1 = Npe[N]
    out[1:index1] = E
    reduction_multiply!(E, N, E, offsets[N])
    for j in 1:N
        if j == N
            dif = 1
        else
            dif = Npe[N-j]
        end
        index2 = index1 + dif
        # assign the next (N+1-j)(N+2-j)/2 entries as l_j * (E^N_{N_j})^T u^f
        E = E[1:dif]
        out[index1 + 1: index2] .= ((isodd(j) ? -1.0 : 1.0) * factorial(N)/(factorial(N-j) * factorial(j) * (1.0 + j))) .* E
        if j < N
            index1 = index2
            reduction_multiply!(E, N-j, E, offsetting(Val(N-j)))
        end
        # opportunity for further (small) (optimization?): can derive offset[N-1] from offset[N]
    end
    return out
end



offset = offset_table

N = 7
index1 = 36
@btime E = L0 * x # note L0 is very very sparse, could speedup later, 1 alloc
@btime index1 = $(Npe[N])
@btime out[1:index1] = E
@btime reduction_multiply!(E, ReductionMatrix{N}(), E, offsetting(Val(N)))
j=1
@btime dif = Npe[N-j]
@btime index2 = index1 + dif
@btime lj = ((isodd(j) ? -1 : 1) * factorial(N) / (factorial(j) * factorial(N-j) *  (1 + j)))
@btime 1 % 2
E = L0 * x
@btime factorial(7)/factorial(6)
reduction_multiply!(E, ReductionMatrix{N}(), E, offsetting(Val(N)))
@btime E = E[1:dif]
@btime A = isodd(j) * binomial(N, j)
@btime A = ((isodd(j) ? -1.0 : 1.0) * binomial(N, j) / (1.0 + j)) .* E[1:dif]
@btime out[index1 + 1: index2] .= lj .* E[1:dif]
@btime reduction_multiply!(E, $(ReductionMatrix{N-j}()), E, offsetting(Val(N-j)))

for j in 1:N
    if j == N
        dif = 1
    else
        dif = Npe[N-j]
    end
    index2 = index1 + dif
    # assign the next (N+1-j)(N+2-j)/2 entries as l_j * (E^N_{N_j})^T u^f
    out[index1 + 1: index2] .= ((isodd(j) ? -1.0 : 1.0) * binomial(N, j) / (1.0 + j)) .* E[1:dif]
    if j < N
        index1 = index2
        reduction_multiply!(E, ReductionMatrix{N-j}(), E, offsetting(Val(N-j)))
    end
    # opportunity for further (small) (optimization?): can derive offset[N-1] from offset[N]
end

j = 1
dif = div((N + 1 - j) * (N + 2 - j), 2)
index2 = index1 + dif
# assign the next (N+1-j)(N+2-j)/2 entries as l_j * (E^N_{N_j})^T u^f
@btime l_j = binomial(N, j) / (1.0 + j)
@btime 
@btime l_j = 1.0/4.0
@btime out[(index1 + 1): index2] .= l_j .* E[1:diff]

@btime div((N + 1 - 1) * (N + 2 - 1), 2)
@btime Np[7]
N=7


@btime offsetting(Val(7))
offsets(Tri(), 7)

offsetting(::Val{1}) = (0, 2)
offsetting(::Val{2}) = (0, 3, 5)
offsetting(::Val{3}) = (0, 4, 7, 9)
offsetting(::Val{4}) = (0, 5, 9, 12, 14)
offsetting(::Val{5}) = (0, 6, 11, 15, 18, 20)
offsetting(::Val{6}) = (0, 7, 13, 18, 22, 25, 27)
offsetting(::Val{7}) = (0, 8, 15, 21, 26, 30, 33, 35)



offset_table
for j in 0:6
    println(div((N-j) * (N-j + 1), 2), " ", div((N-j + 1) * (N-j + 2), 2) - 1)
    @test offset_table[div((N-j) * (N-j + 1), 2): div((N-j + 1) * (N-j + 2), 2) - 1] == collect(offsets(Tri(), N-j))
end

index1 = index2
@btime if j < N
    reduction_multiply!($(E), $(ReductionMatrix{N-j}()), $(E), $(offset[N-j]))
end
# opportunity for further (small) (optimization?): can derive offset[N-1] from offset[N]

return out

lift(7)

println("fast! vs lift(7) * x")
N = 7
x = rand(Float64, 36)
out = zeros(Float64, 120)
offset_table = ntuple(i -> offsets(Tri(), i), 7)
L0 = (N + 1)^2/2 * transpose(ElevationMatrix{N+1}()) * ElevationMatrix{N+1}()
#the above are precomputed
@btime fast!($(out), $(ElevationMatrix{7}()), $(L0), $(x), $(Npe), offset_table)
@btime mul!($(out), $(lift(7)), $(x))


@btime a = 1

(N * (N + 1) / 2), ((N + 1) * (N + 2) / 2 - 1)
offset_table[28:36]

offsets(Tri(), 2)


@test fast!(out, ElevationMatrix{7}(), L0, x, Npe) ≈ lift(7) * x

@code_warntype fast!(out, ElevationMatrix{7}(), L0, x, Npe, offset_table)



"""profiling"""

N = 7
x = rand(Float64, 36)
out = zeros(Float64, 120)
L0 = (N + 1)^2/2 * transpose(ElevationMatrix{N+1}()) * ElevationMatrix{N+1}()
Elev = ElevationMatrix{N}()
Profile.init(n = 10^7, delay = 0.00001)
@profview for i in 1:20000 fast!(out, Elev, L0, x) end

@time fast!(out, Elev, L0, x)

