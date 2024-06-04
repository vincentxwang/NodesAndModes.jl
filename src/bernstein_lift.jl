using NodesAndModes
using Plots
using LinearAlgebra
using Test
using BenchmarkTools
using SparseArrays

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

            # val = 0
            # val += i/N * x[multiindex_to_linear(i-1, j, k, offsets2)]
            # val += j/N * x[multiindex_to_linear(i, j-1, k, offsets2)]
            # val += k/N * x[multiindex_to_linear(i, j, k-1, offsets2)]
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
function fast!(out, ::ReductionMatrix{N}, x, offset) where {N}
    row = 1
    for j in 0:N-1
        for i in 0:N-1-j
            k = N-1-i-j
            val = muladd((i+1)/N, x[multiindex_to_linear(i+1, j, k, offset)], 0.0)
            val = muladd((j+1)/N, x[multiindex_to_linear(i, j+1, k, offset)], val)
            val = muladd((k+1)/N, x[multiindex_to_linear(i, j, k+1, offset)], val)

            # val = 0
            # val += i/N * x[multiindex_to_linear(i-1, j, k, offsets2)]
            # val += j/N * x[multiindex_to_linear(i, j-1, k, offsets2)]
            # val += k/N * x[multiindex_to_linear(i, j, k-1, offsets2)]
            out[row] = val # not sure why putting / N above is faster?
            row += 1
        end
    end
    return out
end

x = rand(Float64, 36)

@test fast!(zeros(28), ReductionMatrix{7}(), x, offsets(Tri(), 7)) ≈ transpose(transpose(x) * ElevationMatrix{7}())




""" Multiplication """

x = rand(Float64, 36)
@btime $(lift(7)) * $(x)

N=7
L0 = (N + 1)^2/2 * transpose(ElevationMatrix{N+1}()) * ElevationMatrix{N+1}()
spy(L0)

function fast!(out, ::ElevationMatrix{N}, x, L0) where {N}
    index1 = div((N + 1) *(N + 2), 2)
    @inbounds out[1:index1] = L0 * x
    E = transpose(ElevationMatrix{N}()) * L0 * x
    @inbounds for j in 1:N
        index2 = index1 + div((N + 1 - j) * (N + 2 - j), 2)
        out[(index1 + 1): index2] = @fastmath ((isodd(j) ? -1 : 1) * binomial(N, j) / (1 + j)) * E
        index1 = index2
        E = transpose(ElevationMatrix{N-j}()) * E
    end
    return out
end
out = zeros(Float64, 120)

@btime $(L0) * $(x)
println("fast! vs lift(7) * x")
@btime fast!($(out), $(ElevationMatrix{7}()), $(x), $(L0))
@btime $(lift(7)) * $(x)

ElevationMatrix{7}()

fast!(out, ElevationMatrix{7}(), x, L0)[37]

transpose(ElevationMatrix{7}()) * x * 7 / 2

(lift(7) * x)[37]


ElevationMatrix{7}()

L0 * ElevationMatrix{N}()


"""
new
"""

using StartUpDG 
using LinearAlgebra
using SparseArrays
using Plots


cartesian_to_barycentric(elem, coords...) = 
    cartesian_to_barycentric(elem, vcat(permutedims.(coords)...))

# coords = [x y z]' = 3 x num_points
function cartesian_to_barycentric(::Tri, coords::AbstractMatrix)
    bary = hcat([[(col[1] + 1) / 2,
                  (col[2] + 1) / 2, 
                 -(col[1] + col[2]) / 2, 
                  ] for col in eachcol(coords)]...)
    return (bary[i,:] for i in 1:3)    
end

function cartesian_to_barycentric(::Tet, coords::AbstractMatrix)
    bary = hcat([[(1 + col[1]) / 2, 
                  (1 + col[2]) / 2, 
                  (1 + col[3]) / 2,
                 -(1 + col[1] + col[2] + col[3]) / 2] for col in eachcol(coords)]...)
    return (bary[i,:] for i in 1:4)
end

bernstein_basis(elem::Tri, N, r, s) = 
    bernstein_basis(elem, N, cartesian_to_barycentric(elem, r, s)...)

bernstein_basis(elem::Tet, N, r, s, t) = 
    bernstein_basis(elem, N, cartesian_to_barycentric(elem, r, s, t)...)

function bernstein_basis(::Line, N, r)
    x = @. 0.5 * (1 + r)
    V =  hcat(@. [(factorial(N)/(factorial(i) * factorial(N - i))) * x^i * (1-x)^(1-i) for i in 0:N ]...)
    return V, nothing 
end

function bernstein_basis(::Tri, N, r, s, t)
    V =  hcat(@. [(factorial(N)/(factorial(i) * factorial(j) * factorial(N - i - j))) * r^i * s^j * t^(N - i - j) for j in 0:N for i in 0:N-j]...)
    # V =  hcat(@. [(factorial(N)/(factorial(i) * factorial(j) * factorial(N - i - j))) * r^i * s^j * t^(N - i - j) for i in 0:N for j in 0:N-i]...)
    Vi = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[1] for j in 0:N for i in 0:N-j]...)
    Vj = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[2] for j in 0:N for i in 0:N-j]...)
    Vk = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[3] for j in 0:N for i in 0:N-j]...)
    return V, Vi, Vj, Vk
end

function bernstein_basis(::Tet, N, r, s, t, u)
    V = hcat(@. [(factorial(N)/(factorial(i) * factorial(j) * factorial(k) * factorial(N - i - j - k))) *
        r^i * s^j * t^k * u^(N - i - j - k) for k in 0:N for j in 0:N-k for i in 0:N-k-j]...)
    # V = hcat(@. [(factorial(N)/(factorial(i) * factorial(j) * factorial(k) * factorial(N - i - j - k))) *
    #     r^i * s^j * t^k * u^(N - i - j - k) for j in 0:N for k in 0:N-j for i in 0:N-j-k]...)
    Vi = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[1] for k in 0:N for j in 0:N-k for i in 0:N-k-j]...)
    Vj = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[2] for k in 0:N for j in 0:N-k for i in 0:N-k-j]...)
    Vk = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[3] for k in 0:N for j in 0:N-k for i in 0:N-k-j]...)
    Vl = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[4] for k in 0:N for j in 0:N-k for i in 0:N-k-j]...)
    return V, Vi, Vj, Vk, Vl
end
 
function evaluate_bernstein_derivatives(::Tet, N, r, s, t, u, i, j, k, l)
    if i + j + k + l != N
        throw(DomainError([i,j,k,l],"Barycentric coordinates do not sum to total degree."))
    end
    if i > 0
        vi = @. (factorial(N)/(factorial(i-1) * factorial(j) * factorial(k) * factorial(l))) * r^(i-1) * s^j * t^k * u^l
    else 
        # vector with all zeros
        vi = 0 .* r
    end
    if j > 0
        vj = @. (factorial(N)/(factorial(i) * factorial(j-1) * factorial(k) * factorial(l))) * r^i * s^(j-1) * t^k * u^l
    else 
        # vector with all zeros
        vj = 0 .* s
    end
    if k > 0
        vk = @. (factorial(N)/(factorial(i) * factorial(j) * factorial(k-1) * factorial(l))) * r^i * s^j * t^(k-1) * u^l
    else 
        # vector with all zeros
        vk = 0 .* t
    end
    if l > 0
        vl = @. (factorial(N)/(factorial(i) * factorial(j) * factorial(k) * factorial(l-1))) * r^i * s^j * t^k * u^(l-1)
    else 
        # vector with all zeros
        vl = 0 .* u
    end
    return vi, vj, vk, vl
end

function evaluate_bernstein_derivatives(::Tri, N, r, s, t, i, j, k)
    # Since we are working with a triangle, set u = 0 and l = 0
    u = zeros(length(r))
    l = 0

    vi, vj, vk, _ = evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, l)
    return vi, vj, vk
end

N = 7
rd = RefElemData(Tet(), N)
# LIFT: quad points on faces -> polynomial in volume
# ---> define Vq: polynomials on faces -> quad points on faces

# ∑ u(x_i) * w_i = ∫u(x)
# (A' * W * A)_ij = (a_i, a_j)_L2 = ∫a_i(x) a_j(x)
(; rf, sf, tf, wf) = rd
rf, sf, tf, wf = reshape.((rf, sf, tf, wf), :, 4) 

rd_tri = RefElemData(Tri(), rd.N, quad_rule_vol = (rf[:,4], sf[:,4], wf[:,4]))

# # ---> nodal_LIFT = LIFT * Vq: Lagrange polynomials on faces -> Lagrange polynomials in volume
nodal_LIFT = rd.LIFT * kron(I(4), rd_tri.Vq)

# nodal_to_bernstein_volume * nodal_LIFT * bernstein_to_nodal_face
VB, _ = bernstein_basis(Tri(), N, rd_tri.r, rd_tri.s)
bernstein_to_nodal_face = kron(I(4), VB)
bernstein_to_nodal_volume, _ = bernstein_basis(Tet(), N, rd.rst...)

# directly constructing the Bernstein lift matrix
Vq, _ = bernstein_basis(Tet(), N, rd.rstq...)
MB = Vq' * diagm(rd.wq) * Vq
Vf, _ = bernstein_basis(Tet(), N, rd.rstf...)
VBf1, _ = bernstein_basis(Tri(), N, rf[:,1], tf[:,1])
VBf2, _ = bernstein_basis(Tri(), N, rf[:,2], sf[:,2])
VBf3, _ = bernstein_basis(Tri(), N, sf[:,3], tf[:,3])
VBf4, _ = bernstein_basis(Tri(), N, rf[:,4], sf[:,4])
MBf = Vf' * diagm(rd.wf) * blockdiag(sparse.((VBf1, VBf2, VBf3, VBf4))...)

using SparseArrays
bernstein_LIFT = MB \ MBf # bernstein_to_nodal_volume \ (nodal_LIFT * bernstein_to_nodal_face)
bernstein_LIFT = Matrix(droptol!(sparse(bernstein_LIFT), 10 * length(bernstein_LIFT) * eps()))

bernstein_LIFT
spy(bernstein_LIFT)
# comparing the two to make sure the difference is small
using Test
@test norm(bernstein_LIFT - bernstein_to_nodal_volume \ (nodal_LIFT * bernstein_to_nodal_face)) < 100 * length(bernstein_LIFT) * eps()

@test bernstein_LIFT[:, 109:144] ≈ lift(7)


spy(bernstein_LIFT[:, 109:144])
spy(lift(7))

bernstein_LIFT[:, 109:144] * x
lift(7) * x
fast!(out, ElevationMatrix{7}(), x, L0)
