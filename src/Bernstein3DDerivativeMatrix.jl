struct BernsteinDerivativeMatrix_3D_r{N} <: AbstractMatrix{Float64} end
struct BernsteinDerivativeMatrix_3D_s{N} <: AbstractMatrix{Float64} end
struct BernsteinDerivativeMatrix_3D_t{N} <: AbstractMatrix{Float64} end

function Base.size(::BernsteinDerivativeMatrix_3D_r{N}) where {N}
    Np = div((N + 1) * (N + 2) * (N + 3), 6)
    return (Np, Np)
end

Base.size(::BernsteinDerivativeMatrix_3D_s{N}) where {N} = size(BernsteinDerivativeMatrix_3D_r{N}())
Base.size(::BernsteinDerivativeMatrix_3D_t{N}) where {N} = size(BernsteinDerivativeMatrix_3D_r{N}())

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

"""
    get_coeff(i1, j1, k1, l1, i2, j2, k2, l2)

Returns the value of the `(i1, j1, k1, l1), (i2, j2, k2, l2)`-th entry of the 3D Bernstein derivative matrix with respect to `i` (first barycentric coordinate).
"""
function get_coeff(i1, j1, k1, l1, i2, j2, k2, l2)
    if (i1, j1, k1, l1) == (i2, j2, k2, l2) return i1
    elseif (i1 + 1, j1 - 1, k1, l1) == (i2, j2, k2, l2) return j1
    elseif (i1 + 1, j1, k1 - 1, l1) == (i2, j2, k2, l2) return k1
    elseif (i1 + 1, j1, k1, l1 - 1) == (i2, j2, k2, l2) return l1
    else return 0 end
end

function linear_to_ijkl_lookup(N)
    return [(i, j, k, N - i - j - k) for k in 0:N for j in 0:N - k for i in 0:N - j - k]
end

function Base.getindex(::BernsteinDerivativeMatrix_3D_r{N}, m, n) where {N}
    linear_to_ijkl = linear_to_ijkl_lookup(N)
    (i1, j1, k1, l1) = linear_to_ijkl[m]
    (i2, j2, k2, l2) = linear_to_ijkl[n]
    # du/dr = 1/2 du/di - 1/2 du/dl
    return 0.5 * (get_coeff(i1, j1, k1, l1, i2, j2, k2, l2) - get_coeff(l1, j1, k1, i1, l2, j2, k2, i2))
end


function Base.getindex(::BernsteinDerivativeMatrix_3D_s{N}, m, n) where {N}
    linear_to_ijkl = linear_to_ijkl_lookup(N)
    (i1, j1, k1, l1) = linear_to_ijkl[m]
    (i2, j2, k2, l2) = linear_to_ijkl[n]
    # du/ds = 1/2 du/dj - 1/2 du/dl
    return 0.5 * (get_coeff(j1, i1, k1, l1, j2, i2, k2, l2) - get_coeff(l1, j1, k1, i1, l2, j2, k2, i2))
end

function Base.getindex(::BernsteinDerivativeMatrix_3D_t{N}, m, n) where {N}
    linear_to_ijkl = linear_to_ijkl_lookup(N)
    (i1, j1, k1, l1) = linear_to_ijkl[m]
    (i2, j2, k2, l2) = linear_to_ijkl[n]
    # du/dt = 1/2 du/dk - 1/2 du/dl
    return 0.5 * (get_coeff(k1, j1, i1, l1, k2, j2, i2, l2) - get_coeff(l1, j1, k1, i1, l2, j2, k2, i2))
end

"""
    fast!(out, ::BernsteinDerivativeMatrix_3D_r{N}, x, offset) where {N}

Computes the derivative with respect to the r coordinate and stores the result in `out`.

# Parameters
- `out`: The output vector to store the result.
- `x`: The input vector.
- `offset`: The precomputed offset values given by `offsets(N)`.
"""
function fast!(out, N, x, tri_offset, tet_offset)
    row = 1
    @inbounds for k in 0:N
        for j in 0:N - k
            for i in 0:N - j - k
                l = N - i - j - k
                val = 0.0
                x_row = x[row]

                x1 = x_row
                x4 = (i > 0) ? x[ijk_to_linear(i - 1, j, k, tri_offset, tet_offset)] : 0.0
                @fastmath val += i * (x1 - x4)

                if j > 0
                    x1 = x[ijk_to_linear(i + 1, j - 1, k, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i, j - 1, k, tri_offset, tet_offset)]
                    @fastmath val += j * (x1 - x4)
                end

                if k > 0
                    x1 = x[ijk_to_linear(i + 1, j, k - 1, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i, j, k - 1, tri_offset, tet_offset)]
                    @fastmath val += k * (x1 - x4)
                end

                x1 = (l > 0) ? x[ijk_to_linear(i + 1, j, k, tri_offset, tet_offset)] : 0.0
                x4 = x_row
                @fastmath val += l * (x1 - x4)

                @fastmath out[row] = 0.5 * val

                row += 1
            end
        end
    end
    return out
end

function fast!(out, ::BernsteinDerivativeMatrix_3D_s{N}, x, tri_offset, tet_offset) where {N}
    row = 1
    for k in 0:N
        for j in 0:N - k
            for i in 0:N - j - k
                l = N - i - j - k
                val = 0.0
                x_row = x[row]

                if i > 0
                    x2 = x[ijk_to_linear(i - 1, j + 1, k, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i - 1, j, k, tri_offset, tet_offset)]
                    @fastmath val += i * (x2 - x4)
                end

                x2 = x_row
                x4 = (j > 0) ? x[ijk_to_linear(i, j - 1, k, tri_offset, tet_offset)] : 0.0
                @fastmath val += j * (x2 - x4)

                if k > 0
                    x2 = x[ijk_to_linear(i, j + 1, k - 1, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i, j, k - 1, tri_offset, tet_offset)]
                    @fastmath val += k * (x2 - x4)
                end

                x2 = (l > 0) ? x[ijk_to_linear(i, j + 1, k, tri_offset, tet_offset)] : 0.0
                x4 = x_row
                @fastmath val += l * (x2 - x4)

                @fastmath out[row] = 0.5 * val

                row += 1
            end
        end
    end
    return out
end

function fast!(out, ::BernsteinDerivativeMatrix_3D_t{N}, x, tri_offset, tet_offset) where {N}
    row = 1
    for k in 0:N
        for j in 0:N - k
            for i in 0:N - j - k
                l = N - i - j - k
                val = 0.0
                x_row = x[row]

                if i > 0
                    x3 = x[ijk_to_linear(i - 1, j, k + 1, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i - 1, j, k, tri_offset, tet_offset)]
                    @fastmath val += i * (x3 - x4)
                end

                if j > 0
                    x3 = x[ijk_to_linear(i, j - 1, k + 1, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i, j - 1, k, tri_offset, tet_offset)]
                    @fastmath val += j * (x3 - x4)
                end

                x3 = x_row
                x4 = (k > 0) ? x[ijk_to_linear(i, j, k - 1, tri_offset, tet_offset)] : 0.0
                @fastmath val += k * (x3 - x4)

                x3 = (l > 0) ? x[ijk_to_linear(i, j, k + 1, tri_offset, tet_offset)] : 0.0
                x4 = x_row
                @fastmath val += l * (x3 - x4)

                @fastmath out[row] = 0.5 * val

                row += 1
            end
        end
    end
    return out
end

"""
test
"""

using Test

N=7
@test BernsteinDerivativeMatrix_3D_r{N}() ≈ 0.5 * (evaluate_bernstein_derivative_matrices(Tet(), N)[1] - evaluate_bernstein_derivative_matrices(Tet(), N)[4])
@test BernsteinDerivativeMatrix_3D_s{N}() ≈ 0.5 * (evaluate_bernstein_derivative_matrices(Tet(), N)[2] - evaluate_bernstein_derivative_matrices(Tet(), N)[4])
@test BernsteinDerivativeMatrix_3D_t{N}() ≈ 0.5 * (evaluate_bernstein_derivative_matrices(Tet(), N)[3] - evaluate_bernstein_derivative_matrices(Tet(), N)[4])

# for n = 7
x = rand(Float64, 120)
out = similar(x)
@test fast!(copy(out), BernsteinDerivativeMatrix_3D_r{N}(), x, tri_offsets(N), tet_offsets(N)) ≈ 0.5 * (evaluate_bernstein_derivative_matrices(Tet(), N)[1] - evaluate_bernstein_derivative_matrices(Tet(), N)[4]) * x
@test fast!(copy(out), BernsteinDerivativeMatrix_3D_s{N}(), x, tri_offsets(N), tet_offsets(N)) ≈ 0.5 * (evaluate_bernstein_derivative_matrices(Tet(), N)[2] - evaluate_bernstein_derivative_matrices(Tet(), N)[4]) * x
@test fast!(copy(out), BernsteinDerivativeMatrix_3D_t{N}(), x, tri_offsets(N), tet_offsets(N)) ≈ 0.5 * (evaluate_bernstein_derivative_matrices(Tet(), N)[3] - evaluate_bernstein_derivative_matrices(Tet(), N)[4]) * x

using BenchmarkTools
@btime fast!($(copy(out)), 7, $(x), $(tri_offsets(7)), $(tet_offsets(7)))
mat = evaluate_bernstein_derivative_matrices(Tet(), 7)[1] - evaluate_bernstein_derivative_matrices(Tet(), 7)[4]
@btime mul!($(copy(out)), $(mat), $(x))
