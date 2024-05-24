using NodesAndModes

"""
    bernstein_basis(::Union{Tri, Tet}, N, r, s, t)

Returns a Np x Np matrix V, where V_mn = the n-th Bernstein basis evaluated at the m-th point, followed by its derivative matrices.

Use `::Tri` for 2D and `::Tet` for 3D. We order the Bernstein basis according to exponents in dictonary order. Does not work for N > 20
because of `factorial()` limitations.

# Arguments
- `N::Int`: Bernstein basis degree
- `r,s,t::AbstractArray{T,1}`: Np-sized vectors of distinct 2D barycentric coordinates to be used as interpolatory points
"""
function bernstein_basis(::Tri, N, r, s, t)
    V =  hcat(@. [(factorial(N)/(factorial(i) * factorial(j) * factorial(N - i - j))) * r^i * s^j * t^(N - i - j) for i in 0:N for j in 0:N-i]...)
    Vi = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[1] for i in 0:N for j in 0:N-i]...)
    Vj = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[2] for i in 0:N for j in 0:N-i]...)
    Vk = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[3] for i in 0:N for j in 0:N-i]...)
    return V, Vi, Vj, Vk
end

function bernstein_basis(::Tet, N, r, s, t, u)
    V = hcat(@. [(factorial(N)/(factorial(i) * factorial(j) * factorial(k) * factorial(N - i - j - k))) *
        r^i * s^j * t^k * u^(N - i - j - k) for i in 0:N for j in 0:N-i for k in 0:N-i-j]...)
    Vi = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[1] for i in 0:N for j in 0:N-i for k in 0:N-i-j]...)
    Vj = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[2] for i in 0:N for j in 0:N-i for k in 0:N-i-j]...)
    Vk = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[3] for i in 0:N for j in 0:N-i for k in 0:N-i-j]...)
    Vl = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[4] for i in 0:N for j in 0:N-i for k in 0:N-i-j]...)
    return V, Vi, Vj, Vk, Vl
end
 
"""
    evaluate_bernstein_derivatives(::Tri, N, r, s, t, i, j, k)
    evaluate_bernstein_derivatives(::Tet, N, r, s, t, u, i, j, k, l)

Evaluates the derivatives of the (i,j,k)-th 2D Bernstein basis function at points defined by r, s, t. Throws error if ``i + j + k != N ``. Analogous for 3D.

Outputs a vector for each direction.
"""
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

"""
    cartesian_to_barycentric(::Union{Tri, Tet}, coords)

Converts a matrix of cartesian coordinates into a matrix of barycentric coordinates. 3 x num_points
"""
function cartesian_to_barycentric(::Tri, coords::AbstractMatrix)
    hcat([[(col[2] + 1) / 2, - (col[1] + col[2])/2, (col[1] + 1) / 2] for col in eachcol(coords)]...)
end

function cartesian_to_barycentric(::Tet, coords)
    hcat([[- (1 + col[1] + col[2] + col[3]) / 2, (1 + col[1])/2, (1 + col[2]) / 2, (1 + col[3]) / 2] for col in eachcol(coords)]...)
end

function evaluate_bernstein_derivative_matrices(::Tri, N)
    r, s = nodes(Tri(), N)
    coords = transpose(hcat(r,s))
    bary_coords = cartesian_to_barycentric(Tri(), coords)
    V, Vi, Vj, Vk = bernstein_basis(Tri(), N, bary_coords[1,:], bary_coords[2,:], bary_coords[3,:])
    return round.(Int, V \ Vi), round.(Int, V \ Vj), round.(Int, V \ Vk)
end

function evaluate_bernstein_derivative_matrices(::Tet, N)
    r, s, t = nodes(Tet(), N)
    coords = transpose(hcat(r,s,t))
    bary_coords = cartesian_to_barycentric(Tet(), coords)
    V, Vi, Vj, Vk, Vl = bernstein_basis(Tet(), N, bary_coords[1,:], bary_coords[2,:], bary_coords[3,:], bary_coords[4,:])
    return round.(Int, V \ Vi), round.(Int, V \ Vj), round.(Int, V \ Vk), round.(Int, V \ Vl)
end

evaluate_bernstein_derivative_matrices(Tri(), 7)[1]
evaluate_bernstein_derivative_matrices(Tet(), 7)[1]


