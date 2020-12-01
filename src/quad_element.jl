"""
    basis(N,r,s)

Computes orthonormal basis of degree N at coordinates (r,s)

"""
function basis(N,r,s)
    Np = convert(Int,(N+1)*(N+1))
    sk = 1
    V,Vr,Vs = ntuple(x->zeros(length(r), Np),3)
    for i=0:N
        for j=0:N
            V[:,sk]  = jacobiP(r, 0, 0, i).*jacobiP(s, 0, 0, j)
            Vr[:,sk] = grad_jacobiP(r, 0, 0, i).*jacobiP(s, 0, 0, j)
            Vs[:,sk] = jacobiP(r, 0, 0, i).*grad_jacobiP(s, 0, 0, j)
            sk += 1
        end
    end
    return V,Vr,Vs
end

# ===================================================

"""
    nodes(N)

Compute optimized interpolation nodes of degree N

"""

function nodes(N)
    r1D,w1D = gauss_lobatto_quad(0,0,N)
    s,r = meshgrid(r1D)
    return r[:], s[:]
end

"""
    equi_nodes(N)

Compute equispaced nodes of degree N.

"""

function equi_nodes(N)
    r1D = LinRange(-1,1,N+1)
    s,r = meshgrid(r1D)
    return r[:], s[:]
end

"""
    quad_nodes(N)

Compute quadrature nodes and weights of degree N

"""
function quad_nodes(N)
    r1D,w1D = gauss_quad(0,0,N)
    s,r = meshgrid(r1D)
    ws,wr = meshgrid(w1D)
    w = @. wr*ws
    return r[:], s[:], w[:]
end