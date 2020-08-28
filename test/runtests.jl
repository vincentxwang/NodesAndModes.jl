using NodesAndModes
using Test
using LinearAlgebra

@testset "1D tests" begin
    tol = 1e2*eps()

    N = 0
    r,w = gauss_lobatto_quad(0,0,N)
    @test sum(w) ≈ 2

    for N = 1:2
        r,w = gauss_quad(0,0,N)
        @test sum(w) ≈ 2
        @test abs(sum(w.*r)) < tol

        V = vandermonde_1D(N,r)
        @test V'*diagm(w)*V ≈ I

        Vr = grad_vandermonde_1D(N,r)
        Dr = Vr/V
        @test norm(sum(Dr,dims=2)) < tol
        @test Dr*r ≈ ones(N+1)

        r,w = gauss_lobatto_quad(0,0,N)
        @test sum(w) ≈ 2
        @test abs(sum(w.*r)) < tol

        # check if Lobatto is exact for 2N-2 polynoms
        V = vandermonde_1D(N-1,r)
        @test V'*diagm(w)*V ≈ I

        Dr = grad_vandermonde_1D(N,r)/vandermonde_1D(N,r)
        @test norm(sum(Dr,dims=2)) < tol
        @test Dr*r ≈ ones(N+1)
    end
end

@testset "2D tri tests" begin
    using NodesAndModes.Tri

    tol = 1e2*eps()

    N = 3
    rq,sq,wq = quad_nodes_2D(2*N)
    @test sum(wq)≈2
    @test sum(rq.*wq)≈ -2/3
    @test sum(sq.*wq)≈ -2/3

    Vq = vandermonde_2D(N,rq,sq)
    @test Vq'*diagm(wq)*Vq ≈ I

    r,s = nodes_2D(N)
    V = vandermonde_2D(N,r,s)
    Dr,Ds = (A->A/V).(grad_vandermonde_2D(N,r,s))
    @test norm(sum(Dr,dims=2)) + norm(sum(Ds,dims=2)) < tol
    @test norm(Dr*s)+norm(Ds*r) < tol
    @test Dr*r ≈ ones(length(r))
    @test Ds*s ≈ ones(length(s))

    r,s = equi_nodes_2D(N)
    V = vandermonde_2D(N,r,s)
    Dr,Ds = (A->A/V).(grad_vandermonde_2D(N,r,s))
    @test norm(sum(Dr,dims=2)) + norm(sum(Ds,dims=2)) < tol
    @test norm(Dr*s)+norm(Ds*r) < tol
    @test Dr*r ≈ ones(length(r))
    @test Ds*s ≈ ones(length(s))
    # export vandermonde_2D, grad_vandermonde_2D
    # export nodes_2D, equi_nodes_2D, quad_nodes_2D
end
