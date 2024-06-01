using Test

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
test
"""

N = 8
index = 1
for k in 0:N
    for j in 0:N-k
        for i in 0:N-k-j
            # println(ijk_to_linear(i,j,k,tri_offsets(N),tet_offsets(N)))
            @test ijk_to_linear(i,j,k,tri_offsets(N),tet_offsets(N)) == index
            index += 1
        end
    end
end

