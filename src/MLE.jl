

function tri_offsets(N)
    tup = [0]
    count = 0
    for i in 1:N
        count += N + 2 - i
        push!(tup, count)
    end
    return tuple(tup...)
end


multiindex_to_linear(i, j, k, offset) = @inbounds (min(i,j,k) < 0) ? 0 : i + offset[j+1] + 1 

# E: N -> N - 1. mutiplies out = E * x
function reduction_multiply!(out, N, x, offset)
    row = 1
    @inbounds for j in 0:N-1
        for i in 0:N-1-j
            k = N-1-i-j
            val = muladd((i+1)/N, x[multiindex_to_linear(i+1, j, k, offset)], 0.0)
            val = muladd((j+1)/N, x[multiindex_to_linear(i, j+1, k, offset)], val)
            val = muladd((k+1)/N, x[multiindex_to_linear(i, j, k+1, offset)], val)
            out[row] = val # not sure why putting / N above is faster?
            row += 1
        end
    end
    return out
end

# N-degree lift matrix * x
function fast!(out, N, L0, x, offset)
    E = L0 * x # note L0 is very very sparse, could speedup later
    index1 = div((N + 1) * (N + 2), 2)
    out[1:index1] = E
    E = reduction_multiply!(E, N, E, offset[N])
    for j in 1:N
        diff = div((N + 1 - j) * (N + 2 - j), 2)
        index2 = index1 + diff
        # assign the next (N+1-j)(N+2-j)/2 entries as l_j * (E^N_{N_j})^T u^f
        out[(index1 + 1): index2] .= ((isodd(j) ? -1.0 : 1.0) * binomial(N, j) / (1.0 + j)) .* E[1:diff]
        index1 = index2
        if j < N
            reduction_multiply!(E, N-j, E, offset[N-j])
        end
        # opportunity for further (small) (optimization?): can derive offset[N-1] from offset[N]
    end
    return out
end

N=7
# all hardcoded for N=7
x = rand(Float64, 36)
out = zeros(Float64, 120)
# random L0
L0 = rand(Float64, 36, 36)

offset_table = [tri_offsets(N) for i in 1:20]

@btime fast!($(out), $(N), $(L0), $(x), $(offset_table))



