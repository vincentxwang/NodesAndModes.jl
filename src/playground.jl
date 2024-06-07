using BenchmarkTools

struct MyStruct{N} end

function pass_as_struct!(out, ::MyStruct{N}, x) where N
    offset = (0, 8, 15, 21, 26, 30, 33, 35)
    row = 1
    @inbounds for j in 0:N-1
        for i in 0:N-1-j
            k = N-1-i-j
            val = muladd((i+1), x[i + 2 + offset[j+1]], 0)
            val = muladd((j+1), x[i + 1 + offset[j+1]], val)
            val = muladd((k+1), x[i + 1 + offset[j+1]], val) # k + 1
            out[row] = val/N
            row += 1
        end
    end
    return out 
end

function pass_as_int!(out, N, x)
    offset = (0, 8, 15, 21, 26, 30, 33, 35)
    row = 1
    @inbounds for j in 0:N-1
        for i in 0:N-1-j
            k = N-1-i-j
            val = muladd((i+1), x[i + 2 + offset[j+1]], 0)
            val = muladd((j+1), x[i + 1 + offset[j+1]], val)
            val = muladd((k+1), x[i + 1 + offset[j+1]], val) # k + 1
            out[row] = val/N
            row += 1
        end
    end
    return out 
end

x = rand(Float64, 36)

@btime pass_as_struct!($(zeros(28)), MyStruct{7}(), $(x))
@btime pass_as_int!($(zeros(28)), 7, $(x))

function loop_with_struct!(x)
    E = pass_as_struct!(zeros(28), MyStruct{7}(), x)
    for j in 1:6
        pass_as_struct!(E, MyStruct{7 - j}(), E)
    end
end

function loop_with_int!(x)
    E = pass_as_int!(zeros(28), 7, x)
    for j in 1:6
        pass_as_int!(E, 7-j, E)
    end
end

@btime loop_with_struct!($(x))
@btime loop_with_int!($(x))
