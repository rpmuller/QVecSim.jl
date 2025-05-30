using QVecSim
using BenchmarkTools

"""
Archiving timing results here:

5/17/21:
105.751 μs (42 allocations: 16.44 KiB)
3.317 ms (88 allocations: 200.78 KiB)
90.436 ms (146 allocations: 3.01 MiB)
2.329 s (220 allocations: 48.02 MiB)
"""

function timing(rng=8:4:20)
    for i in rng
        @btime entangle_all($i)
    end
	return nothing
end

timing()
