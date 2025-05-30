### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 79c9f038-8dfa-11eb-1965-3d7328e60dac
using LinearAlgebra

# ╔═╡ 88170e78-8dfa-11eb-2595-4d790546c234
using StaticArrays

# ╔═╡ eb3c1c97-8c3b-4938-b79b-c3302317b3b4
using PlutoUI

# ╔═╡ 8bdcff68-8dfa-11eb-16ed-57f0080561c7
using Test

# ╔═╡ 95e0ccc4-8dfa-11eb-0317-25c20bc851cd
using Profile, ProfileSVG

# ╔═╡ 184abe24-8f75-11eb-3d23-638d60c52204
 using DataFrames

# ╔═╡ 6e8e65ce-0831-49c5-83d6-6400ff672327
using Traceur

# ╔═╡ f65d8a8c-8e3e-11eb-0e9c-878df19fa203
using Plots

# ╔═╡ d9f3e41a-8df9-11eb-0c43-fb7fa62effd2
md"""
# QVecSim - Quantum Vector State Simulation in Julia
Rick Muller

QVecSim is a Julia-based quantum vector state simulator. The code uses a quantum register that is exponential in the number of qubits, and then tries to be as
efficient as possible in applying operators to this state.

The notebook is broken into three sections. 

- The first describes basic data structures and helper functions. 
- The second describes single qubit operators.
- The third describes two qubit operators.

The notebook concludes with some timing and profiling information.
"""

# ╔═╡ 58216f68-b026-4a64-9311-b1c6459937ac
TableOfContents()

# ╔═╡ 3debc96a-8dfa-11eb-188f-ed42ee216fda
md"## Imports/Dependencies
"

# ╔═╡ ffc27bde-8df9-11eb-299e-6d302848198f
md"## QVec Structure and Helper Functions"

# ╔═╡ 12df21ec-8dfa-11eb-26c2-3f970507a0b3
md"""
### QReg(vec) 
Create a qubit register with convenience functions

Because of Julia's indexing, the `i`th component of the vector contains 
the complex coefficient of the wave function for the qubit corresponding
to the bit representation of `i-1`.

Hence, $|-\rangle$ = qm = [1 -1]/sqrt(2). Here, index 1 corresponds to the 
$|0\rangle$ qubit
and has a coefficient of 1, and index 2 corresponds to the $|1\rangle$ qubit 
and has a coefficient of -1.
"""
				

# ╔═╡ 27707a50-8dfa-11eb-15ee-c7b030f04b53
mutable struct QReg <: AbstractVector{Complex{Float64}}
    n::Int
    v::Array{Complex{Float64},1}
    function QReg(vec)
        @assert ispow2(length(vec))
        n = floor(Int,log2(length(vec)))
        new(n,normalize(complex(vec)))
    end
end

# ╔═╡ ad0b1b66-8dfa-11eb-1acb-b7197fa8099b
Base.size(q::QReg) = (q.n,)

# ╔═╡ b7fb273c-8dfa-11eb-306e-95fde10bcc6b
Base.getindex(q::QReg, i::Int) = q.v[i]

# ╔═╡ bafe358c-8dfa-11eb-0a36-0b141a16b7ce
Base.setindex!(q::QReg, val, i) = (q.v[i] = val)

# ╔═╡ ca472cc4-8dfa-11eb-11b7-17bbbbcb951e
begin
	import Base: ==
	==(q::QReg,r::QReg) = (q.v == r.v)
end

# ╔═╡ cff33e10-8dfa-11eb-0bdf-63c3f7bd27cc
begin
	import Base: -
	-(q::QReg) = QReg(-q.v)
end

# ╔═╡ 008971c8-8dfb-11eb-0171-c771c51aa0ce
Base.kron(q::QReg,r::QReg) = QReg(kron(q.v,r.v))

# ╔═╡ 07946484-8dfb-11eb-2fce-f33626c0688b
⊗(q::QReg,r::QReg) = kron(q,r)

# ╔═╡ 108441a4-8dfb-11eb-265f-09374ef2f660
Base.copy(q::QReg) = QReg(copy(q.v))

# ╔═╡ 12108a80-8dfb-11eb-2add-4b7993e4d67d
Base.isapprox(q::QReg,r::QReg) = isapprox(q.v,r.v,atol=1e-15)

# ╔═╡ b3f9ba3a-8dfb-11eb-0d9e-33ed9c659fd4
md"### QReg printing functions"

# ╔═╡ 16000424-8dfb-11eb-384c-6ff88e8b1121
qcoeff(qi) = isreal(qi) ? real(qi) : qi

# ╔═╡ 24857ea0-8dfb-11eb-06a3-1753bbd039e5
qterm(i,qi,n=1) = "$(qcoeff(qi))|$(string(i-1,base=2,pad=n))>"

# ╔═╡ 2ef224b2-8dfb-11eb-364b-cbc7076fa40d
small(x,eps=1e-15) = abs(x) < eps

# ╔═╡ 29644b7e-8dfb-11eb-2f23-419061bbae4a
qterms(q::QReg) = [qterm(i,qi,q.n) for (i,qi) in enumerate(q.v) if !small(qi)]

# ╔═╡ b4da6daa-8e8c-11eb-06a2-d1d7059b52bf
#Base.show(io::IO,::MIME"text/plain",q::QReg) = print(io,to_str(q))

# ╔═╡ f4ddc038-8e8b-11eb-02b1-b553ae4c65cb
function to_str(q::QReg)
	terms = qterms(q)
    totlen = sum(length(term) for term in terms)
    delimiter = totlen > 60 ? "\n" : " "
    return join(terms,delimiter)
end

# ╔═╡ 33eac174-8dfb-11eb-202c-61a0d5a8130a
Base.show(io::IO,q::QReg) = print(io,to_str(q))

# ╔═╡ cb71065a-8e8c-11eb-02d1-6385071af875
Base.show(io::IO,::MIME"text/html",q::QReg) = print(io,to_str(q))

# ╔═╡ a00ec274-8e8f-11eb-0e7e-3b66aff56927
md"With a show(io,::MIME\"text/html\",obj) method defined, Pluto can pretty-print the object."

# ╔═╡ 4811fbe8-8dfb-11eb-30b0-01ee4ca1ea0a
md"### Pauli and other matrices"

# ╔═╡ 60448e10-8dfb-11eb-28fa-670ebf6be9d9
begin
	# Pauli matrices
	σI = SMatrix{2,2,Complex{Float64}}([1 0; 0 1])
	σX = SMatrix{2,2,Complex{Float64}}([0 1; 1 0])
	σY = SMatrix{2,2,Complex{Float64}}([0 -1im;	1im 0])
	σZ = SMatrix{2,2,Complex{Float64}}([1 0; 0 -1])

	# Other matrices
	mH = SMatrix{2,2,Complex{Float64}}([1 1; 1 -1])/sqrt(2)
	mS = SMatrix{2,2,Complex{Float64}}([1 0; 0 1im])
	mCPHASE = SMatrix{4,4,Complex{Float64}}([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1])
	mCNOT = SMatrix{4,4,Complex{Float64}}([1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0])
end

# ╔═╡ 7681d6f6-8dfb-11eb-0288-075fbea55620
md"### Define one and two qubit states"

# ╔═╡ 8b4e0e42-8dfb-11eb-16a5-0788c3178e35
begin
	# Define single qubits
	q0 = QReg([1,0])
	q1 = QReg([0,1])
	qp = QReg([1,1])
	qm = QReg([1,-1])
	# and two qubit states
	q00 = q0⊗q0
	q01 = q0⊗q1
	q10 = q1⊗q0
	q11 = q1⊗q1
end

# ╔═╡ c633a98c-8e04-11eb-0902-97b09cb550b7
md"Using the `ket` function, or the q\"abc\" string literal, you can easily write ket-like terms and wave functions"

# ╔═╡ c0e1c8ac-8e03-11eb-0c44-b79882584e9b
function ket(s::String)
	states = Dict('0'=>q0,'1'=>q1,'+'=>qp,'-'=>qm)
	return reduce(kron,[states[c] for c in s])
end

# ╔═╡ 10012a06-8e04-11eb-28b9-2540ca541e53
macro q_str(p)
	ket(p)
end

# ╔═╡ 35e34d8c-8e8c-11eb-33b8-3163abb3401c
q00⊗q1, q"001"

# ╔═╡ 93bf2ae8-8dfb-11eb-0675-af9f36af810d
md"### qzero(n)
Make a zero state with n qubits"


# ╔═╡ f77d641e-8dfb-11eb-3769-939155abd65f
qzero(n::Int) = QReg(vcat([1],zeros(2^n-1)))

# ╔═╡ fd12d332-8dfb-11eb-1eb6-9d1356811359
md"""### conjugate_index(i,target) 
Find index of other part of target qubit

Given the component indexed at `i`, which has the bits of integer i-1,
compute the other index of the component.

For example, if we are looking at the component |0> (i=1) in the b=1
position, the other qubit index corresponds to |1>, which is index 2.

A more complicated example is the |10> component in the b=2 position
(the "1"). This would be called with index i=3 (00,01,10...) and the
other index would be 1, corresponding to |00>
"""

# ╔═╡ 11420b84-8dfc-11eb-0df4-6d54d4eb238c
conjugate_index(i::Int,b::Int) = (i-1) ⊻ (1 << (b-1))+1

# ╔═╡ 92f42cc0-8dfc-11eb-25b9-83e8ef5337c9
md"### Basic tests"

# ╔═╡ 9c31c8d6-8dfc-11eb-1390-9967dc2cf79e
@testset "Basics" begin
    @test QReg([1,1]) == QReg([2,2]) # Just a normalization difference
    @test QReg([1,1]).v == Complex{Float64}[1.0+0im, 1.0+0im]/sqrt(2)
    @test kron(qp,qp) == QReg([1,1,1,1])
    @test qzero(2) == kron(q0,q0)
    @test qzero(4) == kron(q00,q00)

    # For example, if we are looking at the component |0> (i=1) in
    # the b=1 position, the other qubit index corresponds to |1>,
    # which is index 2.
    @test conjugate_index(1,1) == 2
    # It also follows that the reverse is true, that the other index
    # for 2 is 1 if we're targetting the b bit:
    @test conjugate_index(2,1) == 1

    #A more complicated example is the |10> component in the b=2
    # position (the "1"). This would be called with index i=3 (00,01,10...)
    # and the other index would be 1, corresponding to |00>
    @test conjugate_index(3,2) == 1
    # It also follows that the reverse is true:
    @test conjugate_index(1,2) == 3

    # Test unary minus
    @test -(-qp) ≈ qp
    @test -(-qm) ≈ qm 
	
	# Ket string literals
	@test q"0" == q0
	@test q"00" == q00
	@test q"+" == qp
	@test q"10" == q10
end

# ╔═╡ 8629b5b4-8dfc-11eb-2779-1d9ceec910f9
md"## One Qubit Gates"

# ╔═╡ 1d29af10-8dfc-11eb-3793-819b31c502ad
md"""### oneq!(M,q,target)

Perform an arbitrary one-qubit gate defined by the matrix `M` on the qubit register `q` at the `target` position.

For each operation, one has to find the indices i,j of the entries
into the register, then apply w[i],w[j] = M*[v[i],v[j]].

The only complication is that we need to get the indices in the right
order. M is ordered such that it should operate on the |0> part of the
qubit as the first index, and the |1> part of the qubit in the second
qubit. Therefore, if the `target` bit is set (=1), we need to flip the
indices to put them in the right order.

Static arrays are used if possible to accelerate things.
"""

# ╔═╡ 3bd323b2-8dfc-11eb-1862-fb7d438cfa57
function oneq!(M::SMatrix{2,2,Complex{Float64}},q::QReg,target::Int=1)
    temp = MVector{2,Complex{Float64}}(zeros(2))
    for (i,qi) in enumerate(q.v)
        j = conjugate_index(i,target)
        if i > j continue end
        if small(abs(qi) + abs(q[j])) continue end
        temp[1],temp[2] = qi,q[j]
        q[i],q[j] = M*temp
    end
    return nothing
end

# ╔═╡ 70c3bf30-8dfc-11eb-278e-17b0abeedfc1
md"Simple wrapper for version that doesn't modify the existing state"

# ╔═╡ 4a3e91e6-8dfc-11eb-061e-33133fc724e7
function oneq(M::SMatrix{2,2,Complex{Float64}},q::QReg,target::Int=1)
	q2 = copy(q)
	oneq!(M,q2,target)
	return q2
end

# ╔═╡ 3e2a2ad8-8e91-11eb-1787-0f8777275184
md"The nice thing about having the `oneq` function call `oneq!` is that I don't have to write separate test functions."

# ╔═╡ d1b349d2-8dfc-11eb-087f-8f3ab7ce1a03
md"Standard gates"

# ╔═╡ dc1fdb4c-8dfc-11eb-2542-c3637b2790d6
begin
	X(r::QReg,target=1) = oneq(σX,r,target)
	Y(r::QReg,target=1) = oneq(σY,r,target)
	Z(r::QReg,target=1) = oneq(σZ,r,target)
	H(r::QReg,target=1) = oneq(mH,r,target)
	
	X!(r::QReg,target=1) = oneq!(σX,r,target)
	Y!(r::QReg,target=1) = oneq!(σY,r,target)
	Z!(r::QReg,target=1) = oneq!(σZ,r,target)
	H!(r::QReg,target=1) = oneq!(mH,r,target)
end

# ╔═╡ ecccc89c-8dfc-11eb-1247-cdb8b0b6dfc8
md"""### PauliRotation(θx,θy,θz,prefactor)

Generate the rotation matrix corresponding to an arbitrary rotation about the angles θx,θy,θz

See formula (2) in the [wikipedia page on Pauli matrices](http://en.wikipedia.org/wiki/Pauli_matrices#Commutation_relations)

If you input prefactor=-1j, you match the standard Pauli matrix
definitions for theta=π. But overall phase factors don't really
matter.
"""

# ╔═╡ 16045e66-8dfd-11eb-21b6-d5f028493271
function PauliRotation(θx,θy,θz,prefactor=-1im)
    v = [θx,θy,θz]
    θ = sqrt(v⋅v)
    if !small(θ)
        n = v/θ
    else
        θ = 0
        n = [0,0,0] # arbitrary
    end
    prefactor*(σI*cos(θ/2) + 1im*sin(θ/2)*(n[1]*σX+n[2]*σY+n[3]*σZ))
end

# ╔═╡ 8a041a84-8dfd-11eb-2eed-af556a07bee9
md"### One Qubit Tests"

# ╔═╡ 2eb51d4a-8dfd-11eb-0258-fdf4a0d1fc4e
@testset "One Qubit Tests" begin
	@test oneq(σX,q0) == q1
	@test oneq(σX,q1) == q0
	@test oneq(σX,qp) ≈ qp
	@test oneq(σX,qm) ≈ -qm
	@test oneq(mH,q0) ≈ qp
	@test oneq(mH,q1) ≈ qm
	@test oneq(mH,qp) ≈ q0 
	@test oneq(mH,qm) ≈ q1
	
	@test X(q0) == q1
	@test X(q1) == q0
	@test X(q00,2) == q10
	@test X(q01,2) == q11
	@test X(q10,2) == q00
	@test X(q11,2) == q01
	
	@test Y(q0) == 1im*q1
	@test Y(q1) == -1im*q0
	
	@test Z(q0) == q0
	@test Z(q1) == -q1
	@test H(q0) ≈ qp
	@test H(q1) ≈ qm
	
	@test PauliRotation(π,0,0) ≈ σX
	@test PauliRotation(0,π,0) ≈ σY
	@test PauliRotation(0,0,π) ≈ σZ
	
end

# ╔═╡ 3007c44e-8dfe-11eb-3e0b-218d1afa03ee
md"## Two Qubit Operators"

# ╔═╡ 41e91316-8dfe-11eb-24f1-63f40d5c6885
md"""### twoq!(M,q,control,target)
Perform an arbitrary two-qubit operation
defined by the matrix `M` on the qubit register `q` with `control` at 
the `target` position.

For each nonzero term in the qubit register, we have to find the indices 
i,j,k,l of the entries, and put them in the right permutation.

Static arrays are used if possible to accelerate things.
"""

# ╔═╡ 6664b14e-8dfe-11eb-3767-636a7584b466
function twoq!(M::SMatrix{4,4,Complex{Float64}},q::QReg,control::Int,target::Int)
    temp = MVector{4,Complex{Float64}}(zeros(4))
    for (i,qi) in enumerate(q.v)
        j = conjugate_index(i,target)
        if i > j continue end
        k = conjugate_index(i,control)
        if i > k continue end
        l = conjugate_index(j,control)

        if small(abs(qi) + abs(q[j]) + abs(q[k]) + abs(q[l])) continue end

        temp[1],temp[2],temp[3],temp[4] = qi,q[j],q[k],q[l]
        q[i],q[j],q[k],q[l] = M*temp
    end
    return nothing
end

# ╔═╡ 7566ba4a-8dfe-11eb-24cf-bb7202a23a5f
md"Simple wrapper for version that doesn't modify the existing state"

# ╔═╡ 7a8bf486-8dfe-11eb-1f8a-7f86a36a5ee8
function twoq(M::SMatrix{4,4,Complex{Float64}},r::QReg,control::Int,target::Int)
	q = copy(r)
	twoq!(M,q,control,target)
	return q
end

# ╔═╡ cd7f4670-8dfe-11eb-0710-a9012ac5ad5d
md"Common operator definitions"

# ╔═╡ cb6d86bc-8dfe-11eb-21a4-918f8ab217cb
begin
	CNOT(q,control,target) = twoq(mCNOT,q,control,target)
	CPHASE(q,control,target) = twoq(mCPHASE,q,control,target)
	
	CNOT!(q,control,target) = twoq!(mCNOT,q,control,target)
	CPHASE!(q,control,target) = twoq!(mCPHASE,q,control,target)
end

# ╔═╡ 3cdd7faa-8dff-11eb-26a4-596f5294ded3
md"### Two Qubit Operator Tests"

# ╔═╡ 468aaa46-8dff-11eb-1994-fb6dba71fbac
@testset "Two Qubit Tests" begin
    @test twoq(mCNOT,q00,1,2) == q00
    @test twoq(mCNOT,q01,1,2) == q11
    @test twoq(mCNOT,q10,1,2) == q10
    @test twoq(mCNOT,q11,1,2) == q01

    @test twoq(mCPHASE,q00,2,1) == q00
    @test twoq(mCPHASE,q01,2,1) == q01
    @test twoq(mCPHASE,q10,2,1) == q10
    @test twoq(mCPHASE,q11,2,1) == -q11

    @test CNOT(q00,1,2) == q00
    @test CNOT(q01,1,2) == q11
    @test CNOT(q10,1,2) == q10
    @test CNOT(q11,1,2) == q01

    @test CPHASE(q00,2,1) == q00
    @test CPHASE(q01,2,1) == q01
    @test CPHASE(q10,2,1) == q10
    @test CPHASE(q11,2,1) == -q11
end

# ╔═╡ e24173b4-8dfe-11eb-24ca-3be7a371a220
function H_all(q::QReg)
    nq = copy(q)
    for i in 1:q.n
        nq = H(nq,i)
    end
    return nq
end

# ╔═╡ e9fca5e0-8dfe-11eb-2665-414b59ab6441
function H_all!(q::QReg)
    for i in 1:q.n
        H!(q,i)
    end
    return nothing
end

# ╔═╡ f1f516ec-8dfe-11eb-0973-0d167cbc1dd6
function entangle_all(n::Int)
    q = qzero(n)
    H_all!(q)
    for i in 1:n
        for j in 1:(i-1)
            CNOT!(q,i,j)
        end
    end
    return q
end

# ╔═╡ 4f554b54-8e01-11eb-278b-6dc606417d27
md"## Profiling Code"

# ╔═╡ c6e99580-8dff-11eb-0c0d-9bdb7c51f8e5
@profview entangle_all(16)

# ╔═╡ 58ddee0e-8e01-11eb-37fb-a765ffbce6a4
md"## Timing Code"

# ╔═╡ 01994f78-8dff-11eb-2297-630ddbcd7252
function timing(rng=8:4:20)
	ts = []
    for i in rng
        t = @elapsed entangle_all(i)
		push!(ts,t)
    end
	return DataFrame(n=rng,t=ts)
end

# ╔═╡ 0b9f0b70-8dff-11eb-3696-1bc736d70ee6
timing()

# ╔═╡ 3ad1dd18-8e05-11eb-1568-579ebfa8cd26
md"""
Archiving timing results here:

3/26/21:

	0.000107 seconds (42 allocations: 16.438 KiB)
	0.003727 seconds (88 allocations: 200.781 KiB)
	0.083528 seconds (146 allocations: 3.015 MiB)
	2.551941 seconds (220 allocations: 48.023 MiB, 0.75% gc time)
"""

# ╔═╡ 97964a12-b136-4580-9f42-1227fdb1723e
md"## Traceur"

# ╔═╡ 4fc570ba-d3a0-4255-b586-3eeef04dd763
@trace twoq(mCNOT,q00,1,2)

# ╔═╡ 227378e4-8e3e-11eb-3473-1545aab66768
md"Here's how to plot the elapsed time. Turns out it isn't as useful as the 
table view."

# ╔═╡ eeabfbfc-8e3c-11eb-03ef-25b92ff892a8
function plotelapsed(rng=8:4:20)
	ts = []
    for i in rng
        push!(ts,@elapsed entangle_all(i))
    end
	scatter(rng,ts,label="elapsed")
	plot!(rng,ts,label=nothing)
end

# ╔═╡ Cell order:
# ╟─d9f3e41a-8df9-11eb-0c43-fb7fa62effd2
# ╠═58216f68-b026-4a64-9311-b1c6459937ac
# ╟─3debc96a-8dfa-11eb-188f-ed42ee216fda
# ╠═79c9f038-8dfa-11eb-1965-3d7328e60dac
# ╠═88170e78-8dfa-11eb-2595-4d790546c234
# ╠═eb3c1c97-8c3b-4938-b79b-c3302317b3b4
# ╠═8bdcff68-8dfa-11eb-16ed-57f0080561c7
# ╟─ffc27bde-8df9-11eb-299e-6d302848198f
# ╟─12df21ec-8dfa-11eb-26c2-3f970507a0b3
# ╠═27707a50-8dfa-11eb-15ee-c7b030f04b53
# ╠═ad0b1b66-8dfa-11eb-1acb-b7197fa8099b
# ╠═b7fb273c-8dfa-11eb-306e-95fde10bcc6b
# ╠═bafe358c-8dfa-11eb-0a36-0b141a16b7ce
# ╠═ca472cc4-8dfa-11eb-11b7-17bbbbcb951e
# ╠═cff33e10-8dfa-11eb-0bdf-63c3f7bd27cc
# ╠═008971c8-8dfb-11eb-0171-c771c51aa0ce
# ╠═07946484-8dfb-11eb-2fce-f33626c0688b
# ╠═108441a4-8dfb-11eb-265f-09374ef2f660
# ╠═12108a80-8dfb-11eb-2add-4b7993e4d67d
# ╟─b3f9ba3a-8dfb-11eb-0d9e-33ed9c659fd4
# ╠═16000424-8dfb-11eb-384c-6ff88e8b1121
# ╠═24857ea0-8dfb-11eb-06a3-1753bbd039e5
# ╠═29644b7e-8dfb-11eb-2f23-419061bbae4a
# ╠═2ef224b2-8dfb-11eb-364b-cbc7076fa40d
# ╠═33eac174-8dfb-11eb-202c-61a0d5a8130a
# ╠═b4da6daa-8e8c-11eb-06a2-d1d7059b52bf
# ╠═cb71065a-8e8c-11eb-02d1-6385071af875
# ╠═f4ddc038-8e8b-11eb-02b1-b553ae4c65cb
# ╟─a00ec274-8e8f-11eb-0e7e-3b66aff56927
# ╠═35e34d8c-8e8c-11eb-33b8-3163abb3401c
# ╟─4811fbe8-8dfb-11eb-30b0-01ee4ca1ea0a
# ╠═60448e10-8dfb-11eb-28fa-670ebf6be9d9
# ╟─7681d6f6-8dfb-11eb-0288-075fbea55620
# ╠═8b4e0e42-8dfb-11eb-16a5-0788c3178e35
# ╟─c633a98c-8e04-11eb-0902-97b09cb550b7
# ╠═c0e1c8ac-8e03-11eb-0c44-b79882584e9b
# ╠═10012a06-8e04-11eb-28b9-2540ca541e53
# ╟─93bf2ae8-8dfb-11eb-0675-af9f36af810d
# ╠═f77d641e-8dfb-11eb-3769-939155abd65f
# ╟─fd12d332-8dfb-11eb-1eb6-9d1356811359
# ╠═11420b84-8dfc-11eb-0df4-6d54d4eb238c
# ╟─92f42cc0-8dfc-11eb-25b9-83e8ef5337c9
# ╠═9c31c8d6-8dfc-11eb-1390-9967dc2cf79e
# ╟─8629b5b4-8dfc-11eb-2779-1d9ceec910f9
# ╟─1d29af10-8dfc-11eb-3793-819b31c502ad
# ╠═3bd323b2-8dfc-11eb-1862-fb7d438cfa57
# ╟─70c3bf30-8dfc-11eb-278e-17b0abeedfc1
# ╠═4a3e91e6-8dfc-11eb-061e-33133fc724e7
# ╟─3e2a2ad8-8e91-11eb-1787-0f8777275184
# ╟─d1b349d2-8dfc-11eb-087f-8f3ab7ce1a03
# ╠═dc1fdb4c-8dfc-11eb-2542-c3637b2790d6
# ╟─ecccc89c-8dfc-11eb-1247-cdb8b0b6dfc8
# ╠═16045e66-8dfd-11eb-21b6-d5f028493271
# ╠═8a041a84-8dfd-11eb-2eed-af556a07bee9
# ╠═2eb51d4a-8dfd-11eb-0258-fdf4a0d1fc4e
# ╟─3007c44e-8dfe-11eb-3e0b-218d1afa03ee
# ╟─41e91316-8dfe-11eb-24f1-63f40d5c6885
# ╠═6664b14e-8dfe-11eb-3767-636a7584b466
# ╟─7566ba4a-8dfe-11eb-24cf-bb7202a23a5f
# ╠═7a8bf486-8dfe-11eb-1f8a-7f86a36a5ee8
# ╟─cd7f4670-8dfe-11eb-0710-a9012ac5ad5d
# ╠═cb6d86bc-8dfe-11eb-21a4-918f8ab217cb
# ╟─3cdd7faa-8dff-11eb-26a4-596f5294ded3
# ╠═468aaa46-8dff-11eb-1994-fb6dba71fbac
# ╠═e24173b4-8dfe-11eb-24ca-3be7a371a220
# ╠═e9fca5e0-8dfe-11eb-2665-414b59ab6441
# ╠═f1f516ec-8dfe-11eb-0973-0d167cbc1dd6
# ╟─4f554b54-8e01-11eb-278b-6dc606417d27
# ╠═95e0ccc4-8dfa-11eb-0317-25c20bc851cd
# ╠═c6e99580-8dff-11eb-0c0d-9bdb7c51f8e5
# ╟─58ddee0e-8e01-11eb-37fb-a765ffbce6a4
# ╠═184abe24-8f75-11eb-3d23-638d60c52204
# ╠═01994f78-8dff-11eb-2297-630ddbcd7252
# ╠═0b9f0b70-8dff-11eb-3696-1bc736d70ee6
# ╟─3ad1dd18-8e05-11eb-1568-579ebfa8cd26
# ╟─97964a12-b136-4580-9f42-1227fdb1723e
# ╠═6e8e65ce-0831-49c5-83d6-6400ff672327
# ╠═4fc570ba-d3a0-4255-b586-3eeef04dd763
# ╟─227378e4-8e3e-11eb-3473-1545aab66768
# ╠═f65d8a8c-8e3e-11eb-0e9c-878df19fa203
# ╠═eeabfbfc-8e3c-11eb-03ef-25b92ff892a8
