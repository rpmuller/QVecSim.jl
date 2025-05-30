module QVecSim

export @q_str, oneq, oneq!, twoq, twoq!, 
    q0, q1, qp, qm, q00, q01, q10, q11, qzero, kron, X, Y, Z, H, CNOT, CPHASE,
    σI, σX, σY, σZ, mH, mCPHASE, mCNOT, entangle_all

using LinearAlgebra
using StaticArrays
import Base: ==,-

"""
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

"""
QReg(vec) - Create a qubit register with convenience functions

Because of Julia's indexing, the `i`th component of the vector contains 
the complex coefficient of the wave function for the qubit corresponding
to the bit representation of `i-1`.

Hence, |- > = qm = [1 -1]/sqrt(2). Here, index 1 corresponds to the 
|0> qubit
and has a coefficient of 1, and index 2 corresponds to the |1> qubit 
and has a coefficient of -1.
"""
mutable struct QReg <: AbstractVector{Complex{Float64}}
    n::Int
    v::Array{Complex{Float64},1}
    function QReg(vec)
        @assert ispow2(length(vec))
        n = floor(Int,log2(length(vec)))
        new(n,normalize(complex(vec)))
    end
end
Base.size(q::QReg) = (q.n,)
Base.getindex(q::QReg, i::Int) = q.v[i]
Base.setindex!(q::QReg, val, i) = (q.v[i] = val)
==(q::QReg,r::QReg) = (q.v == r.v)
-(q::QReg) = QReg(-q.v)
Base.kron(q::QReg,r::QReg) = QReg(kron(q.v,r.v))
⊗(q::QReg,r::QReg) = kron(q,r)
Base.copy(q::QReg) = QReg(copy(q.v))
Base.isapprox(q::QReg,r::QReg) = isapprox(q.v,r.v,atol=1e-15)

qcoeff(qi) = isreal(qi) ? real(qi) : qi
qterm(i,qi,n=1) = "$(qcoeff(qi))|$(string(i-1,base=2,pad=n))>"
small(x,eps=1e-15) = abs(x) < eps
qterms(q::QReg) = [qterm(i,qi,q.n) for (i,qi) in enumerate(q.v) if !small(qi)]

function to_str(q::QReg)
	terms = qterms(q)
    totlen = sum(length(term) for term in terms)
    delimiter = totlen > 60 ? "\n" : " "
    return join(terms,delimiter)
end

Base.show(io::IO,q::QReg) = print(io,to_str(q))

Base.show(io::IO,::MIME"text/html",q::QReg) = print(io,to_str(q))

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

function ket(s::String)
	states = Dict('0'=>q0,'1'=>q1,'+'=>qp,'-'=>qm)
	return reduce(kron,[states[c] for c in s])
end

macro q_str(p)
	ket(p)
end

qzero(n::Int) = QReg(vcat([1],zeros(2^n-1)))

"""conjugate_index(i,target) - Find index of other part of target qubit

Given the component indexed at `i`, which has the bits of integer i-1,
compute the other index of the component.

For example, if we are looking at the component |0> (i=1) in the b=1
position, the other qubit index corresponds to |1>, which is index 2.

A more complicated example is the |10> component in the b=2 position
(the "1"). This would be called with index i=3 (00,01,10...) and the
other index would be 1, corresponding to |00>
"""
conjugate_index(i::Int,b::Int) = (i-1) ⊻ (1 << (b-1))+1

""" oneq!(M,q,target) - Perform an arbitrary one-qubit gate defined by the matrix `M` on the qubit register `q` at the `target` position.

For each operation, one has to find the indices i,j of the entries
into the register, then apply w[i],w[j] = M*[v[i],v[j]].

The only complication is that we need to get the indices in the right
order. M is ordered such that it should operate on the |0> part of the
qubit as the first index, and the |1> part of the qubit in the second
qubit. Therefore, if the `target` bit is set (=1), we need to flip the
indices to put them in the right order.

Static arrays are used if possible to accelerate things.
"""
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

"Wrapper for oneq! that doesn't modify the existing state"
function oneq(M::SMatrix{2,2,Complex{Float64}},q::QReg,target::Int=1)
	q2 = copy(q)
	oneq!(M,q2,target)
	return q2
end

X(r::QReg,target=1) = oneq(σX,r,target)
Y(r::QReg,target=1) = oneq(σY,r,target)
Z(r::QReg,target=1) = oneq(σZ,r,target)
H(r::QReg,target=1) = oneq(mH,r,target)

X!(r::QReg,target=1) = oneq!(σX,r,target)
Y!(r::QReg,target=1) = oneq!(σY,r,target)
Z!(r::QReg,target=1) = oneq!(σZ,r,target)
H!(r::QReg,target=1) = oneq!(mH,r,target)

"""PauliRotation(θx,θy,θz,prefactor) - Generate the rotation matrix corresponding to an arbitrary rotation about the angles θx,θy,θz

See formula (2) in the [wikipedia page on Pauli matrices](http://en.wikipedia.org/wiki/Pauli_matrices#Commutation_relations)

If you input prefactor=-1j, you match the standard Pauli matrix
definitions for theta=π. But overall phase factors don't really
matter.
"""
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

"""twoq!(M,q,control,target) - Perform an arbitrary two-qubit operation
defined by the matrix `M` on the qubit register `q` with `control` at 
the `target` position.

For each nonzero term in the qubit register, we have to find the indices 
i,j,k,l of the entries, and put them in the right permutation.

Static arrays are used if possible to accelerate things.
"""
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

"Wrapper for twoq! that doesn't modify the existing state"
function twoq(M::SMatrix{4,4,Complex{Float64}},r::QReg,control::Int,target::Int)
	q = copy(r)
	twoq!(M,q,control,target)
	return q
end

CNOT(q,control,target) = twoq(mCNOT,q,control,target)
CPHASE(q,control,target) = twoq(mCPHASE,q,control,target)

CNOT!(q,control,target) = twoq!(mCNOT,q,control,target)
CPHASE!(q,control,target) = twoq!(mCPHASE,q,control,target)

function H_all(q::QReg)
    nq = copy(q)
    for i in 1:q.n
        nq = H(nq,i)
    end
    return nq
end

function H_all!(q::QReg)
    for i in 1:q.n
        H!(q,i)
    end
    return nothing
end

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

end # module
