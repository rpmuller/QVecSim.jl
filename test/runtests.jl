using QVecSim
using Test

@testset verbose=true "QVecSim" begin
    @testset "Basics" begin
        @test QVecSim.QReg([1,1]) == QVecSim.QReg([2,2]) # Just a normalization difference
        @test QVecSim.QReg([1,1]).v == Complex{Float64}[1.0+0im, 1.0+0im]/sqrt(2)
        @test kron(qp,qp) == QVecSim.QReg([1,1,1,1])
        @test qzero(2) == kron(q0,q0)
        @test qzero(4) == kron(q00,q00)

        # For example, if we are looking at the component |0> (i=1) in
        # the b=1 position, the other qubit index corresponds to |1>,
        # which is index 2.
        @test QVecSim.conjugate_index(1,1) == 2
        # It also follows that the reverse is true, that the other index
        # for 2 is 1 if we're targetting the b bit:
        @test QVecSim.conjugate_index(2,1) == 1

        #A more complicated example is the |10> component in the b=2
        # position (the "1"). This would be called with index i=3 (00,01,10...)
        # and the other index would be 1, corresponding to |00>
        @test QVecSim.conjugate_index(3,2) == 1
        # It also follows that the reverse is true:
        @test QVecSim.conjugate_index(1,2) == 3

        # Test unary minus
        @test -(-qp) ≈ qp
        @test -(-qm) ≈ qm 
        
        # Ket string literals
        @test q"0" == q0
        @test q"00" == q00
        @test q"+" == qp
        @test q"10" == q10
    end
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
        
        @test QVecSim.PauliRotation(π,0,0) ≈ σX
        @test QVecSim.PauliRotation(0,π,0) ≈ σY
        @test QVecSim.PauliRotation(0,0,π) ≈ σZ
        
    end
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
end
