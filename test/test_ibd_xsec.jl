using Test
using Newtrinos

@testset "IBD cross section (Vogel-Beacom O1)" begin
    ibd = Newtrinos.ibd_xsec.configure(Newtrinos.ibd_xsec.VogelBeacomO1())
    # reference values from the official Daya Bay model (dgm_reactor_neutrino: _enu, _ibdxsecO1 [cm²], _jacobian_dEnu_dEe)
    # for (Edep [MeV], cosθ) -> (Eν [MeV], dσ/dcosθ [1e-41 cm²], dEν/dEdep)
    ref = [
        ((2.0, -0.5), (2.7895916490416215, 0.010359253793212016, 1.0069041460748613)),
        ((3.5, 0.3), (4.292718566900551, 0.04053222484647345, 1.0054383082281273)),
        ((6.0, 0.9), (6.787211406271784, 0.13052306923268958, 1.0013036463950957)),
        ((1.2, 0.0), (1.9845429265445393, 0.0015247784330672716, 1.0028515275214547)),
    ]
    for ((Edep, c), (Enu, σ, J)) in ref
        E = ibd.Enu(Edep, c)
        @test E ≈ Enu rtol = 1e-12
        @test ibd.xsec(E, c) ≈ σ rtol = 1e-8
        @test ibd.jacobian(E, Edep, c) ≈ J rtol = 1e-12
    end
    # below threshold
    @test ibd.xsec(1.7, 0.0) == 0
    # the Jacobian is smooth (no interpolation ripples): at cosθ = 0 it is (M_p + Eν)/(M_p - Ee), rising slowly from 1
    Ed = 1.5:0.01:9.0
    J = [ibd.jacobian(ibd.Enu(E, 0.0), E, 0.0) for E in Ed]
    @test all(1.0 .< J .< 1.03)
    @test all(diff(J) .> 0)
end
