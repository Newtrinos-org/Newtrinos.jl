using Test, Newtrinos, Dates, DataFrames

@testset "Reactor flux" begin
    R = Newtrinos.reactor_flux
    rf = R.configure()
    @testset "Huber–Mueller × IBD yields" begin
        # Huber (PRC 84, 024617) table values of the ²³⁵U and ²³⁹Pu spectra [ν̄e / fission / MeV]
        @test rf.isotope_spectrum(:U235, 4.0) ≈ 0.294 rtol = 0.01
        @test rf.isotope_spectrum(:U235, 6.0) ≈ 0.0389 rtol = 0.03
        @test rf.isotope_spectrum(:Pu239, 4.0) ≈ 0.195 rtol = 0.02
        @test rf.ibd_xsec(1.0) == 0
        # IBD yield per fission (Huber–Mueller, ×10⁻⁴³ cm²): 6.69, 10.10, 4.40, 6.03 (within the cross-section convention)
        pure(i) = NamedTuple{R.ISOTOPES}(Tuple(Float64(j == i) for j in R.ISOTOPES))
        for (i, y) in zip(R.ISOTOPES, (6.69, 10.10, 4.40, 6.03))
            @test R.ibd_yield(pure(i)) / 1e-43 ≈ y rtol = 0.015
        end
        # Bugey-4 anchoring lowers the Huber–Mueller rate by the reactor anomaly (~5–6 %)
        @test 0.92 < R.bugey4_scale((U235 = 0.567, U238 = 0.078, Pu239 = 0.298, Pu241 = 0.057)) < 0.97
    end
    @testset "PRIS reactor database" begin
        u = R.load_units()
        @test nrow(u) >= 75
        kk = u[u.unit .== "KASHIWAZAKI KARIWA-6", :][1, :]
        # Chūetsu-oki earthquake: no operation between August 2007 and the end of 2008
        @test sum(R.daily_power(kk, Date(2007, 8, 1), Date(2008, 12, 31))) == 0
        e2005 = sum(R.daily_power(kk, Date(2005, 1, 1), Date(2005, 12, 31)))
        @test e2005 ≈ kk.thermal_power_MW * 0.966 * 365 rtol = 0.01
        det = R.ecef(36.4225, 137.3153)
        @test R.baseline(det, R.ecef(37.4286, 138.5978)) ≈ 160 atol = 2      # Kashiwazaki-Kariwa – Kamioka
    end
end
