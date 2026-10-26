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
        # IBD kinematics (Strumia & Vissani Eq. 16): prompt energy 2 m_e at threshold, E_ν − E_prompt grows with the neutron recoil
        @test R.neutrino_energy(2 * R.M_E) ≈ R.IBD_THRESHOLD atol = 1e-3
        @test R.neutrino_energy(8.0) - 8.0 ≈ 0.853 atol = 2e-3
        @test all(R.positron_energy(R.neutrino_energy(e)) + R.M_E ≈ e for e in 1.1:0.5:9.0)
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
    @testset "Daya Bay unfolded spectra" begin
        hm, dyb = R.configure(), R.configure(R.DayaBayUnfolded())
        @test hm.model isa R.HuberMueller && dyb.model isa R.DayaBayUnfolded
        t = R.read_dyb_unfolded()
        @test length(t.centres) == 25 && length(t.edges) == 26
        # the model reproduces the published bin contents at the Daya Bay fission fractions (Eq. 9 with Δf = 0)
        f = R.DYB_FRACTIONS
        @test all(R.bin_mean(E -> dyb.spectrum(E, f), t.edges[b], t.edges[b+1]) ≈ t.total[b] for b in 1:25)
        @test all(R.bin_mean(E -> dyb.isotope_spectrum(:U235, E), t.edges[b], t.edges[b+1]) ≈ t.U235[b] for b in 1:25)
        # the 5 MeV excess and the flux deficit relative to Huber–Mueller
        @test dyb.spectrum(6.0, f) / hm.spectrum(6.0, f) > 1.03
        @test dyb.spectrum(3.5, f) / hm.spectrum(3.5, f) < 0.96
        # Eq. 9 is linear in the fission fractions; at other fractions the per-isotope spectra add up to it
        g = (U235 = 0.6, U238 = 0.07, Pu239 = 0.27, Pu241 = 0.06)
        @test dyb.spectrum(5.0, g) - dyb.spectrum(5.0, f) ≈ sum((g[i] - f[i]) * dyb.isotope_spectrum(i, 5.0) for i in R.ISOTOPES)
        @test dyb.rate_scale(g) == 1.0
        @test hm.rate_scale(g) ≈ R.bugey4_scale(g)
    end
end
