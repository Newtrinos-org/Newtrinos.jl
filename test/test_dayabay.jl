using LinearAlgebra
using HDF5
using Test
using Newtrinos

@testset "DayaBay" begin
    @testset "Meta" begin
        p_AD = "6ADAD11"
        @test 11 == Newtrinos.dayabay_rewrite.retrieve_AD(p_AD)
        @test 6 == Newtrinos.dayabay_rewrite.retrieve_period(p_AD)
        @test 1 == Newtrinos.dayabay_rewrite.retrieve_EH(p_AD)
    end

    db = Newtrinos.dayabay_rewrite.configure()
    @testset "IAV" begin
        iav_func = Newtrinos.dayabay_rewrite.get_iav_matrix()
        iav = iav_func(1.0)
        N = size(iav)[1]
        for i in 1:20
            delta = zeros(N)
            delta[i] = 1.0
            @test isapprox(sum(iav[:, i]), 0.0)
            @test isapprox(sum(iav * delta), 0.0)
            break
        end
        for i in 21:N
            delta = zeros(N)
            delta[i] = 1.0
            @test isapprox(sum(iav[:, i]), 1.0)
            @test isapprox(sum(iav * delta), 1.0)
            break
        end
        # the off-diagonal scale changes the shape, not the number of events (official RenormalizeDiag "offdiag")
        scaled = iav_func(2)
        @test isapprox(scaled[1:99, 100] ./ scaled[100, 100], 2.0 .* iav[1:99, 100] ./ iav[100, 100])
        @test isapprox(sum(scaled[:, 100]), 1.0)
        spec = rand(N)
        @test isapprox(iav_func(1.04, spec), iav_func(1.04) * spec)
        @test isapprox(sum(iav_func(1.04, spec)), sum(iav_func(1.0, spec)))
    end

    @testset "LSNL" begin
        lsnl_data = Newtrinos.dayabay_rewrite.read_lsnl_correction()
        lsnl = Newtrinos.dayabay_rewrite.get_lsnl_correction()
        # absolute Evis(Escint); the refined curves pass through the tabulated points
        E = lsnl_data.E
        @test isapprox(lsnl(E, [0, 0, 0, 0]), lsnl_data.f_nom .* E)
        @test isapprox(lsnl(E, [1, 0, 0, 0]), (lsnl_data.f_nom .+ lsnl_data.rel_0) .* E)
        @test isapprox(lsnl(E, [0, 1, 0, 0]), (lsnl_data.f_nom .+ lsnl_data.rel_1) .* E)
        @test isapprox(lsnl(E, [0, 0, 1, 0]), (lsnl_data.f_nom .+ lsnl_data.rel_2) .* E)
        @test isapprox(lsnl(E, [0, 0, 0, 1]), (lsnl_data.f_nom .+ lsnl_data.rel_3) .* E)
        # linear extrapolation of the absolute curve down to 0.5 MeV (official refine_lsnl_data)
        @test 0.69 < lsnl([0.7], zeros(4))[1] < 0.70
        @test all(diff(lsnl(collect(0.5:0.01:12.0), zeros(4))) .> 0)
    end

    @testset "energy response" begin
        D = Newtrinos.dayabay_rewrite
        params = db.params
        edges = collect(0.0:0.05:12.0)
        c = (edges[1:end-1] .+ edges[2:end]) ./ 2
        sigma_E = D.eres(c, params.eres_a, params.eres_b, params.eres_c)
        for i in 30:10:200
            counts = zeros(length(c))
            counts[i] = 1
            @test isapprox(sum(D.smear_bin_centres(counts, sigma_E, edges)), 1.0, rtol = 1e-6)
        end
        counts = rand(length(c))
        image = D.get_lsnl_correction()(edges, zeros(4))
        # events mapped above the last edge (12 MeV) are lost
        mask = (edges[1:end-1] .>= 0.5) .& (image[2:end] .<= 12.0)
        @test isapprox(sum(D.project_to_bins(image, counts .* mask, edges)), sum(counts .* mask))
        @test D.project_to_bins(edges, counts, edges) ≈ counts
        idx = D.rebin_index(edges, [0.7, 1.3, 1.55, 12.0])
        @test count(==(1), idx) == 12 && count(==(2), idx) == 5 && count(==(3), idx) == 209 && count(==(0), idx) == 14
    end

    assets = db.assets
    params = Newtrinos.get_params((;db))
    @testset "assets" begin
        fwd = db.forward_model
        @test length(db.assets.observed) == length(fwd(params))
    end

    @testset "parameters" begin
        @test length(params.acc_scale) == length(assets.detectors_6AD) + length(assets.detectors_8AD) + length(assets.detectors_7AD)
        @test length(params.alpha_n_rate) == length(assets.detectors_6AD) + length(assets.detectors_8AD) + length(assets.detectors_7AD)
        @test length(params.fast_n_unc_scale) == length(assets.EH_list) * length(assets.period_list)
        @test length(params.lihe_unc_scale) == length(assets.EH_list) * length(assets.period_list)
        @test length(params.iav_offdiag_scale) == length(assets.df_exp[!, "AD"])
        @test length(params.lsnl_pull) == 4
    end

end