using Test
using Newtrinos
using StructArrays
using Distributions

@testset "Earth Layers" begin

    @testset "Configuration" begin
        el = Newtrinos.earth_layers.configure()
        @test el isa Newtrinos.earth_layers.EarthLayers
        @test el.cfg isa Newtrinos.earth_layers.PREM
    end

    @testset "VariableDensity configuration" begin
        el = Newtrinos.earth_layers.configure(Newtrinos.earth_layers.VariableDensity())
        @test el isa Newtrinos.earth_layers.EarthLayers
        @test el.cfg isa Newtrinos.earth_layers.VariableDensity
        @test haskey(el.params, :electron_density_scale)
        @test el.params.electron_density_scale == 1.0
        @test haskey(el.priors, :electron_density_scale)
        @test Distributions.insupport(el.priors.electron_density_scale, el.params.electron_density_scale)

        # compute_layers/compute_paths are unaffected by VariableDensity itself (delegate
        # to the wrapped PREM profile) — the scale is applied separately, see scale_densities
        el_prem = Newtrinos.earth_layers.configure(Newtrinos.earth_layers.PREM())
        layers_var = el.compute_layers()
        layers_prem = el_prem.compute_layers()
        @test layers_var.radius == layers_prem.radius
        @test layers_var.p_density == layers_prem.p_density
        @test layers_var.n_density == layers_prem.n_density
    end

    @testset "scale_densities" begin
        el = Newtrinos.earth_layers.configure()
        layers = el.compute_layers()

        # scale = 1.0 is a no-op
        layers_unit = Newtrinos.earth_layers.scale_densities(layers, 1.0)
        @test layers_unit.p_density ≈ layers.p_density
        @test layers_unit.n_density ≈ layers.n_density
        @test layers_unit.radius == layers.radius

        # scale > 1: proton (electron) density increases proportionally
        scale = 1.2
        layers_scaled = Newtrinos.earth_layers.scale_densities(layers, scale)
        @test layers_scaled.p_density ≈ layers.p_density .* scale

        # total density (proton + neutron) is conserved
        @test layers_scaled.p_density .+ layers_scaled.n_density ≈ layers.p_density .+ layers.n_density atol = 1e-10

        # radii untouched
        @test layers_scaled.radius == layers.radius

        # scale < 1 also conserves total density and stays non-negative for physical scales
        layers_down = Newtrinos.earth_layers.scale_densities(layers, 0.8)
        @test layers_down.p_density .+ layers_down.n_density ≈ layers.p_density .+ layers.n_density atol = 1e-10
        @test all(layers_down.p_density .>= -1e-10)

        # end-to-end: VariableDensity's electron_density_scale param, applied via
        # scale_densities, reproduces the same layers as calling scale_densities directly
        el_var = Newtrinos.earth_layers.configure(Newtrinos.earth_layers.VariableDensity())
        params = merge(el_var.params, (electron_density_scale = scale,))
        layers_via_param = Newtrinos.earth_layers.scale_densities(el_var.compute_layers(), params.electron_density_scale)
        @test layers_via_param.p_density ≈ layers_scaled.p_density
    end

    @testset "ray_circle_path_length" begin
        #L = rcpl(radius, radius_detector, cos(zenith))
        rcpl = Newtrinos.earth_layers.ray_circle_path_length
        
        # Ray through center of circle (cz = -1, straight down)
        L = rcpl(6371.0, 6371.0, -1.0) 
        @test L ≈ 2 * 6371.0 atol=1.0

        # No intersection when radius is too small
        L = rcpl(1.0, 6371.0, -0.1) 
        @test L == 0.0

        # Horizontal ray (cz ≈ 0) at the surface should have zero path length through the Earth
        L_inner = rcpl(3000.0, 6371.0, 0.0) 
        @test L_inner == 0.0  

        # Detector at sphere surface, straight down L = 2r
        @test rcpl(6371.0, 6371.0, -1.0) ≈ 12742.0

        # Detector inside sphere, straight down L = y + r
        @test rcpl(6371.0, 5000.0, -1.0) ≈ 11371.0

        # Detector outside sphere, straight down L = 2r
        @test rcpl(3000.0, 6371.0, -1.0) ≈ 6000.0

        # Detector at center of sphere → L = r (forward half only)
        @test rcpl(6371.0, 0.0, -1.0) ≈ 6371.0

        # different directions equivalent at center
        @test rcpl(6371.0, 0.0, 0.0) ≈ rcpl(6371.0, 0.0, -0.5)

        # Outside sphere (r=6000, y=6371), angled ray that hits with cz = -0.8 
        expected = 2 * sqrt(6000.0^2 - 6371.0^2 * 0.36)
        @test rcpl(6000.0, 6371.0, -0.8) ≈ expected

        # No intersection (disc < 0) (r=5000, y=6371, cz=-0.5)
        @test rcpl(5000.0, 6371.0, -0.5) == 0.0

        # Tangent case L = 0
        r_tan = 3185.5
        y_tan = 6371.0
        cz_tan = -sqrt(1.0 - (r_tan / y_tan)^2)
        @test rcpl(r_tan, y_tan, cz_tan) == 0.0

        # Detector inside, ray going upward (r=6371, y=5000, cz=+0.5)
        expected_up = -2500.0 + sqrt(6371.0^2 - 5000.0^2 + (5000.0 * 0.5)^2)
        @test rcpl(6371.0, 5000.0, 0.5) ≈ expected_up

        # L < 1 threshold: chord exists but is sub-km 
        r_tiny = 6371.0 * sqrt(0.75) + 0.00001
        @test rcpl(r_tiny, 6371.0, -0.5) == 0.0
    end

    @testset "Layer computation" begin
        el = Newtrinos.earth_layers.configure()
        layers = el.compute_layers()
        # Should have multiple layers
        @test length(layers) > 1 

        # Radii should be decreasing (from atmosphere to core)
        @test issorted(layers.radius, rev=true) 

        # Densities should be non-negative
        @test all(layers.p_density .>= 0.0)
        @test all(layers.n_density .>= 0.0)

        # Outermost layer should have zero density (atmosphere)
        @test layers.p_density[1] ≈ 0.0
        @test layers.n_density[1] ≈ 0.0

        # Inner layers should have higher density
        @test layers.p_density[end] > layers.p_density[2]
    end

    @testset "Path computation" begin
        el = Newtrinos.earth_layers.configure()
        layers = el.compute_layers()
        r_outermost = layers.radius[1]
        rcpl = Newtrinos.earth_layers.ray_circle_path_length
        r_det = 6369.0

        # Total path length for straight down should be ~2 * R_earth
        paths_down = el.compute_paths([-1.0], layers)
        total_length = sum(p.length for p in paths_down[1])
        @test total_length ≈ 2 * 6369.0 atol=100.0  # approximately diameter

        # Horizontal path should be shorter than vertical
        paths_horiz = el.compute_paths([-0.1], layers)
        total_horiz = sum(p.length for p in paths_horiz[1])
        @test total_horiz < total_length

        # Multiple coszen values
        czs = [-1.0, -0.5, -0.1]
        paths_multi = el.compute_paths(czs, layers)
        @test length(paths_multi) == 3

        # More vertical = longer path
        lengths = [sum(p.length for p in path) for path in paths_multi]
        @test issorted(lengths, rev=true)

        # Total path invariant: for any cz, total = rcpl(r_outermost, r_det, cz)
        for cz in [-1.0, -0.5, -0.3, -0.1]
            for r_det in [6369.0, 6371.0, 5500.0]
                paths = Newtrinos.earth_layers.compute_paths(cz, layers, r_det)
                total = sum(p.length for p in paths)
                expected = rcpl(r_outermost, r_det, cz)
                @test total ≈ expected atol=1.0
            end
        end

        # Vector interface uses default r_detector = 6369
        czs = [-1.0, -0.5, -0.1]
        paths_vec = el.compute_paths(czs, layers)
        for (i, cz) in enumerate(czs)
            paths_explicit = Newtrinos.earth_layers.compute_paths(cz, layers, 6369)
            total_vec = sum(p.length for p in paths_vec[i])
            total_explicit = sum(p.length for p in paths_explicit)
            @test total_vec ≈ total_explicit
        end

        # critical cases
        # Horizontal ray (cz = 0): only outermost layers intersected
        paths_horiz = Newtrinos.earth_layers.compute_paths(0.0, layers, r_det)
        total_horiz = sum(p.length for p in paths_horiz)
        expected_horiz = rcpl(r_outermost, r_det, 0.0)
        @test total_horiz ≈ expected_horiz atol=1.0

        # Near-horizontal (cz = -0.01): short path, few layers
        paths_near = Newtrinos.earth_layers.compute_paths(-0.01, layers, r_det)
        @test length(paths_near) >= 1
        total_near = sum(p.length for p in paths_near)
        @test total_near > total_horiz  # slightly more vertical = slightly longer
        @test total_near < sum(p.length for p in Newtrinos.earth_layers.compute_paths(-1.0, layers, r_det))

        # Monotonicity: more vertical = longer total path
        czs = [0.0, -0.01, -0.1, -0.3, -0.5, -0.7, -0.9, -1.0]
        totals = [sum(p.length for p in Newtrinos.earth_layers.compute_paths(cz, layers, r_det)) for cz in czs]
        @test issorted(totals)

        # No zero-length path segments
        for cz in [-1.0, -0.5, -0.1]
            paths = Newtrinos.earth_layers.compute_paths(cz, layers, r_det)
            for p in paths
                @test p.length > 0.0
            end
        end

    end

    # test with custom layer configurations to verify path calculations in controlled scenarios
    @testset "Custom layer configurations" begin        
        rcpl = Newtrinos.earth_layers.ray_circle_path_length
        
        # Two-layer model: mantle + core (mantle: r=6371, core: r=3000)
        # cz=-1, r_detector=6371 (at surface)
        layers_2 = StructArray{Newtrinos.Layer}(
            ([6371.0, 3000.0],   # radius
             [2.0, 5.0],         # p_density
             [2.0, 5.0])         # n_density
        )

        paths_2 = Newtrinos.earth_layers.compute_paths(-1.0, layers_2, 6371.0)
        @test length(paths_2) == 3
        @test paths_2[1].length ≈ 3371.0 # 2*6371 total path, minus 2*3000 through core -> 2*3371 = 6742 through mantle
        @test paths_2[1].layer_idx == 1
        @test paths_2[2].length ≈ 6000.0 # 2*3000 total path through core
        @test paths_2[2].layer_idx == 2
        @test paths_2[3].length ≈ 3371.0 #similar to first mantle segment
        @test paths_2[3].layer_idx == 1
        @test sum(p.length for p in paths_2) ≈ 12742.0

        # Three-layer model: atmosphere + mantle + core (atmosphere: r=6391, mantle: r=6371, core: r=3480)
        # cz=-1, r_detector=6369 (2km below surface)
        layers_3 = StructArray{Newtrinos.Layer}(
            ([6391.0, 6371.0, 3480.0],
             [0.0, 2.0, 5.0],
             [0.0, 2.0, 5.0])
        )

        paths_3 = Newtrinos.earth_layers.compute_paths(-1.0, layers_3, 6369.0)
        @test length(paths_3) == 4
        @test paths_3[1].length ≈ 20.0 atol=1.0
        @test paths_3[1].layer_idx == 1
        @test paths_3[2].length ≈ 2891.0 atol=1.0 # 2*6371 total path, minus 2*3480 through core -> 2*2891 = 5782 through mantle
        @test paths_3[2].layer_idx == 2
        @test paths_3[3].length ≈ 6960.0 atol=1.0 # 2*3480 total path through core
        @test paths_3[3].layer_idx == 3
        @test paths_3[4].length ≈ 2889.0 atol=1.0 # similar to first mantle segment, but -2 km due to detector being 2km below surface
        @test paths_3[4].layer_idx == 2
        @test sum(p.length for p in paths_3) ≈ 6369.0 + 6391.0 atol=1.0

        # Single-layer model
        # cz = -1, r_detector=6371 (at surface)
        layers_1 = StructArray{Newtrinos.Layer}(
            ([6371.0,],
             [5.0,],
             [5.0,])
        )
        paths_1 = Newtrinos.earth_layers.compute_paths(-1.0, layers_1, 6371.0)
        @test sum(p.length for p in paths_1) ≈ 12742.0

        # verify symmetry of mantle segment lengths for ingoing/outgoing paths (for 2 layer config)
        # cz=-0.5, r_detector=6371
        paths_angled = Newtrinos.earth_layers.compute_paths(-0.5, layers_2, 6371.0)
        mantle_segments = [p for p in paths_angled if p.layer_idx == 1]
        if length(mantle_segments) == 2
            @test mantle_segments[1].length ≈ mantle_segments[2].length atol=0.1
        end
    end
end
