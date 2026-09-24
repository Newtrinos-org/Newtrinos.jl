@testset "Helpers" begin

    @testset "bin" begin
        # Simple binning of events
        df = [1.5; 2.5; 3.5; 4.5;;]  # 4 events as a matrix with 1 column
        edges = [1.0, 2.0, 3.0, 4.0, 5.0]
        counts = Newtrinos.Helpers.bin(df, edges)
        @test counts == [1.0, 1.0, 1.0, 1.0]

        # Multiple events in same bin
        df = [1.1; 1.5; 1.9; 3.5;;]
        counts = Newtrinos.Helpers.bin(df, edges)
        @test counts == [3.0, 0.0, 1.0, 0.0]

        # Events outside range are excluded, 1.5 is in bin 1
        df = [0.5; 1.5; 5.5;;]
        counts = Newtrinos.Helpers.bin(df, edges)
        @test counts == [1.0, 0.0, 0.0, 0.0]

        # Edge value: searchsortedlast puts 2.0 in bin [2.0, 3.0)
        df = [2.0;;]
        counts = Newtrinos.Helpers.bin(df, edges)
        @test counts == [0.0, 1.0, 0.0, 0.0]

        # Value at first edge goes into first bin
        df = [1.0;;]
        counts = Newtrinos.Helpers.bin(df, edges)
        @test counts == [1.0, 0.0, 0.0, 0.0]

        # Empty input
        df = zeros(0, 1)
        counts = Newtrinos.Helpers.bin(df, edges)
        @test counts == [0.0, 0.0, 0.0, 0.0]
    end

    @testset "rebin" begin
        # Simple rebinning
        df = [1.5 10.0; 2.5 20.0; 3.5 30.0]
        edges = [1.0, 2.0, 3.0, 4.0]
        counts = Newtrinos.Helpers.rebin(df, edges)
        @test counts == [10.0, 20.0, 30.0]

        # Multiple entries in same bin get summed
        df = [1.1 5.0; 1.5 15.0; 3.5 30.0]
        counts = Newtrinos.Helpers.rebin(df, edges)
        @test counts == [20.0, 0.0, 30.0]

        # Entries outside range excluded, 1.5 is in bin 1
        df = [0.5 100.0; 1.5 10.0; 4.5 100.0]
        counts = Newtrinos.Helpers.rebin(df, edges)
        @test counts == [10.0, 0.0, 0.0]
    end
end
