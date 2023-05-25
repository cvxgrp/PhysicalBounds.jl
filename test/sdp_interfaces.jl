@testset "SDP Interfaces" begin

    n = 10
    C = maxcut_problem(n; rand_seed=0)

    Eii(x,n) = sparse([x], [x], [1.0], n, n)
    Av = [Eii(x,n) for x in 1:n]
    b = ones(n)

    """
    # The code below solves a test problem (MAXCUT) with JuMP. It was pre-run
    # for efficiency. Results are below

    sdp = Model(Hypatia.Optimizer)
    @variable(sdp, X[1:n, 1:n] in PSDCone())
    @constraint(sdp, diag(X) .== 1)
    @objective(sdp, Min, sum(C .* X))
    JuMP.optimize!(sdp)
    pstar_primal = objective_value(sdp)
    Xv_primal = value.(X)

    sdp_dual = Model(Hypatia.Optimizer)
    @variable(sdp, y[1:n])
    @constraint(sdp, sum(Av[i]*y[i] for i in 1:n) + C in PSDCone())
    @objective(sdp, Min, sum(b[i]*y[i] for i in 1:n))
    JuMP.optimize!(sdp)
    pstar_dual = objective_value(sdp)
    yv_dual = value.(y)
    """
    pstar_primal = -11.974197645018767
    Xv_primal = [1.0 0.11072453249972838 0.9204062559644304 -0.8985877318362102 -0.6868375296392297 0.34596961247273283 -0.9357637567419007 0.9355177861330196 -0.6891391261592946 -0.42456449952800673; 0.11072453249972838 1.0 0.4904708605340585 -0.5355915035229953 -0.7983916642159112 0.9707838677521702 0.2468472162687046 0.4546920513733232 -0.7964780016703872 -0.9468402212774851; 0.9204062559644304 0.4904708605340585 1.0 -0.9986180847075772 -0.9163259845760752 0.6852522338157628 -0.7234184305300837 0.9991757776448295 -0.9175913267238557 -0.744749090366991; -0.8985877318362102 -0.5355915035229953 -0.9986180847075772 1.0 0.9361039948674852 -0.7225805342345523 0.6861350611701668 -0.9956617263520798 0.937214762815553 0.7787914234412205; -0.6868375296392297 -0.7983916642159112 -0.9163259845760752 0.9361039948674852 1.0 -0.9195521956219682 0.3864241543982544 -0.899316197724986 0.9999949576637684 0.9496598084313501; 0.34596961247273283 0.9707838677521702 0.6852522338157628 -0.7225805342345523 -0.9195521956219682 1.0 0.007105384872119334 0.6551237659388607 -0.918301286530664 -0.9963719086840032; -0.9357637567419007 0.2468472162687046 -0.7234184305300837 0.6861350611701668 0.3864241543982544 0.007105384872119334 1.0 -0.750847574747831 0.3893473101483241 0.0780239638735286; 0.9355177861330196 0.4546920513733232 0.9991757776448295 -0.9956617263520798 -0.899316197724986 0.6551237659388607 -0.750847574747831 1.0 -0.9006985470999531 -0.7170461402091903; -0.6891391261592946 -0.7964780016703872 -0.9175913267238557 0.937214762815553 0.9999949576637684 -0.918301286530664 0.3893473101483241 -0.9006985470999531 1.0 0.9486614708646184; -0.42456449952800673 -0.9468402212774851 -0.744749090366991 0.7787914234412205 0.9496598084313501 -0.9963719086840032 0.0780239638735286 -0.7170461402091903 0.9486614708646184 1.0000000000000002]
    pstar_dual = 11.974197681176074
    yv_dual = [1.0642151045101664, 1.537336984143792, 1.4627267938622082, 1.5649168256185602, 1.4406737441706545, 0.8630465441972368, 0.49472792901351303, 1.3142608205268786, 1.1430291630990679, 1.0892637720339957]
    

    @testset "Hypatia" begin
        # Primal form interface
        model = PB.build_Hypatia_model_primal(C, Av, b)
        solver = PB.optimize(model; verbose=false)
        primal_obj = Solvers.get_primal_obj(solver)
        x = Solvers.get_x(solver)

        @test isapprox(PB.unvec_symm_scs(x), Xv_primal, rtol=1e-5)
        @test isapprox(primal_obj, pstar_primal, rtol=1e-5)

        # Dual Form Solve
        model = PB.build_Hypatia_model_dual(C, Av, b)
        solver = PB.optimize(model; verbose=false)
        dual_obj = Solvers.get_primal_obj(solver)
        y = Solvers.get_x(solver)

        @test isapprox(dual_obj, pstar_dual, rtol=1e-5)
        @test isapprox(y, yv_dual, rtol=1e-5)
    end

    @testset "COSMO" begin
        
        model_cosmo = PB.build_COSMO_model_primal(C, Av, b)
        result = PB.optimize!(model_cosmo)
        primal_obj = result.obj_val
        x = result.x

        @test ≈(PB.unvec_symm_scs(x), Xv_primal, rtol=1e-4)
        @test ≈(primal_obj, pstar_primal, rtol=1e-4)

        model_cosmo = PB.build_COSMO_model_dual(C, Av, b)
        result = PB.optimize!(model_cosmo)
        dual_obj = result.obj_val
        y = result.x

        @test ≈(dual_obj, pstar_dual, rtol=1e-4)
        @test ≈(y, yv_dual, rtol=1e-4)
        
    end

    @testset "JuMP" begin
        model, X = PB.primal_problem_solve(C, Av, b)
        primal_obj = objective_value(model)
        @test isapprox(value.(X), Xv_primal, rtol=1e-5)
        @test isapprox(primal_obj, pstar_primal, rtol=1e-5)

        model, y = PB.dual_problem_solve(C, Av, b)
        dual_obj = objective_value(model)
        @test isapprox(dual_obj, pstar_dual, rtol=1e-5)
        @test isapprox(value.(y), yv_dual, rtol=1e-5)
    end
end