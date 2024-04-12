
using LinearAlgebra
using Pkg
#using Pkg
#Pkg.add("QuantumInformation")
#Pkg.add("JLD2")
#Pkg.add("FileIO")
#Pkg.add("CSV")
#Pkg.add("DataFrames")
#Pkg.add("Plots")
#Pkg.add("Measurements")

using QuantumInformation
using DelimitedFiles
using JLD2, FileIO
using CSV, DataFrames

using Plots, Measurements
Plots.gr()
include("rho_sampling.jl")

ffprec = 1e-15
#Id2 = Matrix{ComplexF64}(I, 2, 2)

psi_1 = [0 1 -1 0]*(1/sqrt(2))
psi_2 = [0 1 1 0]*(1/sqrt(2))
psi_3 = [1 0 0 -1]*(1/sqrt(2))
psi_4 = [1 0 0 1]*(1/sqrt(2))

# Bell projectors
P1 = reshape(kron(psi_1, psi_1), (4,4))
P2 = reshape(kron(psi_2, psi_2), (4,4))
P3 = reshape(kron(psi_3, psi_3), (4,4))
P4 = reshape(kron(psi_4, psi_4), (4,4))

P13 = reshape(kron(psi_1, psi_3), (4,4))
P24 = reshape(kron(psi_2, psi_4), (4,4))
P14 = reshape(kron(psi_1, psi_4), (4,4))
P23 = reshape(kron(psi_2, psi_3), (4,4))

Id2 = [1 0; 0 1]
Id4 = Id2 ⊗ Id2
Id16 = Id4 ⊗ Id4 
CNOT = [1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 0. 1.; 0. 0. 1. 0.]
CNOT_0 = [0. 1. 0. 0.; 1. 0. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.]

# measurement
p00 = [1. 0. 0. 0.;0. 0. 0. 0.;0. 0. 0. 0.;0. 0. 0. 0.]
p01 = [0. 0. 0. 0.;0. 1. 0. 0.;0. 0. 0. 0.;0. 0. 0. 0.]
p10 = [0. 0. 0. 0.;0. 0. 0. 0.;0. 0. 1. 0.;0. 0. 0. 0.]
p11 = [0. 0. 0. 0.;0. 0. 0. 0.;0. 0. 0. 0.;0. 0. 0. 1.]

p_all = (p00, p01, p10, p11)
U = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]

s_x = [0. 1.;1. 0.]
s_y = [0 -1im; 1im  0]
s_z = [1. 0.;0. -1.]
# Operations for the Bennet protocol

b1 = (Id2 + 1im*s_x)*(1/sqrt(2))
b2 = (Id2 - 1im*s_y)*(1/sqrt(2))

b3 = [1im 0.;0. 1.]
b4 = Id2
B1 = b1 ⊗ b1
B2 = b2 ⊗ b2

B3 = b3 ⊗ b3
B4 = b4 ⊗ b4
b_ops = (b1 ⊗ b1, b2 ⊗ b2, b3 ⊗ b3, b4 ⊗ b4)
s_y_A1 = (s_y ⊗ Id2) ⊗ Id2 ⊗ Id2
s_y_A2 = (Id2 ⊗ Id2)  ⊗ (s_y ⊗ Id2)
b_A1A2 = b1' ⊗ b1 ⊗ b1' ⊗ b1

W = [1. 0. 0. 0.; 0. 0. 0. 1.; 0. 1. 0. 0.; 0. 0. 1. 0.]

# Calculates the fidelities with respect to the fix points of the Bennett protocol if
# the state is purifiable according to the Bennett conditions.

function fidelity_bennett(rho)
    
    r1 = real(tr(rho*P1))

    if r1 > 0.5
        out = r1
    else
        out = 0
    end

    out
end

# Calculates the fidelities with respect to the fix points of the Deutsch protocol if
# the state is purifiable according to the Deutsch conditions.

function fidelities_deutsch(rho)

    r1 = real(tr(rho*P1))
    r2 = real(tr(rho*P2))
    r3 = real(tr(rho*P3))
    r4 = real(tr(rho*P4))

    if (2*r1 - 1)*(1 - 2*r4) > 0
        out4 = r4
    else
        out4 = 0
    end

    if  (2*r2 - 1)*(1 - 2*r3) > 0
        out2 = r2
    else
        out2 = 0
    end 

    out2, out4
end

# Calculates the fidelities with respect to the fix points of the MFI protocol if
# the state is purifiable according to the MFI conditions.

function fidelities_mfi(rho)

        r1 =  real(tr(P1 * rho))
        r2 =  real(tr(P2 * rho))
        r3 =  real(tr(P3 * rho))
        r4 =  real(tr(P4 * rho))
        r13 = tr(P13 * rho)
        r24 = tr(P24 * rho)
    
        if (2*r1 - 1)*(1 - 2*r3) > -4*imag(r13)^2 - 4*real(r24)^2 
            out2=r1
        else
            out2=0
        end

        if (2*r2-1)*(1 - 2*r4) > -4*imag(r24)^2 - 4*real(r13)^2
            out4=r2
        else
            out4=0
        end

    out2, out4
end

# Calculates the fidelities with respect to the fix points of the CNOT protocol if
# the state is purifiable according to the CNOT conditions.

function fidelities_cnot(rho)

        r1 =  real(tr(P1 * rho))
        r2 =  real(tr(P2 * rho))
        r3 =  real(tr(P3 * rho))
        r4 =  real(tr(P4 * rho))
        r14 = tr(P14 * rho)
        r23 = tr(P23 * rho)
    
        if (2*r1 - 1)*(1 - 2*r4) > -4*imag(r23)^2 - 4*real(r14)^2 
            out2=r4
        else
            out2=0
        end
        
        if (2*r2-1)*(1 - 2*r3) > -4*imag(r14)^2 - 4*real(r23)^2
            out4=r2
        else
            out4=0
        end

    out2, out4
end

# Calculates the concurrence of a 4x4 density matrix

function partial_transpose_(rho, d1, d2)
    idx = [d2 for i = 1:d1]
    blkm = PseudoBlockArray(rho, idx, idx)
    bfm = Array{eltype(rho)}(undef, d2, d2)
    trm = Array{eltype(rho)}(undef, d2, d2)
    for i = 1:d1
        for j = 1:d1
            getblock!(bfm, blkm, i, j);
            transpose!(trm, bfm)
            setblock!(blkm, trm, i, j)
        end
    end
    rho
end

# Calculates the concurrence of a 4x4 density matrix

function partial_trace_(rho, d1, d2)
    idx = [d2 for i = 1:d1]
    blkm = PseudoBlockArray(rho, idx, idx)
    bfm = Array{eltype(rho)}(undef, d2, d2)
    trm = Array{eltype(rho)}(undef, d2, d2)
    
    rho
end

function Sn(i,j)
    S = copy(Id_4)
    S[i,j] = 0
    S[j, j] = 0
    S[i, j] = 1
    S[j, i] = 1
    S
end

# Mapping for A1B1

function m1_A1B1(mat)
    m = tensor([Id4, mat])
    perm = tensor(tensor(Id2, Sn(2,3)), Id2)
    m = perm * mat * perm'
end

# Mapping for A2B2

function m1_A1B1(mat)
    m = tensor([mat, Id4])
    perm = tensor(tensor(Id2, Sn(2,3)), Id2)
    m = perm * mat * perm'
end

# Converts dm to Werner form.

function convert_to_werner(dmat)
    #println(tr(dmat))

    dmat_new = (1/4)* (B1'* B1' * dmat * B1 * B1 + 
                B2'* B2' * dmat * B2 * B2 + 
                B3'* B3' * dmat * B3 * B3 + 
                B4'* B4' * dmat * B4 * B4)
    #println(B2'* B2' * B2 * B2)


    dmat_new_2 = (1/3)*(B1' * dmat_new * B1 + B2' * dmat_new * B2 + B3' * dmat_new * B3)
    #println(tr(dmat_new_2))
    dmat_new_2
end

# Converts dm from bell diagonal to Werner form.

function convert_bell_to_werner(dmat)

    dmat_werner = (1/3)*(B1' * dmat * B1 + B2' * dmat * B2 + B3' * dmat * B3)


    dmat_werner
end

# Converts dm to bell diagonal form.

function convert_to_bell_diag(dmat)

    dmat_bd = (1/4)* (B1'* B1' * dmat * B1 * B1 + 
                B2'* B2' * dmat * B2 * B2 + 
                B3'* B3' * dmat * B3 * B3 + 
                B4'* B4' * dmat * B4 * B4)
    dmat_bd

end

# Applies a quantum operation to the density matrix rho.

function apply_operation(rho, op_matrix)

    G_a = op_matrix
    Gc_a = op_matrix'

    UU = reshape((Id2 ⊗ U) ⊗ Id2, 16, 16)

    Ma = G_a ⊗ Id4
    Ma_c = Gc_a ⊗ Id4
    Ua = (UU * Ma) * UU
    Ua_c = (UU * Ma_c) * UU

    Mb = Id4 ⊗ G_a
    Mb_c = Id4 ⊗ Gc_a
    Ub = (UU * Mb) * UU
    Ub_c = (UU * Mb_c) * UU



    rho_new = (Ua * Ub * rho * Ub_c * Ua_c)


    return rho_new

end

# Purification step for the MFI protocol

function mfi_purification_step(dmat1, dmat2, M)

    rho_proj = zeros(ComplexF64, 4, size(dmat1)[1], size(dmat1)[2])
    c = zeros(Float64, 4)
    p = zeros(Float64, 4)
    fid = zeros(Float64, 2, 4)

    # General protocol: if concurrence increases, keep that state, throw the rest away

    dmat_pair = dmat1 ⊗ dmat2

    rho_t = apply_operation(dmat_pair, M)



    for i in 1:4
        proj =  Id4 ⊗ p_all[i]
      

        rho_t_new = (proj * rho_t * proj')
        prob = real(tr(proj * proj' * rho_t))

        rho_t_new = rho_t_new / (prob + ffprec)
        
        rf = ptrace(rho_t_new, [2,2,2,2], [3,4])
   
        c[i] = real(compute_concurrence_rho(rf))

        if c[i] > 0
            p[i] = real(prob)
        else
            p[i] = 0
        end
   
        if i==1
            op = (b3 ⊗ (b3 * s_x))
        elseif i==2
            op = (b3 ⊗ b3)
        elseif i==3
            op = ((b3 * s_x) ⊗ (b3 * s_x))
        else
            op = ((b3 * s_x) ⊗ b3)
        end
        


        rf = op * rf * op'

        fids = fidelities_mfi(rf)
        fid[1, i] = real(fids[1])
        fid[2, i] = real(fids[2])

        rho_proj[i,:,:] = rf
    end


    return rho_proj[1, :, :], c[1], 2*p[1], fid[:, 1]

end

# Purification step for the Deutsch protocol.

function deutsch_purification_step(dmat1, dmat2, unitary_gate)

    dmat1 = convert_to_bell_diag(dmat1)
    dmat2 = convert_to_bell_diag(dmat2)

    dmat_pair = dmat1 ⊗ dmat2
    dmat_pair = b_A1A2 * dmat_pair * b_A1A2'

    rho_t = apply_operation(dmat_pair, unitary_gate)
 
    rho_proj = zeros(ComplexF64, 4, size(dmat1)[1], size(dmat1)[2])
    c = zeros(Float64, 4)
    p = zeros(Float64, 4)
    fid = zeros(Float64, 2, 4)
    
    #See https://journals.aps.org/pra/pdf/10.1103/PhysRevA.93.032317
    # General protocol: if concurrence increases, keep that state, throw the rest away

    for i in 1:4
        proj =  Id4 ⊗ p_all[i]
        rho_t_new = (proj * rho_t * proj')
        prob = real(tr(proj * proj' * rho_t))

        rho_t_new = rho_t_new / (prob + ffprec)

        rf = ptrace(rho_t_new, [2,2,2,2], [3,4])

        
        c[i] = real(compute_concurrence_rho(rf))
        fids = fidelities_deutsch(rf)
        fid[1, i] = real(fids[1])
        fid[2, i] = real(fids[2])

        if c[i] > 0
            p[i] = prob
        else
            p[i] = 0
        end

        rho_proj[i,:,:] = rf
    end

    # Here we need to write the discarding mechanism
    return rho_proj[1, :, :], c[1], 2*p[1], fid[:, 1]

end

# Success probability of the Deutsch protocol.

function success_prob_deutsch(state)
    r1 =  real(tr(P1 * state))
    r2 =  real(tr(P2 * state))
    r3 =  real(tr(P3 * state))
    r4 =  real(tr(P4 * state))
    D = (r1 + r4)^2 + (r2 + r3)^2
    return D
end

# 1-qubit rotation unitary.

function r_sigma(alpha, beta, gamma)
    
    U = [cos(alpha/2) -im * exp(-im * gamma) * sin(alpha/2); -im * exp(im * gamma) * sin(alpha/2) cos(alpha/2)]

    return U
end

# 2-qubit rotation unitary.

function two_qubit_rotation(alpha)
    
    U_1 = r_sigma(alpha[1], alpha[2], alpha[3]) ⊗ r_sigma(alpha[4], alpha[5], alpha[6])
    U_2 = r_sigma(alpha[7], alpha[8], alpha[9]) ⊗ r_sigma(alpha[10], alpha[11], alpha[12])
    U = U_1 * CNOT * U_2 * CNOT_0

    return U
end

# Purification step of the cnot protocol.

function cnot_purification_step(dmat1, dmat2, unitary_gate)

    dmat_pair = dmat1 ⊗ dmat2
    dmat_pair = b_A1A2 * dmat_pair * b_A1A2'

    rho_t = apply_operation(dmat_pair, unitary_gate)

    proj =  Id4 ⊗ p11
    rho_t_new = (proj * rho_t * proj')

    rho_t_new = rho_t_new / (tr(proj * proj' * rho_t) + ffprec)
    rf = ptrace(rho_t_new, [2,2,2,2], [3,4])
    c = real(compute_concurrence_rho(rf))
    fids = fidelities_cnot(rf)

    if c > 0
        prob = real(tr(proj * proj' * rho_t))
    else
        prob = 0
    end



    return rf, c, prob, fids

end
 
# Purificaiton step of the Bennett protocol.

function bennett_purification_step(dmat1, dmat2, unitary_gate)

    # convert to Werner state

    dmat1 = convert_to_werner(dmat1)
    dmat2 = convert_to_werner(dmat2)

    dmat_pair = dmat1 ⊗ dmat2
    dmat_pair = s_y_A1 * s_y_A2 * dmat_pair * s_y_A2' * s_y_A1'

    rho_t = apply_operation(dmat_pair, unitary_gate)

    rho_proj = zeros(ComplexF64, 4, size(dmat1)[1], size(dmat1)[2])
    c = zeros(Float64, 4)
    p = zeros(Float64, 4)
    fid  = zeros(Float64, 2, 4)
    
    #https://journals.aps.org/pra/pdf/10.1103/PhysRevA.93.032317

    for i in 1:4
        proj =  Id4 ⊗ p_all[i]
        rho_t_new = (proj * rho_t * proj')
        prob = real(tr(proj * proj' * rho_t))

        rho_t_new = rho_t_new / (prob + ffprec)
  
        rf = ptrace(rho_t_new, [2,2,2,2], [3,4])
  
        if i == 1 || i == 4
            rf = (sy ⊗ Id2) * rf * (sy ⊗ Id2)'
        else
            continue
        end

        c[i] = real(compute_concurrence_rho(rf)) # concurrence
        fid[1, i] = real(fidelity_bennett(rf)) # fidelity for Bennett

        if c[i] > 0 # Probability function
            p[i] = prob
        else
            p[i] = 0
        end
        rho_proj[i,:,:] = rf
    end

    return rho_proj[1, :, :], c[1], 2*p[1], fid[:, 1]

end

# Success probability of the Bennett protocol.

function success_prob_bennett(state)
    r1 = real(tr(P1 * state))
    D = (5 - 4r1 + 8r1^2)/9
    return D
end

# Iterative application of the Protocols.

function apply_n(f, x_0, x_1, cycle_len, arg)

    c_list = zeros(cycle_len)
    p_list = zeros(cycle_len)
    fid_list = zeros(2, cycle_len)

    for j in 1:cycle_len
        #print(x_0)
        x_0, c, p, fid = f(x_0, x_0, arg)
        c_list[j] = real(c)
        p_list[j] = real(p)
        #println(fid)
        fid_list[1, j] = real(fid[1])
        fid_list[2, j] = real(fid[2])

    end
    #if c_list[1] > 0.2
    #    println(p_list)
    #    println(c_list)
    #end
    return c_list, p_list, fid_list
end

# Werner state.

function special_werner(F)
    F*P1 + ((1-F)/3)*(P2 + P3 + P4)
end

# Bell-diagonal state.

function bell_diagonal(F2, F3, F4)
    (1 - (F4 + F2 + F3))*P1 + F2*P2 + F3*P3 + P4*F4
end

# Loads dm data from .txt file

function load_dm_data(dm_file, n)
    dm_data = zeros(4, 4, n)
    for i in 1:n
        dm_data[:, i] = load(dm_file * "_" * string(i) * ".txt")
    end
    dm_data
end

# Samples density matrices and purifies them

function sample_and_purify(protocol_step, n, m_purif, operation, use_b=true, saved_data=nothing)

    rep = n

    if isnothing(saved_data)
        a = zeros(15)
        b = zeros(15)

        for _ in 1:100
            b = next_vec(15, b)
        end
    else
        a = saved_data["a_vectors"][:, 1]
        b = saved_data["b_vectors"][:, 1]
    end

    results = zeros(1 + m_purif)
    var_res = zeros(1 + m_purif)

    n = 0
    #b = 0
    x = 0
    y = 0
    all_c_values = zeros(m_purif, rep)
    all_p_values = zeros(m_purif, rep)
    all_f_values = zeros(2, m_purif, rep)

    @simd for j in 1:rep
        
        if isnothing(saved_data)
            a = next_vec(15, a)
            b = next_vec(15, b)
        else
            a = saved_data["a_vectors"][:, j]
            b = saved_data["b_vectors"][:, j]
        end

        c_a = compute_concurrence(a)
        if c_a > 1e-15
            y += 1

        end
        
        rho_a = mstate(a)

        if use_b
            rho_b = mstate(b)
        else
            rho_b = rho_a
        end

        c_purif, p_purif, fid_purif = apply_n(protocol_step, rho_a, rho_b, m_purif, operation)
        results += append!([c_a], c_purif)
        var_res += append!([c_a^2], c_purif.^2)

        all_c_values[:, j] = c_purif
        all_p_values[:, j] = p_purif
        all_f_values[:, :, j] = fid_purif
 
        if !check_ppt(mstate(flip(a)))    

            x += 1

        end

    end


    str = string(n)
    results = results/rep

    std_res = sqrt.(var_res/rep - (results).^2)

    #println("Concurrence: ", results/rep)
    #println("Standard deviation: ", std_res)
    #println("Standard deviation up: ", std_above)
    #println("Standard deviation down: ", std_below)

    return results, std_res, all_c_values, all_p_values, all_f_values
end

# Success probability of the MFI protocol.

function success_prob_mfi(state)
    r1 =  real(tr(P1 * state))
    r2 =  real(tr(P2 * state))
    r3 =  real(tr(P3 * state))
    r4 =  real(tr(P4 * state))
    r13 = tr(P13 * state)
    r24 = tr(P24 * state)
    D = (r1 + r3)^2 + (r2 + r4)^2 - (2*real(r13))^2 - (2*real(r24))^2
    return D/2
end

# Checks if the density matrix is purifiable by the MFI protocol.

function is_purifiable_mfi(state)
    r1 =  real(tr(P1 * state))
    r2 =  real(tr(P2 * state))
    r3 =  real(tr(P3 * state))
    r4 =  real(tr(P4 * state))
    r13 = tr(P13 * state)
    r24 = tr(P24 * state)

    if (2*r1 - 1)*(1 - 2*r3) > -4*imag(r13)^2 - 4*real(r24)^2 || (2*r2-1)*(1 - 2*r4) > -4*imag(r24)^2 - 4*real(r13)^2
        check=1
    else
        check=0
    end
    check
end

# Creates pairs of 4x4 density matrices starting from a dataset of N 4x4 density matrices.

function pair_up_dm(dmat_dataset)
    n = size(dmat_dataset)[2]
    dmat_pairs = zeros(4, 4, n)
    for i in 1:n
        dmat_pairs[:, :, i] = reshape(kron(dmat_dataset[:, i], dmat_dataset[:, i]), (4,4))
    end
    dmat_pairs
end

# Computes the asymptotes by checking the protocol conditions for successful purification and applies them to the dataset.

function estimate_asymptotes(n, idx, saved_data=nothing, use_b=true)

    rep = n


    if isnothing(saved_data)
        a = zeros(15)
        b = zeros(15)

        for _ in 1:100
            b = next_vec(15, b)
        end
    else
        a = saved_data["a_vectors"][:, 1]
        b = saved_data["b_vectors"][:, 1]
    end


    as_bennett = zeros(2)
    as_deutsch = zeros(2)
    as_mfi = zeros(2)
    as_cnot = zeros(2)
    x = 0

    @simd for j in 1:rep
        
        if isnothing(saved_data)
            a = next_vec(15, a)
            b = next_vec(15, b)
        else
            a = saved_data["a_vectors"][:, j]
            b = saved_data["b_vectors"][:, j]
        end 
        #print(a)
        rho_a = mstate(a)

        if use_b
            rho_b = mstate(b)
        else
            rho_b = rho_a
        end

        base = Base.OneTo(2)
        
        if is_purifiable_bennett(rho_a) > 0
            check_bennett = 1
        else
            check_bennett = 0
        end

        if is_purifiable_deutsch(rho_a) > 0
            check_deutsch = 1
        else
            check_deutsch = 0
        end

        if is_purifiable_mfi(rho_a) > 0
            check_mfi = 1
        else
            check_mfi = 0
        end

        if is_purifiable_cnot(rho_a) > 0
            check_cnot_00 = 1
        else
            check_cnot_00 = 0
        end 

        as_bennett[1] += check_bennett
        as_deutsch[1] += check_deutsch
        as_mfi[1] += check_mfi
        as_cnot[1] += check_cnot_00

        as_bennett[2] += check_bennett^2
        as_deutsch[2] += check_deutsch^2
        as_mfi[2] += check_mfi^2
        as_cnot[2] += check_cnot_00^2
         
        if !check_ppt(mstate(flip(a)))    

            x += 1

        
        end
    end

    jldopen("data_asymptotes" * "_" *  string(n) * "_" * string(use_b) * "_" * string(idx) * ".jld2", "w") do file
        file["bennett"] = as_bennett/(rep)
        file["deutsch"] = as_deutsch/(rep)
        file["mfi"] = as_mfi/(rep)
        println(as_mfi/rep)
        file["cnot"] = as_cnot/(rep)
        file["general"] = x/rep

    end

    return as_bennett/(rep), as_deutsch/(rep), as_mfi/(rep), as_cnot/(rep), x/rep
end

# Checks if the state is purifible by the cnot protocol.

function is_purifiable_cnot(state, measurement="00")
    r1 =  real(tr(P1 * state))
    r2 =  real(tr(P2 * state))
    r3 =  real(tr(P3 * state))
    r4 =  real(tr(P4 * state))
    r14 = tr(P14 * state)
    r23 = tr(P23 * state)

    if measurement == "00"
        if (2*r2 - 1)*(1 - 2*r3) > -4*imag(r14)^2 - 4*real(r23)^2 || (2*r1-1)*(1 - 2*r4) > -4*imag(r23)^2 - 4*real(r14)^2
            check=1
        else
            check=0
        end
    else
        if (2*r2 - 1)*(1 - 2*r3) > 4*imag(r14)^2 + 4*real(r23)^2 || (2*r1-1)*(1 - 2*r4) > 4*imag(r23)^2 + 4*real(r14)^2
            check=1
        else
            check=0
        end
    end
    check
end

# Checks if the state is purifiable by the Detusch protocol.

function is_purifiable_deutsch(state)

    state = convert_to_bell_diag(state)

    F1 =  real(tr(P1 * state))
    F2 =  real(tr(P2 * state))
    F3 =  real(tr(P3 * state))
    F4 =  real(tr(P4 * state))

    if F1 > 0.5 || F2 > 0.5 || F3 > 0.5 || F4 > 0.5
        check=1
    else
        check=0
    end
    check
end


# Checks if the state is purifiable by the Bennett protocol.

function is_purifiable_bennett(state)

    state = convert_to_werner(state)

    F1 =  real(tr(P1 * state))

    if F1 > 0.5
        check=1
    else
        check=0
    end

    check
end


# Runs all the purification protocols on the data.

function run_purification_protocols(x, N, purif_iter, M, idx, use_b=true, saved_data=nothing)

    res_bennett, _, bennett_cvals, bennet_probs, bennett_fidvals =
    sample_and_purify(bennett_purification_step, N, purif_iter, CNOT, use_b)
    println(res_bennett)
    println(sum(bennett_fidvals, dims=3)/N)

    res_deutsch, _, deutsch_cvals, deutsch_probs, deutsch_fidvals = 
    sample_and_purify(deutsch_purification_step, N, purif_iter, CNOT, use_b)
    println(res_deutsch)
    println(sum(deutsch_fidvals, dims=3)/N)

    res_mfi, _ , mfi_cvals, mfi_probs, mfi_fidvals = 
    sample_and_purify(mfi_purification_step, N, purif_iter, M, use_b)
    println(res_mfi)

    res_cnot, _, cnot_cvals, cnot_probs, cnot_fidvals = 
    sample_and_purify(cnot_purification_step, N, purif_iter, CNOT, use_b, saved_data)
    println(res_cnot)
    println(sum(cnot_fidvals, dims=3)/N)

    concurrence_array_data = bennett_cvals, deutsch_cvals, mfi_cvals, cnot_cvals
    prob_array_data = bennet_probs, deutsch_probs, mfi_probs, cnot_probs
    fid_array_data = bennett_fidvals, deutsch_fidvals, mfi_fidvals, cnot_fidvals

    name_arr = ["bennett", "deutsch", "mfi", "cnot"]

    jldopen("data" * "_" * "purification" * "_" * string(N) * "_" * string(idx) * "_" * string(use_b) * ".jld2", "w") do file
        for i_arr in eachindex(name_arr)
            file[name_arr[i_arr] ] = [concurrence_array_data[i_arr]/N, prob_array_data[i_arr]/N, fid_array_data[i_arr]/N]
        end
    end

    return concurrence_array_data, prob_array_data
end

    
function main()
    M = P1 + P3

    # Small test
    #dm_test = 0.2*P1 + 0.8*P3
    #rho, c, f = mfi_purification_step(dm_test, dm_test, M)
    #println(rho)
    #println(tr(rho*P1))
    #println(tr(rho*P4))
    #println(c)
    #println(f)

    N_array = [1000, ]
    println(N_array)
    cwd = pwd()

    use_b = false
    generate_matrices = false

    for N in N_array
        purif_iter = 15
        x = range(0, purif_iter)

        if generate_matrices

            for i in 1:10

                if !(isfile("random_dm_2" * "_" * string(i) * "_" * string(N) * ".csv"))
                    println("Generating data...")
                    sampled_dms_a, sampled_vectors_a = generate_and_save_dms(N)
                    if use_b
                        sampled_dms_b, sampled_vectors_b = generate_and_save_dms(N)
                    else
                        sampled_dms_b = sampled_dms_a 
                        sampled_vectors_b = sampled_vectors_a
                    end
                    println("Dm data shape:", size(sampled_dms_a))
                else
                    println("loading data: " * string(i))
                    sampled_dms_a = Matrix(load_csv_data("random_dm_2_"   * string(i) * "_" * string(N) * ".csv"))
                    sampled_vectors_a = Matrix(load_csv_data("random_a_vectors_"  * string(i) * "_" * string(N) * ".csv"))
                    if use_b
                        sampled_dms_b = Matrix(load_csv_data("random_dm_2_"   * string(i) * "_" * string(N) * ".csv"))
                        sampled_vectors_b = Matrix(load_csv_data("random_a_vectors_"   * string(i) * "_" * string(N) * "_vectors.csv"))
                    else
                        sampled_dms_b = sampled_dms_a 
                        sampled_vectors_b = sampled_vectors_a
                    end
                

                    sampled_dms_a = reshape(sampled_dms_a,  4, 4, N, 2)
                    sampled_dms_b = reshape(sampled_dms_b, 4, 4, N, 2)

                    sampled_dms_a = sampled_dms_a[:, :, :, 1] + 1im*sampled_dms_a[:, :, :, 2]
                    sampled_dms_b = sampled_dms_b[:, :, :, 1] + 1im*sampled_dms_b[:, :, :, 2]
            
                    sampled_vectors_a = reshape(sampled_vectors_a, 15, N, 2)
                    sampled_vectors_b = reshape(sampled_vectors_b, 15, N, 2)
     
            
                    sampled_vectors_a = sampled_vectors_a[:, :, 1]
                    sampled_vectors_b = sampled_vectors_b[:, :, 1]
           
                    end
            
                println(size(sampled_vectors_a))
                println(size(sampled_dms_a))
      
                saved_data = Dict("rho_vectors_a"=> sampled_dms_a, "rho_vectors_b"=> sampled_dms_b, 
                    "a_vectors"=> sampled_vectors_a, "b_vectors"=> sampled_vectors_b)

          
                asym = estimate_asymptotes(N, i, saved_data, use_b)
                println(asym)
                run_purification_protocols(x, N, purif_iter, M, i, use_b, saved_data)
    
            end

        else
            for i in 1:2
                asym = estimate_asymptotes(N, i, nothing, use_b)
                println(asym)
                run_purification_protocols(x, N, purif_iter, M, i, use_b)
            end
        end
    end
end

# Reads some data saved in CSV.

function load_csv_data(filename)
    CSV.read(filename, DataFrame, types=Float64)
end

# Generates (with hit-and-run MC) and saves density matrices in CSV.

function generate_and_save_dms(N)

    sampled_dms, sampled_vectors = sample_dms(N)
    sampled_dms_reshaped = reshape(sampled_dms, N*16)
    sampled_vectors_reshaped = reshape(sampled_vectors, N*15)
    CSV.write("random_dm_" * string(N) * ".csv", DataFrame(sampled_dms_reshaped), types=Float64)
    CSV.write("random_a_" * string(N) * "_vectors.csv", DataFrame(sampled_vectors_reshaped), types=Float64)

    sampled_dms, sampled_vectors
end

main()