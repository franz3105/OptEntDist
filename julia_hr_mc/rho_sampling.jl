using LinearAlgebra
using BlockArrays, Random, SparseArrays
using Plots
using DataFrames, CSV

import Random

# Seeds the simulation
Random.seed!(1234)

function random_vector(n)

    v= randn(n)

    while norm(v) < 1e-9

        v = randn(n)  # random standard normal distribution

    end

    return v / norm(v)  

end

#full state Newton conditions

function newr2(a) 

    3/4 - a[1]^2 - a[10]^2 - a[11]^2 - a[12]^2 - a[13]^2 - a[14]^2 - a[15]^2 - 

    a[2]^2 - a[3]^2 - a[4]^2 - a[5]^2 - a[6]^2 - a[7]^2 - a[8]^2 - a[9]^2 >= 0 

end

    

function newr3(a)

    return (-4*a[1]^2 - 4*a[10]^2 - 4*a[11]^2 - 4*a[12]^2 - 4*a[13]^2 - 4*a[14]^2 - 4*a[15]^2 - 4*a[2]^2 

    - 4*a[3]^2 + 16*(a[10]*a[2] + a[13]*a[3])*a[4] - 4*a[4]^2 + 16*(a[11]*a[2] + a[14]*a[3])*a[5] - 4*a[5]^2 

    + 16*(a[12]*a[2] + a[15]*a[3])*a[6] - 4*a[6]^2 + 16*(a[12]*a[14] - a[11]*a[15] + a[1]*a[4])*a[7] 

    - 4*a[7]^2 - 16*(a[12]*a[13] - a[10]*a[15] - a[1]*a[5])*a[8] - 4*a[8]^2 + 16*(a[11]*a[13] - a[10]*a[14] 

    + a[1]*a[6])*a[9] - 4*a[9]^2 + 1 >= 0 )

end

    

function newr4(a)

    return (16*a[1]^4 + 16*a[10]^4 + 16*a[11]^4 + 16*a[12]^4 + 16*a[13]^4 + 128*a[10]*a[11]*a[13]*a[14] 

    + 16*a[14]^4 + 16*a[15]^4 + 16*a[2]^4 + 16*a[3]^4 + 16*a[4]^4 + 16*a[5]^4 + 16*a[6]^4 + 16*a[7]^4

    + 16*a[8]^4 + 16*a[9]^4 + 8*(4*a[1]^2 - 1)*a[10]^2 + 8*(4*a[1]^2 + 4*a[10]^2 - 1)*a[11]^2 

    + 8*(4*a[1]^2 + 4*a[10]^2 + 4*a[11]^2 - 1)*a[12]^2 + 8*(4*a[1]^2 + 4*a[10]^2 - 4*a[11]^2 

    - 4*a[12]^2 - 1)*a[13]^2 + 8*(4*a[1]^2 - 4*a[10]^2 + 4*a[11]^2 - 4*a[12]^2 + 4*a[13]^2 - 1)*a[14]^2 

    + 8*(4*a[1]^2 - 4*a[10]^2 - 4*a[11]^2 + 4*a[12]^2 + 4*a[13]^2 + 4*a[14]^2 - 1)*a[15]^2 + 8*(4*a[1]^2 

    - 4*a[10]^2 - 4*a[11]^2 - 4*a[12]^2 + 4*a[13]^2 + 4*a[14]^2 + 4*a[15]^2 - 1)*a[2]^2 - 128*(a[10]*a[13]

    + a[11]*a[14] + a[12]*a[15])*a[2]*a[3] + 8*(4*a[1]^2 + 4*a[10]^2 + 4*a[11]^2 + 4*a[12]^2 - 4*a[13]^2

    - 4*a[14]^2 - 4*a[15]^2 + 4*a[2]^2 - 1)*a[3]^2 - 8*(4*a[1]^2 + 4*a[10]^2 - 4*a[11]^2 - 4*a[12]^2 

    + 4*a[13]^2 - 4*a[14]^2 - 4*a[15]^2 + 4*a[2]^2 + 4*a[3]^2 + 1)*a[4]^2 - 8*(4*a[1]^2 - 4*a[10]^2

    + 4*a[11]^2 - 4*a[12]^2 - 4*a[13]^2 + 4*a[14]^2 - 4*a[15]^2 + 4*a[2]^2 + 4*a[3]^2 - 4*a[4]^2 

    + 1)*a[5]^2 - 8*(4*a[1]^2 - 4*a[10]^2 - 4*a[11]^2 + 4*a[12]^2 - 4*a[13]^2 - 4*a[14]^2 + 4*a[15]^2 

    + 4*a[2]^2 + 4*a[3]^2 - 4*a[4]^2 - 4*a[5]^2 + 1)*a[6]^2 - 8*(4*a[1]^2 - 4*a[10]^2 + 4*a[11]^2 

    + 4*a[12]^2 - 4*a[13]^2 + 4*a[14]^2 + 4*a[15]^2 - 4*a[2]^2 - 4*a[3]^2 + 4*a[4]^2 - 4*a[5]^2 

    - 4*a[6]^2 + 1)*a[7]^2 - 8*(4*a[1]^2 + 4*a[10]^2 - 4*a[11]^2 + 4*a[12]^2 + 4*a[13]^2 - 4*a[14]^2 

    + 4*a[15]^2 - 4*a[2]^2 - 4*a[3]^2 - 4*a[4]^2 + 4*a[5]^2 - 4*a[6]^2 - 4*a[7]^2 + 1)*a[8]^2 

    - 8*(4*a[1]^2 + 4*a[10]^2 + 4*a[11]^2 - 4*a[12]^2 + 4*a[13]^2 + 4*a[14]^2 - 4*a[15]^2 - 4*a[2]^2 

    - 4*a[3]^2 - 4*a[4]^2 - 4*a[5]^2 + 4*a[6]^2 - 4*a[7]^2 - 4*a[8]^2 + 1)*a[9]^2 - 8*a[1]^2 

    + 128*(a[10]*a[12]*a[13] + a[11]*a[12]*a[14])*a[15] - 64*(2*a[1]*a[12]*a[14] - 2*a[1]*a[11]*a[15] 

    - a[10]*a[2] - a[13]*a[3])*a[4] + 64*(2*a[1]*a[12]*a[13] - 2*a[1]*a[10]*a[15] + a[11]*a[2] 

    + a[14]*a[3] - 2*(a[10]*a[11] + a[13]*a[14])*a[4])*a[5] - 64*(2*a[1]*a[11]*a[13] - 2*a[1]*a[10]*a[14] 

    - a[12]*a[2] - a[15]*a[3] + 2*(a[10]*a[12] + a[13]*a[15])*a[4] + 2*(a[11]*a[12] 

    + a[14]*a[15])*a[5])*a[6] - 64*(2*a[1]*a[10]*a[2] + 2*a[1]*a[13]*a[3] - a[12]*a[14] + a[11]*a[15] 

    - a[1]*a[4] - 2*(a[15]*a[2] - a[12]*a[3])*a[5] + 2*(a[14]*a[2] - a[11]*a[3])*a[6])*a[7] 

    - 64*(2*a[1]*a[11]*a[2] + 2*a[1]*a[14]*a[3] + a[12]*a[13] - a[10]*a[15] + 2*(a[15]*a[2] 

    - a[12]*a[3])*a[4] - a[1]*a[5] - 2*(a[13]*a[2] - a[10]*a[3])*a[6] - 2*(a[10]*a[11] + a[13]*a[14] 

    - a[4]*a[5])*a[7])*a[8] - 64*(2*a[1]*a[12]*a[2] + 2*a[1]*a[15]*a[3] - a[11]*a[13] + a[10]*a[14] 

    - 2*(a[14]*a[2] - a[11]*a[3])*a[4] + 2*(a[13]*a[2] - a[10]*a[3])*a[5] - a[1]*a[6] - 2*(a[10]*a[12] 

    + a[13]*a[15] - a[4]*a[6])*a[7] - 2*(a[11]*a[12] + a[14]*a[15] - a[5]*a[6])*a[8])*a[9] + 1  >= 0 )

end

#Check if vector a fulfills the Newton conditions

function checknewt(a)

    return (newr2(a) && newr3(a) && newr4(a))

end 

#flip signs for PPT condition

function flip(a)

    return [a[i]*(-1)^(i in [5,8,11,14]) for i in 1:15 ]

end

#get the next state

function next_vec(d, current)

    dir = random_vector(d)

    vmax =  2 * sqrt(3 / 4)

    vmin = -2 * sqrt(3 / 4)

    v = vmin + (vmax-vmin)*rand()

    a = current + v * dir

    while !checknewt(a)

        (v < 0) ? (vmin = v) : (vmax = v)

        v = vmin + (vmax - vmin) * rand()

        a = current + v * dir    

    end

    return a

end

##concurrence##

#Basis d=2

s2 = Array{Complex{Float64}}(undef, 2, 2, 4)

s2[:,:,1] = [1 0; 0 1]/sqrt(2)

s2[:,:,2] = [0 1; 1 0]/sqrt(2)

s2[:,:,3] = [0 -1im; 1im 0]/sqrt(2)

s2[:,:,4] = [1 0; 0 -1]/sqrt(2)

#Basis 2x2

k16 = Array{Complex{Float64}}(undef, 4, 4, 16)

for i in 0:3, j in 1:4

    k16[:,:,i*4+j] =  kron(s2[:,:,i+1],s2[:,:,j])

end

#generate density matrix from vector a

#mul!(C,A,B,x,y)::  C := x*A*B + y*C

function mstate_a(a)

    state = Array{Complex{Float64}}(I/4,4,4)

    for i in 1:15

        mul!(state, a[i],  k16[:,:,i+1], 1, 1)

    end

    return state

end

#order the elements of a to fit the basis

function mstate(a)

    return mstate_a([a[4] a[5] a[6] a[1] a[7] a[8] a[9] a[2] a[10] a[11] a[12] a[3] a[13] a[14] a[15]])

end

#compute the concurrence of a state

#compute the partial transpose for the state

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

# Check that the ppt condition is fulfilled

function check_ppt(rho)
    minimum(eigvals(rho))>=0
    
end

# Computes the concurrence of the two-qubit density matrix starting from the vector a.

function compute_concurrence(a)

    s_y = 2*k16[:,:,11]

    rho = mstate(a)

    rho_tilde = s_y*(conj.(rho))*s_y

    #lambda1 = real(eigvals(sqrt(sqrt(rho)*rho_tilde*sqrt(rho))))

    #println(lambda1)
    #println(real(eigvals(rho*rho_tilde)))
    lambda = sqrt.(real(eigvals(rho*rho_tilde)) .+ 1e-15)

    return max(0, 2*lambda[4] - sum(lambda))

end

# Computes the concurrence of the two-qubit density matrix.

function compute_concurrence_rho(rho)

    s_y = 2*k16[:,:,11]

    rho_tilde = s_y*(conj.(rho))*s_y

    lambda = sqrt.(real(eigvals(rho*rho_tilde)) .+ 1e-15)

    return max(0, 2*lambda[4] - sum(lambda))

end

# Samples values of the concurrence using the hit-and-run Monte Carlo method.

function sample_concurrence(n=1000)

    a = zeros(15)

    results = zeros(1)
    all_c_values = zeros(n)

    @time for j in 1:n

        a = next_vec(15, a)

        results[1] += compute_concurrence(a)
        all_c_values[j] = compute_concurrence(a)

    end

    str = string(n)

    for i in 1:length(results)

        str *= string(";  ", results[i]/n)

    end
    println(str)
    all_c_values, results

end

# Samples the two-qubit density matrix using the hit-and-run Monte Carlo method.

function sample_dms(N)
    a = zeros(15)
    rhos = Array{Complex{Float64}}(undef, 16, N)
    a_vectors = Array{Complex{Float64}}(undef, 15, N)
    for j in 1:N
        a = next_vec(15, a)
        rhos[:, j] = reshape(mstate(a), 16)
        a_vectors[:, j] = a
    end
    rhos, a_vectors
end

function sample_concurrence_inf(N)

    #states generated between intermediate results

    rep = N

    a = zeros(15)
    all_c_values = zeros(N)
    all_f_values = zeros(N)

    results = zeros(1)

    n = 0
    b = 0
    x = 0
    y = 0

    while true

        

        for j in 1:rep

            a = next_vec(15, a)
            c_a = compute_concurrence(a)
            fid = fidelity_psi_min(mstate(a))

            all_c_values[j] = c_a
            all_f_values[j] = fid
            results[1] += c_a

            #if !check_ppt(partial_transpose_(mstate(a), 2, 2))
            if !check_ppt(mstate(flip(a)))    

                x += 1

            end

            if c_a > 1e-15
                y += 1

            end

        end

    
        n += rep


        str = string(n)

        for i in 1:length(results)

            str *= string(";  ", results[i]/n)

        end

        println(str)
        println(x)
        println(y)

        
    end
    
all_c_values, results
end


function plot_concurrence_distribution(N=1)

    data, data_avg = sample_concurrence(N)

    data, data_avg
end

function generate_dm_dataset()
    N = 10^6
    @simd for i in 1:1
        println("Iteration: ", i)
        sampled_dms, sampled_vectors = sample_dms(N)
        sampled_dms = reshape(sampled_dms, N*16)
        sampled_vectors = reshape(sampled_vectors, N*15)
        CSV.write("data/random_dm_2_" * string(i) * "_" * string(N) * ".csv", 
        DataFrame(sampled_dms), types=Float64)
        CSV.write("data/random_a_vectors_" * string(i) * "_" * string(N) * ".csv",
        DataFrame(sampled_vectors), types=Float64)
    end
end

#generate_dm_dataset()

#N = 1000000
#data, _ = plot_concurrence_distribution(N)
#data_filtered = data[data .> 10^-15]
#sampled_dms, sampled_vectors = sample_dms(N)
#sampled_dms = reshape(sampled_dms, N*16)
#sampled_vectors = reshape(sampled_vectors, N*15)
#CSV.write("random_dm_2.csv", DataFrame(sampled_dms), types=Float64)
#CSV.write("random_a_vectors.csv", DataFrame(sampled_dms), types=Float64)
#time and test 

#test(100000)

#run forever, stop with ctrl+c

#inf_run()

