modes = ["custom", "mip"]

for mode in modes
    @show mode
    for dimension in 3:15
        for seed in 1:10
            
            @show seed, dimension
            run(`sbatch batch_birkhoff.sh $mode $dimension $seed`)
        end 
    end
end

