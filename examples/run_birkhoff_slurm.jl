modes = ["custom", "mip"]

for mode in modes
    for dimension in 15:30
        for seed in seeds
            
            @show seed, dimension
            run(`sbatch batch_birkhoff.sh $mode $dimension $seed`)
        end 
    end
end

