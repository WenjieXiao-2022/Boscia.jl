modes = ["custom", "mip"]

for mode in modes
    @show mode
    for dimension in 3:10
        for seed in 1:5
            
            @show seed, dimension
            run(`sbatch batch_birkhoff.sh $mode $dimension $seed`)
        end 
    end
end

