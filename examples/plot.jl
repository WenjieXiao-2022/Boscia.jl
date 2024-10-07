using PyPlot
using DataFrames
using CSV

function plot_term(;by_time=true, log_scale=true)
    file_name = ""

    fs = 14

    if by_time
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/birkhoff_comparison_non_grouped.csv")))

        #fig = plt.figure(figsize=(6.5,4.5)) 
        fig, ax = plt.subplots(1, sharex=true, sharey=false, figsize=(6.5,4.5))
        colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
        markers = ["o", "s", "^", "P", "X", "H", "D"]
        linestyle = ["-", ":", "-.", "--"]

        PyPlot.matplotlib[:rc]("text", usetex=true)
        PyPlot.matplotlib[:rc]("font", size=fs, family="cursive")
        PyPlot.matplotlib[:rc]("axes", labelsize=fs)
        PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
        \usepackage{libertine}
        \usepackage{libertinust1math}
        """)
        #ax = fig.add_subplot(111)

        df_hun = deepcopy(df)
        filter!(row -> !(row.terminationBoscia_Custom == 0),  df_hun)
        x_hun = sort(df_hun[!,"timeBoscia_Custom"]) 
        #ax.plot(x_hun, 1:nrow(df_hun), label="Hungarian", color=colors[2], linestyle=linestyle[1], marker=markers[1])
        ax.plot(x_hun, 1:nrow(df_hun), label="Boscia + Hungarian", color=colors[2], linestyle=linestyle[1], marker=markers[1])

        df_mip = deepcopy(df)
        filter!(row -> !(row.terminationBoscia_MIP == 0),  df_mip)
        x_mip = sort(df_mip[!,"timeBoscia_MIP"]) 
        #ax.plot(x_mip, 1:nrow(df_mip), label="MIP SCIP", color=colors[6], linestyle=linestyle[3], marker=markers[3])
        ax.plot(x_mip, 1:nrow(df_mip), label="Boscia + MIP SCIP", color=colors[6], linestyle=linestyle[3], marker=markers[3])

        ax.grid()
        log_scale ? ax.set_xscale("log") : nothing

        ax.legend(loc="lower right")#, bbox_to_anchor=(0.5, -0.3), fontsize=12,fancybox=true, shadow=false, ncol=2) 

        ylabel("Solved to optimality", fontsize=fs)
        xlabel("Time (s)", fontsize=fs)

        fig.tight_layout()

        file_name = joinpath(@__DIR__, "csv/plot_birkhoff_termination.pdf")
    else
        df = DataFrame(CSV.File(joinpath(@__DIR__, "csv/birkhoff_comparison_summary_by_dimension.csv")))

        fig, axs = plt.subplots(2, sharex=true, sharey=false, figsize=(6.5,5.5))

        colors = ["b", "m", "c", "r", "g", "y", "k", "peru"]
        markers = ["o", "s", "^", "P", "X", "H", "D"]
        linestyle = ["-", ":", "-.", "--"]

        PyPlot.matplotlib[:rc]("text", usetex=true)
        PyPlot.matplotlib[:rc]("font", size=11, family="cursive")
        PyPlot.matplotlib[:rc]("axes", labelsize=14)
        PyPlot.matplotlib[:rc]("text.latex", preamble=raw"""
        \usepackage{libertine}
        \usepackage{libertinust1math}
        """)

        linewidth = 2

        axs[1].plot(df[!,"Dimension"], df[!,"Boscia_CustomTerm"], label="Hungarian", color=colors[2], linestyle=linestyle[1], marker=markers[1])
        axs[1].plot(df[!,"Dimension"], df[!,"Boscia_MIPTerm"], label="MIP SCIP", color=colors[6], linestyle=linestyle[3], marker=markers[3])

        axs[1].grid()

        axs[2].plot(df[!,"Dimension"], df[!,"Boscia_CustomTime"], label="Hungarian", color=colors[2], linestyle=linestyle[1], marker=markers[1])
        axs[2].plot(df[!,"Dimension"], df[!,"Boscia_MIPTime"], label="MIP SCIP", color=colors[6], linestyle=linestyle[3], marker=markers[3])

        axs[2].grid()
        axs[2].legend(loc="lower right")#, bbox_to_anchor=(0.5, -0.3), fontsize=12,fancybox=true, shadow=false, ncol=2) 

        axs[1].set_ylabel("Solved to optimality", loc="center")
        axs[2].set_ylabel("Average time", loc="center")
        xlabel("Dimension n")

        fig.tight_layout()

        file_name = joinpath(@__DIR__, "csv/plot_birkhoff_by_dimension.pdf")
    end

    PyPlot.savefig(file_name)
end

#plot_term()