# Dump reproducibility info (Julia/BEAST versions, CPU, RAM, threads)
# to data/versioninfo.txt for the manuscript's reproducibility statement.
import Pkg
Pkg.activate((@__DIR__) * "/..")

using InteractiveUtils
using LinearAlgebra

open(joinpath(@__DIR__, "..", "data", "versioninfo.txt"), "w") do io
    versioninfo(io)
    println(io, "CPU model: ", Sys.cpu_info()[1].model)
    println(io, "CPU threads: ", Sys.CPU_THREADS)
    println(io, "RAM GiB: ", round(Sys.total_memory()/2^30, digits=1))
    println(io, "Julia threads: ", Threads.nthreads())
    println(io, "BLAS threads: ", LinearAlgebra.BLAS.get_num_threads())
    for (uuid, dep) in Pkg.dependencies()
        dep.name in ("BEAST", "CompScienceMeshes", "Makeitso", "DrWatson") &&
            println(io, dep.name, " v", dep.version)
    end
end
println("[saved] data/versioninfo.txt")
