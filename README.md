# tvPersistence.jl

The code has been developed in Julia as a code accompanying the Barunik and Vacha (2023) and Barunik and Vacha (2024) papers, and provides estimation of time-varying persistence using *localised heterogeneous persistence*:

Baruník, J. and Vacha, L. (2023): *The Dynamic Persistence of Economic Shocks*, manuscript [available here for download](https://ideas.repec.org/p/arx/papers/2306.01511.html)

Baruník, J. and Vacha, L. (2024): *Forecasting Volatility of Oil-based Commodities: The Model of Dynamic Persistence*, manuscript [available here for download](https://arxiv.org/pdf/2402.01354.pdf)

## Software requirements

Install [Julia](http://julialang.org/) version 1.6.0 or newer and with the first use of this code install the same version of packages with which the projects is built and work in the environment of the project as

```julia
using Pkg
Pkg.activate(".") # activating project in its directory
Pkg.instantiate() # installing packages with which versions the project is built
```

## Example of usage

This example illustrates how to obtain decomposition of dynamic persistence, as well as forecasting model.

Load Packages


```julia
using RCall
R"library(tvReg)"

using CSV, DataFrames,GLM
using Distributions, LinearAlgebra, Statistics
using Plots, StatsBase, StatsPlots, Colors

# load main functions
include("tvPersistence_functions.jl");
myrainbow=reverse(cgrad(:RdYlBu_7, 7, categorical = true));
```

Load example data


```julia
data_read=CSV.File("data.csv",missingstring=["NA"],header=true) |> DataFrame;
dat0=data_read.CL[ismissing.(data_read.CL).==false];
```

### Part 1: Plot the decomposition

Function tvPersistenceplot plots dynamic persistence decomposition

````julia
decomp_matrix = tvPersistenceplot(data,maxAR,J,kernel_1,kernel_2)

# INPUTS:  data        Data vector of length T
#          maxAR       number of the AR lags in the persistence estimation
#          J,          Depth of the persistence decomposition
#          kernel_1,   Bandwidth of the kernel used for the constant estimation
#          kernel_2,   Bandwidth of the kernel used for the TVP IRF estimation
#
# OUTPUTS:  decomp_matrix           J x T matrix
````


```julia
decomp = tvPersistenceplot(dat0,5,7,0.15,0.05);
```

Example of nice plot


```julia
yearfirstb=decomp./sum(decomp,dims=2);

plot(1:size(yearfirstb,1),yearfirstb,size=(700,700/1.6666),color=[myrainbow[1] myrainbow[2] myrainbow[3] cgrad(:grayC, 7, categorical = true)[2] myrainbow[5] myrainbow[6] myrainbow[7]],frame=:box,
    linestyle=:dot,linealpha=0.7,label=false,legend=:topleft,yaxis="CL",
    xticks=([1,517,1032,1550,2065,2581,3099],["2010","2012","2014","2016","2018","2020","2022"])) 
scatter!(1:12:size(yearfirstb,1),yearfirstb[1:12:size(yearfirstb,1),:],color=[myrainbow[1] myrainbow[2] myrainbow[3] cgrad(:grayC, 7, categorical = true)[2] myrainbow[5] myrainbow[6] myrainbow[7]],
    label=["2 days" "4" "8" "16" "32" "64" "128+"],msc=:white,markersize=3,markershape=[:circle :diamond :utriangle :+ :x :heptagon :dtriangle])
```




![svg](/readme_files/output_10_0.svg)



Export the plot to pdf


```julia
savefig("figure_decomposition.pdf")
```

### Part 2: Forecasting model


Function tvEWDforecast returns forecasts

````julia
forecast, actual, error = tvEWDforecast(data,Tins,horizon,maxAR,J,kernel_1,kernel_2,kernel_3);

# INPUTS:  data        Data vector of length T
#          Tins        In-sample length (Tins<T)   
#          maxAR       number of the AR lags in the persistence estimation
#          J,          Depth of the persistence decomposition
#          horizon     forecast horizon
#          kernel_1,   Bandwidth of the kernel used for the constant estimation
#          kernel_2,   Bandwidth of the kernel used for the TVP IRF estimation
#          kernel_3,   Bandwidth of the kernel used for forecasting of the constant
#
# OUTPUTS:  forecast   forecasts of actual data
#           actual     forecasted (actual) data
#           error      error = forecast - actual
````

Example of h=1 step ahead forecast


```julia
data0=data_read.CL[ismissing.(data_read.CL).==false][1:800];
```


```julia
forecast, actual, error = tvEWDforecast(data0,700,1,2,7,0.1,0.3,0.1);
```


```julia
plot([actual forecast], label=["Data" "Forecast"],frame=:box)
```




![svg](/readme_files/output_19_0.svg)


