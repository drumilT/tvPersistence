
#**************************************************************************
# by Jozef Barunik
#**************************************************************************


function OLSestimator(y,x)
    return (transpose(x)*x) \ (transpose(x)*y)
end

function OLSestimatorconst(y,x)
    x=[ones(size(x)[1]) x]
    return (transpose(x)*x) \ (transpose(x)*y)
end


function IRFalpha(y,x,maxAR,M)

    b=OLSestimator(y,x)
    Eta=y-x*b;
    sigma2=(Eta'*Eta)./(length(y)-maxAR)
    sigma=sqrt.(sigma2)
    Eps=Eta./sigma;

    alphaR=zeros(M)
    alphaR[1]=sigma

    for n=1:(length(alphaR)-1) 
        hstart=max(n-maxAR,0);
        temp=0;
        for h=hstart:n-1 
            temp=temp+alphaR[h+1]*b[n-h]; 
        end
        alphaR[n+1]=temp;
    end
    return (alphaR,Eps)
end

function ARlags(X, p)
    #AR regression to estimate the AR coefficients
    y=X[1:end-p]
    xx = zeros(length(y),p)
    for i=1:length(y)
        xx[i,:]=X[i+1:i+p]; #fill the matrix with lags
    end
    return (y,xx)
end


function IRFalpha_tvp_LLS(y,x,maxAR,M,kernel_width)
    
    # New localized linear LS estimate
    R"aa <- tvOLS($x,$y, bw = $kernel_width)$coefficients"
    b_tvp_all=convert(Array{Float64},R"aa")

    Eta_tvp=zeros(length(y))
    for i=1:length(y)
        Eta_tvp[i] = y[i]-sum(b_tvp_all[i,:].*x[i,:]);
    end

    sigma2_tvp=(Eta_tvp'*Eta_tvp)./(length(y)-maxAR)
    sigma_tvp=sqrt.(sigma2_tvp)
    Eps_tvp=Eta_tvp./sigma_tvp;


    alphaR_tvp=zeros(M);
    alphaR_tvp[1]=sigma_tvp;

    for n=1:(length(alphaR_tvp)-1) 
        hstart=max(n-maxAR,0);
        temp=0;
        for h=hstart:n-1 
            temp=temp+alphaR_tvp[h+1]*b_tvp_all[1,n-h];    # choses last position of kernel!!!
        end
        alphaR_tvp[n+1]=temp;
    end

    return (alphaR_tvp,Eps_tvp)
end

function IRFalpha_tvp_LLS_REVERSE(y,x,maxAR,M,kernel_width)
    
    # New localized linear LS estimate
    R"aa <- tvOLS($x,$y, bw = $kernel_width)$coefficients"
    b_tvp_all=convert(Array{Float64},R"aa")

    Eta_tvp=zeros(length(y))
    for i=1:length(y)
        Eta_tvp[i] = y[i]-sum(b_tvp_all[i,:].*x[i,:]);
    end

    sigma2_tvp=(Eta_tvp'*Eta_tvp)./(length(y)-maxAR)
    sigma_tvp=sqrt.(sigma2_tvp)
    Eps_tvp=Eta_tvp./sigma_tvp;

    alphaR_tvp_1=[]
    for i=1:size(b_tvp_all,1)
        
        alphaR_tvp=zeros(M);
        alphaR_tvp[1]=sigma_tvp;

        for n=1:(length(alphaR_tvp)-1) 
            hstart=max(n-maxAR,0);
            temp=0;
            for h=hstart:n-1 
                #temp=temp+alphaR_tvp[h+1]*b_tvp_all[end,n-h];
                temp=temp+alphaR_tvp[h+1]*b_tvp_all[i,n-h];    # choses last position of kernel!!!
            end
            alphaR_tvp[n+1]=temp;
        end
        push!(alphaR_tvp_1,alphaR_tvp)
    end

    return (alphaR_tvp_1,Eps_tvp)
end


function normker(T, H)
  ww = zeros(T, T)
  for j in 1:T
    for i in 1:T
      z = (i - j)/H
      ww[i,j] = (1.0 / sqrt(2.0 * pi)) * exp((-1.0 / 2.0) * (z * z))  # Normal
      #ww[i,j] = (2.0 / pi ) * (1.0 / (exp(z) + exp(-z))) # sigmoid
      #ww[i,j] = 1.0 / (exp(z)+2.0+exp(-z)) # Logistic
      #ww[i,j] = 1.0 - abs(z)  # traingular
      #ww[i,j] = (3.0/4.0)*(1.0-(z * z)) # epan
    end     
  end 

  s = sum(ww, dims=2)
  adjw = zeros(T, T)  

  for k in 1:T
    adjw[k, :] = ww[k, :] / s[k]
  end

  cons = sum(adjw .^ 2.0, dims = 2)

  for k in 1:T
    adjw[k, :] = (1.0 / cons[k]) * (adjw[k, :])
  end

  return adjw
end


function IRFscale(T,maxAR,alpha0,Eps,KMAX,J)
    # input:  vector alpha of classical Wold innovations
    #         with length 2^JMAX * constant
    #         T sample length
    #         maxAR max lag in the baseline AR
    #         Eps vector of unit variance classical Wold innovations in reverse order
    #         KMAX=2^(JMAX+3) maximum lag on scales
    #         J scale
    # output: vector betaScale of multiscale IRF at scale J with length length(alpha)/(2^J)
    #         vector EpsScale of details at scale J in reverse order with length T-maxAR-2^J+1
    #         vector gScale of component at scale J in reverse order with length T-maxAR-KMAX+1
    #         vector chronGScale is gScale in chronological order
    #         vector decimGScale of decimated component at scale J in reverse order with length floor(length(gScale)/2^J)

    # all processes have ZERO MEAN
   
    M=length(alpha0);
    betaScale=zeros(Int(M./(2.0.^J)));
    for k=0:Int(floor(M/(2^J))-1)
        betaScale[k+1]=(sum(alpha0[k*2^J+1:k*2^J+2^(J-1)]) - sum(alpha0[k*2^J+2^(J-1)+1:k*2^J+2^J]))./sqrt(2^J);
    end
    
    EpsScale=zeros(T-maxAR-2^J+1);
    for t=0:1:(length(EpsScale)-1)
        EpsScale[t+1]= (sum(Eps[t+1:t+2^(J-1)])-sum(Eps[t+2^(J-1)+1:t+2^J]))./sqrt(2^J);
    end
    
    gScale=zeros(T-maxAR-KMAX+1); 
    for t=0:1:(T-maxAR-KMAX)
            for k=0:1:(Int(KMAX/(2^J)-1))
                gScale[t+1]+=betaScale[k+1].*EpsScale[t+k*2^J+1]
            end
    end 
    chronGScale=reverse(gScale);
    
    #decimGScale=zeros(Int(floor(length(gScale)/2^J)));
    #for i=1:1:length(gScale)
    #    if mod(i-1,2^J)==0
    #        decimGScale[Int((i-1)/2^J+1)]=gScale[i];
    #    end
    #end
    
    decimGScale=[];
    for i=1:1:length(gScale)
        if mod(i-1,2^J)==0
            push!(decimGScale,gScale[i]);
        end
    end
    
    return (betaScale,EpsScale,gScale,chronGScale,decimGScale)
end


function IRFforecast_horizon(T,maxAR,alpha0,Eps,KMAX,J,horizon)
    
    # input:  vector alpha of classical Wold innovations
    #         with length 2^JMAX * constant
    #         T sample length
    #         maxAR max lag in the baseline AR
    #         Eps vector of unit variance classical Wold innovations in reverse order
    #         KMAX=2^(JMAX+3) maximum lag on scales
    #         J scale
    #         horizon max lag in forecasts
    # output: matrix betaPlus of multiscale IRF Beta k,p at scale J with length length(alpha)/(2^J), p goes from 1 to horizon 
    #         (see Appendix of Ortu Severino Tamoni Tebaldi)
    #         vector gScale of forecast for the sum of following week values of scale J in reverse order with length T-maxAR-KMAX+1

    # all processes have ZERO MEAN
   
    M=length(alpha0);
    betaPlus=zeros(Int(floor((M-horizon)/(2^J)))-1,horizon); #collects betak,p
    for p=1:horizon #p are the steps ahead as in Appendix of Ortu Severino Tamoni Tebaldi
        for k=0:(Int(floor((M-horizon)/(2^J)))-2)
            betaPlus[k+1,p]=(sum(alpha0[k*2^J+1+p:k*2^J+2^(J-1)+p])-sum(alpha0[k*2^J+2^(J-1)+1+p:k*2^J+2^J+p]))/sqrt(2^J);
        end                     
    end
    
    EpsScale=zeros(T-maxAR-2^J+1);
    for t=0:1:(length(EpsScale)-1)
        EpsScale[t+1]= (sum(Eps[t+1:t+2^(J-1)])-sum(Eps[t+2^(J-1)+1:t+2^J]))./sqrt(2^J);
    end
    
    gScale=zeros(T-maxAR-KMAX+1,horizon); #now gscale has to contain all the p step ahead forecasts
    for p=1:horizon
        for t=0:1:(T-maxAR-KMAX)    
            for k=0:1:(Int(floor((KMAX-horizon)/(2^J)))-2)
                gScale[t+1,p]+=betaPlus[k+1,p]*EpsScale[t+k*2^J+1];
            end
        end
    end 
     gScale=sum(gScale,dims=2) #we make row-wise sums
    
    return (betaPlus, gScale)
end






function EWD_tvLS_parallel(ii,data0,tt,maxAR,JMAX,horizon,kernel_width_for_const,kernel_width_IRF,kernel_width_forecast,AR_lag_forecast)
    
    T=length(data0)
    muR=mean(data0)
    chronR=data0.-muR;
    r=reverse(chronR);

    fcast_length = T-tt-21-horizon; #length of forecast sample

    RVh = zeros(length(r)-horizon+1,1);
    for i=1:length(RVh)
        RVh[i]=mean(r[i:i+horizon-1])
    end

    M=Int.(2^(JMAX)*(floor((tt-maxAR)/(2^(JMAX)))-1))
    KMAX = M

#     horizon_forecast_Jcomp_tvp= zeros(fcast_length);
#     Error_Jcomp_tvp=zeros(fcast_length);
#     errs0_last=zeros(fcast_length);
#     forecasted_constant=zeros(fcast_length);
#     variance_scales=zeros(fcast_length,JMAX);
#     first_forecasted_beta_scales=zeros(fcast_length,JMAX);
#     vv =zeros(JMAX);

#     for ii=0:(fcast_length-1)
        est_sample_tvp = r[(fcast_length+horizon - ii): (fcast_length+tt+horizon-1-ii)]
        
        # new trend estimation
        x=ones(length(est_sample_tvp))
        R"aa <- tvOLS(as.matrix($x),$est_sample_tvp, bw = $kernel_width_for_const)$coefficients"
        tvp_const=convert(Array{Float64},R"aa")
        errs0=est_sample_tvp.-tvp_const
    
        
        (truncR,Rmat)=ARlags(errs0, maxAR);
        (alphaR_tvp,Eps_tvp) = IRFalpha_tvp_LLS(truncR,Rmat,maxAR,M,kernel_width_IRF)


        # Estimate the decomposition
        betaRj_tvp=[]
        rj_tvp=[]
        for j in 1:JMAX
            (betaR, _ , r0, _ , _)=IRFscale(tt,maxAR,alphaR_tvp,Eps_tvp,KMAX,j);
            push!(betaRj_tvp,betaR)
            push!(rj_tvp,r0)
        end

        
        ################# Estimate coefficients of regression model ################
        b_Jcomp_tvp=OLSestimator(est_sample_tvp[1:length(rj_tvp[JMAX])],[tvp_const[1:length(rj_tvp[JMAX])] hcat(rj_tvp...)])
   
        forecasts_ar =[]
        for_ar=[]
        rev=vec(reverse(tvp_const));

        ############# Forecast tvp_const that does not enter EWD ...############
        R"forecasts_ar<- tvReg::forecast(tvAR($rev,p=$AR_lag_forecast,bw = $kernel_width_forecast),n.ahead=$horizon)" 
    
        for_ar=convert(Array{Float64},R"forecasts_ar")  
        for_const = mean(for_ar)
    
        
        ######################## EWD FORECAST #####################
        
        # We forecast the sum horizon-step ahead 
        betaFj_tvp=[]
        rfj_tvp=[]
        for j in 1:JMAX
            (betaF,rf)=IRFforecast_horizon(tt,maxAR,alphaR_tvp,Eps_tvp,KMAX,j,horizon);
            push!(betaFj_tvp,betaF)
            push!(rfj_tvp,rf)
        end
    
        ######### forecasted constant x EWD forecast ##############
    
        RFmat_Jcomp_tvp=[for_const.*ones(size(hcat(rfj_tvp...))[1]) hcat(rfj_tvp...)]; # with const pred
        
        horizon_forecast_Jcomp_tvp=(horizon^(-1).*RFmat_Jcomp_tvp[1,:])'*[horizon*b_Jcomp_tvp[1]; b_Jcomp_tvp[2:end]];
        
        # error with J components
        Error_Jcomp_tvp=horizon_forecast_Jcomp_tvp-RVh[fcast_length-ii]

#     end
    return [horizon_forecast_Jcomp_tvp;Error_Jcomp_tvp]
end

function EWD_tvLS_parallel_Epa(ii,data0,tt,maxAR,JMAX,horizon,kernel_width_for_const,kernel_width_IRF,kernel_width_forecast,AR_lag_forecast)
    
    T=length(data0)
    muR=mean(data0)
    chronR=data0.-muR;
    r=reverse(chronR);

    fcast_length = T-tt-21-horizon; #length of forecast sample

    RVh = zeros(length(r)-horizon+1,1);
    for i=1:length(RVh)
        RVh[i]=mean(r[i:i+horizon-1])
    end

    M=Int.(2^(JMAX)*(floor((tt-maxAR)/(2^(JMAX)))-1))
    KMAX = M

#     horizon_forecast_Jcomp_tvp= zeros(fcast_length);
#     Error_Jcomp_tvp=zeros(fcast_length);
#     errs0_last=zeros(fcast_length);
#     forecasted_constant=zeros(fcast_length);
#     variance_scales=zeros(fcast_length,JMAX);
#     first_forecasted_beta_scales=zeros(fcast_length,JMAX);
#     vv =zeros(JMAX);

#     for ii=0:(fcast_length-1)
        est_sample_tvp = r[(fcast_length+horizon - ii): (fcast_length+tt+horizon-1-ii)]
        
        # new trend estimation
        x=ones(length(est_sample_tvp))
        R"aa <- tvOLS(as.matrix($x),$est_sample_tvp, bw = $kernel_width_for_const)$coefficients"
        tvp_const=convert(Array{Float64},R"aa")
        errs0=est_sample_tvp.-tvp_const
    
        
        (truncR,Rmat)=ARlags(errs0, maxAR);
        (alphaR_tvp,Eps_tvp) = IRFalpha_tvp_LLS_Epa(truncR,Rmat,maxAR,M,kernel_width_IRF)


        # Estimate the decomposition
        betaRj_tvp=[]
        rj_tvp=[]
        for j in 1:JMAX
            (betaR, _ , r0, _ , _)=IRFscale(tt,maxAR,alphaR_tvp,Eps_tvp,KMAX,j);
            push!(betaRj_tvp,betaR)
            push!(rj_tvp,r0)
        end

        
        ################# Estimate coefficients of regression model ################
        b_Jcomp_tvp=OLSestimator(est_sample_tvp[1:length(rj_tvp[JMAX])],[tvp_const[1:length(rj_tvp[JMAX])] hcat(rj_tvp...)])
   
        forecasts_ar =[]
        for_ar=[]
        rev=vec(reverse(tvp_const));

        ############# Forecast tvp_const that does not enter EWD ...############
        R"forecasts_ar<- tvReg::forecast(tvAR($rev,p=$AR_lag_forecast,bw = $kernel_width_forecast),n.ahead=$horizon)" 
    
        for_ar=convert(Array{Float64},R"forecasts_ar")  
        for_const = mean(for_ar)
    
        
        ######################## EWD FORECAST #####################
        
        # We forecast the sum horizon-step ahead 
        betaFj_tvp=[]
        rfj_tvp=[]
        for j in 1:JMAX
            (betaF,rf)=IRFforecast_horizon(tt,maxAR,alphaR_tvp,Eps_tvp,KMAX,j,horizon);
            push!(betaFj_tvp,betaF)
            push!(rfj_tvp,rf)
        end
    
        ######### forecasted constant x EWD forecast ##############
    
        RFmat_Jcomp_tvp=[for_const.*ones(size(hcat(rfj_tvp...))[1]) hcat(rfj_tvp...)]; # with const pred
        
        horizon_forecast_Jcomp_tvp=(horizon^(-1).*RFmat_Jcomp_tvp[1,:])'*[horizon*b_Jcomp_tvp[1]; b_Jcomp_tvp[2:end]];
        
        # error with J components
        Error_Jcomp_tvp=horizon_forecast_Jcomp_tvp-RVh[fcast_length-ii]

#     end
    return [horizon_forecast_Jcomp_tvp;Error_Jcomp_tvp]
end

function IRFalpha_tvp_LLS_Epa(y,x,maxAR,M,kernel_width)
    
    # New localized linear LS estimate
    R"""
    aa <- tvOLS($x,$y, bw = $kernel_width,tkernel = "Epa")$coefficients
    """ #,tkernel = "Epa"
    b_tvp_all=convert(Array{Float64},R"aa")

    Eta_tvp=zeros(length(y))
    for i=1:length(y)
        Eta_tvp[i] = y[i]-sum(b_tvp_all[i,:].*x[i,:]);
    end

    sigma2_tvp=(Eta_tvp'*Eta_tvp)./(length(y)-maxAR)
    sigma_tvp=sqrt.(sigma2_tvp)
    Eps_tvp=Eta_tvp./sigma_tvp;


    alphaR_tvp=zeros(M);
    alphaR_tvp[1]=sigma_tvp;

    for n=1:(length(alphaR_tvp)-1) 
        hstart=max(n-maxAR,0);
        temp=0;
        for h=hstart:n-1 
            temp=temp+alphaR_tvp[h+1]*b_tvp_all[1,n-h];    # choses last position of kernel!!!
        end
        alphaR_tvp[n+1]=temp;
    end

    return (alphaR_tvp,Eps_tvp)
end



function tvPersistenceplot(data,maxAR,JMAX,kernel_width_for_const,kernel_width_IRF)

    data0 = data;
    T=length(data0)
    muR=mean(data0)
    r = data0.-muR;

    tt=T

    M=Int.(2^(JMAX)*(floor((tt-maxAR)/(2^(JMAX)))-1))
    KMAX = M

    errs0=[]
            est_sample = r
                # new trend estimation
            x=ones(length(est_sample))
            R"aa <- tvOLS(as.matrix($x),$est_sample, bw = $kernel_width_for_const)$coefficients"
            tvp_const=convert(Array{Float64},R"aa")
            errs0=est_sample.-tvp_const

            (truncR,Rmat)=ARlags(errs0, maxAR);
            (alphaR_tvp,Eps_tvp) = IRFalpha_tvp_LLS_REVERSE(truncR,Rmat,maxAR,M,kernel_width_IRF)
    #         (alphaR_tvp,Eps_tvp) = IRFalpha_tvp_LLS(truncR,Rmat,maxAR,M,kernel_width_IRF)
            alphaR_tvp_all=Float64.(hcat(alphaR_tvp...)');

            betaRj_tvp_1=zeros(size(alphaR_tvp_all,1),JMAX)
            for i=1:size(alphaR_tvp_all,1)
                # Estimate the decomposition
                betaRj_tvp=[]
                rj_tvp=[]
                for j in 1:JMAX
                    (betaR, _ , r0, _ , _)=IRFscale(tt,maxAR,alphaR_tvp_all[i,:],Eps_tvp,KMAX,j);
                    push!(betaRj_tvp,betaR)
                    push!(rj_tvp,r0)
                end

                for iii=1:JMAX
                    betaRj_tvp_1[i,iii] = betaRj_tvp[iii][1]
                end    
            end
    return(betaRj_tvp_1)
end


function tvEWDforecast(data0,tt,horizon,maxAR,JMAX,kernel_width_for_const,kernel_width_IRF,kernel_width_forecast)
    
    AR_lag_forecast = 1
    T=length(data0)
    muR=mean(data0)
    chronR=data0.-muR;
    r=reverse(chronR);

    fcast_length = T-tt-21-horizon; #length of forecast sample

    RVh = zeros(length(r)-horizon+1,1);
    for i=1:length(RVh)
        RVh[i]=mean(r[i:i+horizon-1])
    end

    M=Int.(2^(JMAX)*(floor((tt-maxAR)/(2^(JMAX)))-1))
    KMAX = M

    horizon_forecast_Jcomp_tvp= zeros(fcast_length);
    Error_Jcomp_tvp=zeros(fcast_length);
    Actual=zeros(fcast_length);
    errs0_last=zeros(fcast_length);
    forecasted_constant=zeros(fcast_length);
    variance_scales=zeros(fcast_length,JMAX);
    first_forecasted_beta_scales=zeros(fcast_length,JMAX);
    vv =zeros(JMAX);

    for ii=0:(fcast_length-1)
        est_sample_tvp = r[(fcast_length+horizon - ii): (fcast_length+tt+horizon-1-ii)]
        
        # new trend estimation
        x=ones(length(est_sample_tvp))
        R"aa <- tvOLS(as.matrix($x),$est_sample_tvp, bw = $kernel_width_for_const)$coefficients"
        tvp_const=convert(Array{Float64},R"aa")
        errs0=est_sample_tvp.-tvp_const
    
        
        (truncR,Rmat)=ARlags(errs0, maxAR);
        (alphaR_tvp,Eps_tvp) = IRFalpha_tvp_LLS_Epa(truncR,Rmat,maxAR,M,kernel_width_IRF)


        # Estimate the decomposition
        betaRj_tvp=[]
        rj_tvp=[]
        for j in 1:JMAX
            (betaR, _ , r0, _ , _)=IRFscale(tt,maxAR,alphaR_tvp,Eps_tvp,KMAX,j);
            push!(betaRj_tvp,betaR)
            push!(rj_tvp,r0)
        end
        
        ################# Estimate coefficients of regression model ################
        b_Jcomp_tvp=OLSestimator(est_sample_tvp[1:length(rj_tvp[JMAX])],[tvp_const[1:length(rj_tvp[JMAX])] hcat(rj_tvp...)])
   
        forecasts_ar =[]
        for_ar=[]
        rev=vec(reverse(tvp_const));

         ############# Forecast tvp_const that does not enter EWD ...############
         R"forecasts_ar<- tvReg::forecast(tvAR($rev,p=$AR_lag_forecast,bw = $kernel_width_forecast),n.ahead=$horizon)" 
    
         for_ar=convert(Array{Float64},R"forecasts_ar")  
         for_const = mean(for_ar)
     
         
         ######################## EWD FORECAST #####################
        
        # We forecast the sum horizon-step ahead 
        betaFj_tvp=[]
        rfj_tvp=[]
        for j in 1:JMAX
            (betaF,rf)=IRFforecast_horizon(tt,maxAR,alphaR_tvp,Eps_tvp,KMAX,j,horizon);
            push!(betaFj_tvp,betaF)
            push!(rfj_tvp,rf)
        end
        
        RFmat_Jcomp_tvp=[for_const.*ones(size(hcat(rfj_tvp...))[1]) hcat(rfj_tvp...)]; # with const pred
        
        horizon_forecast_Jcomp_tvp[ii+1]=(horizon^(-1).*RFmat_Jcomp_tvp[1,:])'*[horizon*b_Jcomp_tvp[1]; b_Jcomp_tvp[2:end]];
        
        # error with J components
        Error_Jcomp_tvp[ii+1]=  (horizon_forecast_Jcomp_tvp[ii+1]-RVh[fcast_length-ii])
        Actual[ii+1]=RVh[fcast_length-ii]
        
    end
    return (horizon_forecast_Jcomp_tvp, Actual, Error_Jcomp_tvp)
end