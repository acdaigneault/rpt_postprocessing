
library(logspline)


data <- read.csv2("../grid_positions.csv", sep=",")
theo <- as.numeric(unlist(subset(data, select = c("counts_it10000000"))))
col <- as.numeric(unlist(subset(data, select = c("nomad_run5"))))
err = ((col) - theo)/theo*100



chsq.test(err) 

fitg <- fitdist(err, "gamma")
fitw <- fitdist(err, "weibull")
fitp <- fitdist(err, "pois", method="mme")
fite <- fitdist(err, "exp")
fitln <- fitdist(err, "lnorm")
fitn <- fitdist(err, "norm")
fitl <- fitdist(err, "logis")

fitn_MC <- fitdist(Err, "norm")
#summary(fitg)
#summary(fitw)
#summary(fitp)
#summary(fite)
#summary(fitln)
summary(fitn)
#summary(fitl)

plot(fitg, demp=TRUE)
plot(fitw, demp=TRUE)
#plot(fitp)
#plot(fite)
#plot(fitln)
plot(fitn)
#plot(fitl)
